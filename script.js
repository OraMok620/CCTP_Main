const video = document.getElementById('video');
const snapBtn = document.getElementById('snapBtn');
const resetBtn = document.getElementById('resetBtn');
const status = document.getElementById('status');
const previews = document.getElementById('previews');
const hiddenCanvas = document.getElementById('hiddenCanvas');
let capturedMats = [];
const MAX_WIDTH = 1024;

window.onload = () => {
    let checkCV = setInterval(() => {
        if (typeof cv !== 'undefined' && cv.Mat) {
            clearInterval(checkCV);
            onOpenCvReady();
        }
    }, 500);
};

function onOpenCvReady() {
    status.innerText = "OpenCV Ready. Tap video to focus, then take photo.";
    snapBtn.disabled = false;
    snapBtn.innerText = "📸 1. Take Object Photo";
    startCamera();
}

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' }, 
            audio: false 
        });
        video.srcObject = stream;
    } catch (err) {
        status.innerText = "Error: Camera access denied.";
    }
}

// --- 修正 2: 補回缺失的 alignImages 函數 ---
function alignImages(refMat, targetMat) {
    let orb = new cv.ORB();
    let kp1 = new cv.KeyPointVector(), kp2 = new cv.KeyPointVector();
    let des1 = new cv.Mat(), des2 = new cv.Mat();

    orb.detectAndCompute(refMat, new cv.Mat(), kp1, des1);
    orb.detectAndCompute(targetMat, new cv.Mat(), kp2, des2);

    let bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
    let matches = new cv.DMatchVector();
    bf.match(des1, des2, matches);

    let goodMatches = [];
    for (let i = 0; i < matches.size(); i++) goodMatches.push(matches.get(i));
    goodMatches.sort((a, b) => a.distance - b.distance);

    if (goodMatches.length < 15) throw "Alignment failed: Not enough features.";

    let srcPts = [], dstPts = [];
    for (let i = 0; i < Math.min(goodMatches.length, 50); i++) {
        let p2 = kp2.get(goodMatches[i].trainIdx).pt;
        let p1 = kp1.get(goodMatches[i].queryIdx).pt;
        srcPts.push(p2.x, p2.y);
        dstPts.push(p1.x, p1.y);
    }

    let srcMat = cv.matFromArray(srcPts.length / 2, 1, cv.CV_32FC2, srcPts);
    let dstMat = cv.matFromArray(dstPts.length / 2, 1, cv.CV_32FC2, dstPts);
    let h = cv.findHomography(srcMat, dstMat, cv.RANSAC, 3);
    
    let aligned = new cv.Mat();
    cv.warpPerspective(targetMat, aligned, h, new cv.Size(refMat.cols, refMat.rows));

    [orb, kp1, kp2, des1, des2, bf, matches, srcMat, dstMat, h].forEach(obj => obj.delete());
    return aligned;
}

snapBtn.onclick = () => {
    const ctx = hiddenCanvas.getContext('2d');
    let scale = Math.min(1, MAX_WIDTH / video.videoWidth);
    hiddenCanvas.width = video.videoWidth * scale;
    hiddenCanvas.height = video.videoHeight * scale;
    ctx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
    
    let mat = cv.imread(hiddenCanvas);
    capturedMats.push(mat);

    let thumb = document.createElement('canvas');
    thumb.className = 'thumb';
    cv.imshow(thumb, mat);
    previews.appendChild(thumb);

    if (capturedMats.length === 1) {
        status.innerText = "Object captured! Now focus on background and snap.";
        snapBtn.innerText = "📸 2. Take Background Photo";
    } else if (capturedMats.length === 2) {
        snapBtn.disabled = true;
        snapBtn.innerText = "Processing...";
        resetBtn.style.display = "inline-block";
        setTimeout(processStacking, 100);
    }
};

function setupDownload(canvas) {
    const downloadBtn = document.getElementById('downloadBtn');
    downloadBtn.style.display = "inline-block";
    downloadBtn.onclick = () => {
        const link = document.createElement('a');
        link.download = 'stacked-photo.jpg';
        link.href = canvas.toDataURL('image/jpeg', 0.95);
        link.click();
    };
}

video.onclick = async (e) => {
    const track = video.srcObject.getVideoTracks()[0];
    const capabilities = track.getCapabilities();
    
    const rect = video.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    // Check if the phone supports manual focus or single-shot autofocus
    const focusMode = capabilities.focusMode?.includes('manual') ? 'manual' : 
                     capabilities.focusMode?.includes('single-shot') ? 'single-shot' : 'continuous';

    try {
        await track.applyConstraints({
            advanced: [{
                focusMode: focusMode, // Try to lock the focus
                pointsOfInterest: [{x, y}]
            }]
        });
        status.innerText = `Focus Locked at ${Math.round(x*100)}%`;
        if (navigator.vibrate) navigator.vibrate(50);
    } catch (err) {
        console.warn("Focus lock failed:", err);
    }
};

function fastAlphaBlend(img1, img2, softMask) {
    let maskFloat = new cv.Mat();
    let invMaskFloat = new cv.Mat();
    let ones = cv.Mat.ones(softMask.rows, softMask.cols, cv.CV_32F);
    softMask.convertTo(maskFloat, cv.CV_32F, 1.0/255.0);
    cv.subtract(ones, maskFloat, invMaskFloat);
    let rgba1 = new cv.MatVector();
    let rgba2 = new cv.MatVector();
    cv.split(img1, rgba1);
    cv.split(img2, rgba2);
    let resultChannels = new cv.MatVector();
    for (let i = 0; i < 3; i++) {
        let chan1 = new cv.Mat(), chan2 = new cv.Mat(), blendedChan = new cv.Mat();
        rgba1.get(i).convertTo(chan1, cv.CV_32F);
        rgba2.get(i).convertTo(chan2, cv.CV_32F);
        cv.multiply(chan1, maskFloat, chan1);
        cv.multiply(chan2, invMaskFloat, chan2);
        cv.add(chan1, chan2, blendedChan);
        blendedChan.convertTo(blendedChan, cv.CV_8U);
        resultChannels.push_back(blendedChan);
        chan1.delete(); chan2.delete(); blendedChan.delete();
    }
    resultChannels.push_back(rgba1.get(3));
    let res = new cv.Mat();
    cv.merge(resultChannels, res);
    [maskFloat, invMaskFloat, ones, rgba1, rgba2, resultChannels].forEach(m => { if(m.delete) m.delete(); });
    return res;
}

function processStacking() {
    try {
        status.innerText = "Aligning and Cleaning Masks...";
        let img1 = capturedMats[0];
        let img2_raw = capturedMats[1];
        let img2 = alignImages(img1, img2_raw);
        let gray1 = new cv.Mat(), gray2 = new cv.Mat();
        cv.cvtColor(img1, gray1, cv.COLOR_RGBA2GRAY);
        cv.cvtColor(img2, gray2, cv.COLOR_RGBA2GRAY);
        let lap1 = new cv.Mat(), lap2 = new cv.Mat(), smoothGray1 = new cv.Mat(), smoothGray2 = new cv.Mat();
        cv.GaussianBlur(gray1, smoothGray1, new cv.Size(5, 5), 0);
        cv.GaussianBlur(gray2, smoothGray2, new cv.Size(5, 5), 0);
        cv.Laplacian(smoothGray1, lap1, cv.CV_64F);
        cv.Laplacian(smoothGray2, lap2, cv.CV_64F);
        let abs1 = new cv.Mat(), abs2 = new cv.Mat();
        cv.convertScaleAbs(lap1, abs1);
        cv.convertScaleAbs(lap2, abs2);
        let mask = new cv.Mat();
        cv.compare(abs1, abs2, mask, cv.CMP_GT);
        let k = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
        cv.morphologyEx(mask, mask, cv.MORPH_OPEN, k);
        let softMask = new cv.Mat();
        cv.GaussianBlur(mask, softMask, new cv.Size(31, 31), 0);
        let result = fastAlphaBlend(img1, img2, softMask);
        let resCanvas = document.getElementById('resCanvas');
        if(!resCanvas) {
            resCanvas = document.createElement('canvas');
            resCanvas.id = 'resCanvas';
            document.body.appendChild(resCanvas);
        }
        cv.imshow('resCanvas', result);
        setupDownload(resCanvas);
        status.innerText = "🎉 Success! Quality Optimized.";
        [img2, gray1, gray2, smoothGray1, smoothGray2, lap1, lap2, abs1, abs2, mask, softMask, result, k].forEach(m => {
            if(m && m.delete) m.delete();
        });
    } catch (err) {
        console.error(err);
        status.innerText = "Error: " + err;
        snapBtn.disabled = false;
        snapBtn.innerText = "Retry Capture";
    }
}

resetBtn.onclick = () => location.reload();