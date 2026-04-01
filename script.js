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
    status.innerText = "Please point at the 'object' and take a photo.";
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
        status.innerText = "Error: Unable to access camera";
    }
}

// --- NEW: ALIGNMENT FUNCTION ---
function alignImages(refMat, targetMat) {
    let orb = new cv.ORB();
    let keypoints1 = new cv.KeyPointVector();
    let keypoints2 = new cv.KeyPointVector();
    let descriptors1 = new cv.Mat();
    let descriptors2 = new cv.Mat();

    // Find keypoints and descriptors
    orb.detectAndCompute(refMat, new cv.Mat(), keypoints1, descriptors1);
    orb.detectAndCompute(targetMat, new cv.Mat(), keypoints2, descriptors2);

    // Match features
    let bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
    let matches = new cv.DMatchVector();
    bf.match(descriptors1, descriptors2, matches);

    // Sort matches by distance
    let goodMatches = [];
    for (let i = 0; i < matches.size(); i++) {
        goodMatches.push(matches.get(i));
    }
    goodMatches.sort((a, b) => a.distance - b.distance);

    // Take top 50 matches to find transform
    let srcPoints = [];
    let dstPoints = [];
    for (let i = 0; i < Math.min(goodMatches.length, 50); i++) {
        srcPoints.push(keypoints2.get(goodMatches[i].trainIdx).pt.x);
        srcPoints.push(keypoints2.get(goodMatches[i].trainIdx).pt.y);
        dstPoints.push(keypoints1.get(goodMatches[i].queryIdx).pt.x);
        dstPoints.push(keypoints1.get(goodMatches[i].queryIdx).pt.y);
    }

    let srcMat = cv.matFromArray(srcPoints.length / 2, 1, cv.CV_32FC2, srcPoints);
    let dstMat = cv.matFromArray(dstPoints.length / 2, 1, cv.CV_32FC2, dstPoints);

    // Find homography matrix
    let h = cv.findHomography(srcMat, dstMat, cv.RANSAC, 3);
    let aligned = new cv.Mat();
    let size = new cv.Size(refMat.cols, refMat.rows);

    // Warp target image to match reference
    cv.warpPerspective(targetMat, aligned, h, size);

    // Cleanup
    orb.delete(); keypoints1.delete(); keypoints2.delete(); descriptors1.delete(); 
    descriptors2.delete(); bf.delete(); matches.delete(); h.delete(); 
    srcMat.delete(); dstMat.delete();

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
        status.innerText = "Object captured! Now hold steady and take Background photo.";
        snapBtn.innerText = "📸 2. Take Background Photo";
    } else if (capturedMats.length === 2) {
        snapBtn.disabled = true;
        snapBtn.innerText = "Aligning & Stacking...";
        resetBtn.style.display = "inline-block";
        setTimeout(processStacking, 100);
    }
};

function processStacking() {
    try {
        let img1 = capturedMats[0]; // Reference
        let img2_raw = capturedMats[1]; // Target to be aligned

        // 1. ALIGNMENT
        let img2 = alignImages(img1, img2_raw);

        // 2. PREPARE GRAYSCALE
        let gray1 = new cv.Mat();
        let gray2 = new cv.Mat();
        cv.cvtColor(img1, gray1, cv.COLOR_RGBA2GRAY);
        cv.cvtColor(img2, gray2, cv.COLOR_RGBA2GRAY);

        // 3. FIND SHARPNESS (Laplacian)
        let lap1 = new cv.Mat();
        let lap2 = new cv.Mat();
        cv.Laplacian(gray1, lap1, cv.CV_64F);
        cv.Laplacian(gray2, lap2, cv.CV_64F);

        let abs1 = new cv.Mat();
        let abs2 = new cv.Mat();
        cv.convertScaleAbs(lap1, abs1);
        cv.convertScaleAbs(lap2, abs2);

        // 4. CREATE SOFT MASK (Feathering)
        let mask = new cv.Mat();
        cv.compare(abs1, abs2, mask, cv.CMP_GT);
        
        let softMask = new cv.Mat();
        let ksize = new cv.Size(31, 31); // Large blur for smooth transition
        cv.GaussianBlur(mask, softMask, ksize, 0);
        
        // 5. ALPHA BLENDING
        let maskFloat = new cv.Mat();
        softMask.convertTo(maskFloat, cv.CV_32F, 1.0/255.0);

        let result = new cv.Mat(img1.rows, img1.cols, img1.type());
        
        // Manual Blend for each pixel: img1*mask + img2*(1-mask)
        for (let i = 0; i < img1.rows; i++) {
            for (let j = 0; j < img1.cols; j++) {
                let m = maskFloat.floatAt(i, j);
                let p1 = img1.ucharPtr(i, j);
                let p2 = img2.ucharPtr(i, j);
                let res = result.ucharPtr(i, j);
                
                res[0] = p1[0] * m + p2[0] * (1 - m); // R
                res[1] = p1[1] * m + p2[1] * (1 - m); // G
                res[2] = p1[2] * m + p2[2] * (1 - m); // B
                res[3] = 255; // Alpha
            }
        }

        // 6. SHOW & DOWNLOAD
        let resCanvas = document.getElementById('resCanvas') || document.createElement('canvas');
        resCanvas.id = 'resCanvas';
        if (!resCanvas.parentElement) document.body.appendChild(resCanvas);
        cv.imshow(resCanvas, result);

        setupDownload(resCanvas);
        
        status.innerText = "🎉 High Quality Stack Complete!";

        // Cleanup
        [img2, gray1, gray2, lap1, lap2, abs1, abs2, mask, softMask, maskFloat, result].forEach(m => m.delete());
        
    } catch (err) {
        console.error(err);
        status.innerText = "Failed: " + err;
    }
}

function setupDownload(canvas) {
    const downloadBtn = document.getElementById('downloadBtn');
    downloadBtn.style.display = "inline-block";
    downloadBtn.onclick = () => {
        const link = document.createElement('a');
        link.download = 'pro-stack.jpg';
        link.href = canvas.toDataURL('image/jpeg', 0.95); 
        link.click();
    };
}

resetBtn.onclick = () => location.reload(); 