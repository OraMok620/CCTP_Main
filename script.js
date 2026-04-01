const video = document.getElementById('video');
const snapBtn = document.getElementById('snapBtn');
const resetBtn = document.getElementById('resetBtn');
const status = document.getElementById('status');
const previews = document.getElementById('previews');
const hiddenCanvas = document.getElementById('hiddenCanvas');
let capturedMats = [];
const MAX_WIDTH = 1024;

video.onclick = async (e) => {
    const track = video.srcObject.getVideoTracks()[0];
    const capabilities = track.getCapabilities();
    const rect = video.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    if (capabilities.focusMode && capabilities.focusMode.includes('continuous')) {
        try {
            await track.applyConstraints({
                advanced: [{ pointsOfInterest: [{x, y}] }]
            });
            status.innerText = `Focusing at ${Math.round(x*100)}, ${Math.round(y*100)}...`;
            if (navigator.vibrate) navigator.vibrate(50);
        } catch (err) {
            console.warn("Focus constraints not supported", err);
        }
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
        let chan1 = new cv.Mat();
        let chan2 = new cv.Mat();
        let blendedChan = new cv.Mat();
        rgba1.get(i).convertTo(chan1, cv.CV_32F);
        rgba2.get(i).convertTo(chan2, cv.CV_32F);
        cv.multiply(chan1, maskFloat, chan1);
        cv.multiply(chan2, invMaskFloat, chan2);
        cv.add(chan1, chan2, blendedChan);
        blendedChan.convertTo(blendedChan, cv.CV_8U);
        resultChannels.push_back(blendedChan);
        chan1.delete(); chan2.delete();
    }
    resultChannels.push_back(rgba1.get(3));
    let res = new cv.Mat();
    cv.merge(resultChannels, res);
    maskFloat.delete(); invMaskFloat.delete(); ones.delete();
    rgba1.delete(); rgba2.delete(); resultChannels.delete();
    return res;
}

function processStacking() {
    try {
        status.innerText = "Aligning and Cleaning Masks...";
        let img1 = capturedMats[0];
        let img2_raw = capturedMats[1];
        let img2 = alignImages(img1, img2_raw); // 使用你之前寫的 ORB alignImages
        let gray1 = new cv.Mat();
        let gray2 = new cv.Mat();
        cv.cvtColor(img1, gray1, cv.COLOR_RGBA2GRAY);
        cv.cvtColor(img2, gray2, cv.COLOR_RGBA2GRAY);
        let lap1 = new cv.Mat();
        let lap2 = new cv.Mat();
        let smoothGray1 = new cv.Mat();
        let smoothGray2 = new cv.Mat();
        cv.GaussianBlur(gray1, smoothGray1, new cv.Size(5, 5), 0);
        cv.GaussianBlur(gray2, smoothGray2, new cv.Size(5, 5), 0);
        cv.Laplacian(smoothGray1, lap1, cv.CV_64F);
        cv.Laplacian(smoothGray2, lap2, cv.CV_64F);
        let abs1 = new cv.Mat();
        let abs2 = new cv.Mat();
        cv.convertScaleAbs(lap1, abs1);
        cv.convertScaleAbs(lap2, abs2);
        let mask = new cv.Mat();
        cv.compare(abs1, abs2, mask, cv.CMP_GT);
        let k = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
        cv.morphologyEx(mask, mask, cv.MORPH_OPEN, k);
        let softMask = new cv.Mat();
        cv.GaussianBlur(mask, softMask, new cv.Size(31, 31), 0);
        let result = fastAlphaBlend(img1, img2, softMask);
        cv.imshow('resCanvas', result);
        setupDownload(document.getElementById('resCanvas'));
        status.innerText = "🎉 Success! Quality Optimized.";
        [img2, gray1, gray2, smoothGray1, smoothGray2, lap1, lap2, abs1, abs2, mask, softMask, result, k].forEach(m => {
            if(m && m.delete) m.delete();
        });
    } catch (err) {
        console.error(err);
        status.innerText = "Error: " + err;
    }
}