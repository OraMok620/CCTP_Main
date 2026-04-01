//Notes 1: document.getElementByID is used to access HTML elements by their ID, allowing us to manipulate them in JavaScript.
const video = document.getElementById('video');
const snapBtn = document.getElementById('snapBtn');
const resetBtn = document.getElementById('resetBtn');
const status = document.getElementById('status');
const previews = document.getElementById('previews');
const hiddenCanvas = document.getElementById('hiddenCanvas');

//Notes 2: capturedMats is an array that will hold the OpenCV Mat objects representing the captured images. 
//Notes 3:MAX_WIDTH is a constant that limits the width of the captured images to optimize performance and memory usage.
let capturedMats = [];
const MAX_WIDTH = 1024;

//Notes 4: The window.onload function checks for the availability of OpenCV.js and initializes the application once it's ready.
window.onload = () => {
    //Notes 5: The setInterval function is used to periodically check if the OpenCV library has loaded and is ready to use.
    let checkCV = setInterval(() => {
        if (typeof cv !== 'undefined' && cv.Mat) {
            clearInterval(checkCV);
            onOpenCvReady();
        }
    }, 500);//500 milliseconds is a reasonable interval to check for the library's readiness without causing excessive CPU usage.
};

//Notes 6: The onOpenCvReady function is called once OpenCV.js is loaded. 
// It updates the status message, enables the snap button, and starts the camera.
function onOpenCvReady() {
    //Notes 7: The status message provides further instructions.
    status.innerText = "Tap screen to focus.";
    snapBtn.disabled = false;
    snapBtn.innerText = "Take photo for Object focused";
    startCamera();
}

//Notes 8: The startCamera function uses the MediaDevices API to access the user's camera and stream it to the video element.
async function startCamera() {
    try {
        //Notes 9: The getUserMedia function is called with constraints to access the camera and disable audio.
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' }, //Notes 10: There are 2 options for facingMode: 'user' for front camera and 'environment' for rear camera.
            audio: false 
        });
        video.srcObject = stream;
    } catch (err) {
        status.innerText = "Error: Camera access denied."; //Notes 11: If the user denies camera access or if there's an error, the catch block updates the status message to inform the user. Showing a user-friendly error message helps improve the user experience and guides them on how to proceed. (ethical consideration)
    }
}

//Notes 12: The exposureCompensate function adjusts the exposure of the second image to match the first image if there's a significant difference in brightness.
function exposureCompensate(img1, img2) {
    //Notes 13: We create two new Mat objects, gray1 and gray2, to hold the grayscale versions of the input images. Converting to grayscale simplifies the process of calculating the average brightness (mean) of each image.
    let gray1 = new cv.Mat(), gray2 = new cv.Mat(); 
    //Notes 14: The cvtColor function is used to convert the input images from RGBA color space to grayscale. This is necessary because we want to analyze the brightness of the images, and working with grayscale simplifies this process.
    cv.cvtColor(img1, gray1, cv.COLOR_RGBA2GRAY); 
    cv.cvtColor(img2, gray2, cv.COLOR_RGBA2GRAY);
    
    //Notes 15: The mean function calculates the average pixel intensity of the grayscale images. We take the first element of the returned array (mean[0]) because it contains the average brightness value for the single channel in the grayscale image.
    let mean1 = cv.mean(gray1)[0];
    let mean2 = cv.mean(gray2)[0];
    
    //Notes 16: We check if both mean values are greater than 0 (to avoid division by zero) and if the absolute difference between the two means is greater than 10. 
    // If these conditions are met, it indicates a significant exposure difference between the two images.
    if (mean1 > 0 && mean2 > 0 && Math.abs(mean1 - mean2) > 10) {
        let ratio = mean1 / mean2;
        let adjusted = new cv.Mat();
        img2.convertTo(adjusted, -1, ratio, 0);
        gray1.delete();
        gray2.delete();
        return adjusted;
    }
    
    //Notes 17: If the exposure difference is not significant, we simply clean up the grayscale Mat objects and return a clone of the second image without any adjustments.
    gray1.delete();
    gray2.delete();
    return img2.clone();
}

// --- Motion Detection ---
function detectMotion(img1, img2) {
    let gray1 = new cv.Mat(), gray2 = new cv.Mat();
    cv.cvtColor(img1, gray1, cv.COLOR_RGBA2GRAY);
    cv.cvtColor(img2, gray2, cv.COLOR_RGBA2GRAY);
    
    // Resize for faster optical flow
    let small1 = new cv.Mat(), small2 = new cv.Mat();
    cv.resize(gray1, small1, new cv.Size(320, 240));
    cv.resize(gray2, small2, new cv.Size(320, 240));
    
    let flow = new cv.Mat();
    cv.calcOpticalFlowFarneback(small1, small2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    
    let mag = new cv.Mat();
    let magParts = new cv.MatVector();
    cv.split(flow, magParts);
    cv.magnitude(magParts.get(0), magParts.get(1), mag);
    
    let avgMotion = cv.mean(mag)[0];
    
    // Cleanup
    [gray1, gray2, small1, small2, flow, mag, magParts].forEach(m => {
        if (m && m.delete) m.delete();
    });
    
    return avgMotion < 2.5; // Threshold for acceptable motion
}

// --- Robust Alignment with Fallback Strategies ---
function alignWithORB(refMat, targetMat) {
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

    if (goodMatches.length < 15) throw "Not enough ORB features";

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

    [orb, kp1, kp2, des1, des2, bf, matches, srcMat, dstMat, h].forEach(obj => {
        if (obj && obj.delete) obj.delete();
    });
    return aligned;
}

function alignWithAKAZE(refMat, targetMat) {
    let akaze = cv.AKAZE.create();
    let kp1 = new cv.KeyPointVector(), kp2 = new cv.KeyPointVector();
    let des1 = new cv.Mat(), des2 = new cv.Mat();

    akaze.detectAndCompute(refMat, new cv.Mat(), kp1, des1);
    akaze.detectAndCompute(targetMat, new cv.Mat(), kp2, des2);

    let bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
    let matches = new cv.DMatchVector();
    bf.match(des1, des2, matches);

    let goodMatches = [];
    for (let i = 0; i < matches.size(); i++) goodMatches.push(matches.get(i));
    goodMatches.sort((a, b) => a.distance - b.distance);

    if (goodMatches.length < 10) throw "Not enough AKAZE features";

    let srcPts = [], dstPts = [];
    for (let i = 0; i < Math.min(goodMatches.length, 40); i++) {
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

    [akaze, kp1, kp2, des1, des2, bf, matches, srcMat, dstMat, h].forEach(obj => {
        if (obj && obj.delete) obj.delete();
    });
    return aligned;
}

function alignWithTemplateMatching(refMat, targetMat) {
    let grayRef = new cv.Mat(), grayTarget = new cv.Mat();
    cv.cvtColor(refMat, grayRef, cv.COLOR_RGBA2GRAY);
    cv.cvtColor(targetMat, grayTarget, cv.COLOR_RGBA2GRAY);
    
    // Simple translation-only alignment using phase correlation
    let result = new cv.Mat();
    cv.phaseCorrelate(grayRef, grayTarget);
    
    let translationMat = cv.matFromArray(2, 3, cv.CV_64F, [1, 0, 0, 0, 1, 0]);
    let aligned = new cv.Mat();
    cv.warpAffine(targetMat, aligned, translationMat, new cv.Size(refMat.cols, refMat.rows));
    
    [grayRef, grayTarget, translationMat].forEach(m => {
        if (m && m.delete) m.delete();
    });
    
    return aligned;
}

function robustAlignment(refMat, targetMat) {
    let strategies = [
        () => alignWithORB(refMat, targetMat),
        () => alignWithAKAZE(refMat, targetMat),
        () => alignWithTemplateMatching(refMat, targetMat)
    ];
    
    let lastError = null;
    for (let strategy of strategies) {
        try {
            return strategy();
        } catch (e) {
            lastError = e;
            console.warn("Strategy failed:", e);
        }
    }
    throw new Error(`All alignment strategies failed: ${lastError}`);
}

// --- Laplacian Pyramid Blending ---
function buildLaplacianPyramid(img, levels) {
    let pyramid = [];
    let current = img.clone();
    
    for (let i = 0; i < levels; i++) {
        let down = new cv.Mat();
        cv.pyrDown(current, down);
        let up = new cv.Mat();
        cv.pyrUp(down, up, new cv.Size(current.cols, current.rows));
        
        let lap = new cv.Mat();
        cv.subtract(current, up, lap);
        pyramid.push(lap);
        
        current.delete();
        current = down;
        up.delete();
    }
    pyramid.push(current);
    
    return pyramid;
}

function reconstructFromPyramid(pyramid) {
    let current = pyramid[pyramid.length - 1].clone();
    
    for (let i = pyramid.length - 2; i >= 0; i--) {
        let up = new cv.Mat();
        cv.pyrUp(current, up, new cv.Size(pyramid[i].cols, pyramid[i].rows));
        let reconstructed = new cv.Mat();
        cv.add(up, pyramid[i], reconstructed);
        current.delete();
        up.delete();
        current = reconstructed;
    }
    
    return current;
}

function laplacianPyramidBlend(img1, img2, mask, levels = 5) {
    // Build Gaussian pyramid for mask
    let maskPyramid = [];
    let currentMask = mask.clone();
    for (let i = 0; i < levels; i++) {
        maskPyramid.push(currentMask.clone());
        let down = new cv.Mat();
        cv.pyrDown(currentMask, down);
        currentMask.delete();
        currentMask = down;
    }
    maskPyramid.push(currentMask);
    
    // Build Laplacian pyramids for images
    let lapPyr1 = buildLaplacianPyramid(img1, levels);
    let lapPyr2 = buildLaplacianPyramid(img2, levels);
    
    // Blend each level
    let blendedPyramid = [];
    for (let i = 0; i <= levels; i++) {
        let maskFloat = new cv.Mat();
        maskPyramid[i].convertTo(maskFloat, cv.CV_32F, 1.0/255.0);
        
        let invMaskFloat = new cv.Mat();
        let ones = cv.Mat.ones(maskPyramid[i].rows, maskPyramid[i].cols, cv.CV_32F);
        cv.subtract(ones, maskFloat, invMaskFloat);
        
        let blended = new cv.Mat();
        let lap1Float = new cv.Mat(), lap2Float = new cv.Mat();
        lapPyr1[i].convertTo(lap1Float, cv.CV_32F);
        lapPyr2[i].convertTo(lap2Float, cv.CV_32F);
        
        let temp1 = new cv.Mat(), temp2 = new cv.Mat();
        cv.multiply(lap1Float, maskFloat, temp1);
        cv.multiply(lap2Float, invMaskFloat, temp2);
        cv.add(temp1, temp2, blended);
        blended.convertTo(blended, cv.CV_8U);
        
        blendedPyramid.push(blended);
        
        [maskFloat, invMaskFloat, ones, lap1Float, lap2Float, temp1, temp2].forEach(m => {
            if (m && m.delete) m.delete();
        });
    }
    
    // Reconstruct
    let result = reconstructFromPyramid(blendedPyramid);
    
    // Cleanup pyramids
    [...lapPyr1, ...lapPyr2, ...maskPyramid, ...blendedPyramid].forEach(m => {
        if (m && m.delete) m.delete();
    });
    
    return result;
}

// --- Edge-Aware Mask Refinement ---
function refineMask(mask, img) {
    let gray = new cv.Mat();
    cv.cvtColor(img, gray, cv.COLOR_RGBA2GRAY);
    
    // Apply guided filter-like smoothing
    let refined = new cv.Mat();
    cv.GaussianBlur(mask, refined, new cv.Size(3, 3), 0);
    
    // Use gradient information to tighten mask
    let gradX = new cv.Mat(), gradY = new cv.Mat();
    cv.Sobel(gray, gradX, cv.CV_32F, 1, 0);
    cv.Sobel(gray, gradY, cv.CV_32F, 0, 1);
    let magnitude = new cv.Mat();
    cv.magnitude(gradX, gradY, magnitude);
    
    let edges = new cv.Mat();
    cv.threshold(magnitude, edges, 30, 255, cv.THRESH_BINARY);
    
    // Dilate edges slightly to include edge regions in mask
    let kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
    cv.dilate(edges, edges, kernel);
    
    // Combine mask with edges (ensure edges are included)
    let edgeMask = new cv.Mat();
    edges.convertTo(edgeMask, cv.CV_8U);
    cv.bitwise_or(refined, edgeMask, refined);
    
    // Cleanup
    [gray, gradX, gradY, magnitude, edges, kernel, edgeMask].forEach(m => {
        if (m && m.delete) m.delete();
    });
    
    return refined;
}

// --- Main Processing with Memory Management ---
function processStacking() {
    let matsToObjectDelete = []; // Track everything for memory safety
    
    try {
        status.innerText = "⏳ Hardening images and aligning...";
        
        let img1 = capturedMats[0];
        let img2_raw = capturedMats[1];

        // --- STEP 1: SIZE & TYPE SANITIZATION ---
        // Ensure both are RGBA (4 channels)
        if (img1.channels() !== 4) cv.cvtColor(img1, img1, cv.COLOR_RGB2RGBA);
        if (img2_raw.channels() !== 4) cv.cvtColor(img2_raw, img2_raw, cv.COLOR_RGB2RGBA);

        // --- STEP 2: ALIGNMENT ---
        let img2 = alignImages(img1, img2_raw);
        matsToObjectDelete.push(img2);

        // DOUBLE CHECK: If alignImages returned a different size, resize it back
        if (img2.rows !== img1.rows || img2.cols !== img1.cols) {
            let fixedImg2 = new cv.Mat();
            cv.resize(img2, fixedImg2, new cv.Size(img1.cols, img1.rows));
            img2.delete();
            img2 = fixedImg2;
            matsToObjectDelete.push(img2);
        }

        // --- STEP 3: SHARPNESS DETECTION ---
        let gray1 = new cv.Mat(), gray2 = new cv.Mat();
        cv.cvtColor(img1, gray1, cv.COLOR_RGBA2GRAY);
        cv.cvtColor(img2, gray2, cv.COLOR_RGBA2GRAY);
        matsToObjectDelete.push(gray1, gray2);

        let lap1 = new cv.Mat(), lap2 = new cv.Mat();
        cv.Laplacian(gray1, lap1, cv.CV_64F);
        cv.Laplacian(gray2, lap2, cv.CV_64F);
        matsToObjectDelete.push(lap1, lap2);

        let abs1 = new cv.Mat(), abs2 = new cv.Mat();
        cv.convertScaleAbs(lap1, abs1);
        cv.convertScaleAbs(lap2, abs2);
        matsToObjectDelete.push(abs1, abs2);

        // --- STEP 4: MASK GENERATION ---
        let mask = new cv.Mat();
        cv.compare(abs1, abs2, mask, cv.CMP_GT); // This is where 6704032 usually happens
        matsToObjectDelete.push(mask);

        let k = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(5, 5));
        cv.morphologyEx(mask, mask, cv.MORPH_OPEN, k);
        matsToObjectDelete.push(k);

        let softMask = new cv.Mat();
        cv.GaussianBlur(mask, softMask, new cv.Size(31, 31), 0);
        matsToObjectDelete.push(softMask);

        // --- STEP 5: BLENDING ---
        let result = fastAlphaBlend(img1, img2, softMask);
        
        cv.imshow('resCanvas', result);
        setupDownload(document.getElementById('resCanvas'));
        result.delete();
        
        status.innerText = "🎉 Success! Stable processing complete.";

    } catch (err) {
        console.error("OpenCV Error:", err);
        status.innerText = "❌ Process failed. Try to hold steadier.";
    } finally {
        // Cleanup all temporary mats
        matsToObjectDelete.forEach(m => { if(m && !m.isDeleted()) m.delete(); });
    }
}

// --- Fast Alpha Blending ---
function fastAlphaBlend(img1, img2, softMask) {
    let mats = [];
    
    // Ensure mask is 1-channel
    let singleChannelMask = new cv.Mat();
    if (softMask.channels() > 1) {
        cv.cvtColor(softMask, singleChannelMask, cv.COLOR_RGBA2GRAY);
    } else {
        singleChannelMask = softMask.clone();
    }
    mats.push(singleChannelMask);

    let maskFloat = new cv.Mat();
    let invMaskFloat = new cv.Mat();
    let ones = cv.Mat.ones(img1.rows, img1.cols, cv.CV_32F);
    
    singleChannelMask.convertTo(maskFloat, cv.CV_32F, 1.0/255.0);
    cv.subtract(ones, maskFloat, invMaskFloat);
    mats.push(maskFloat, invMaskFloat, ones);

    let rgba1 = new cv.MatVector();
    let rgba2 = new cv.MatVector();
    cv.split(img1, rgba1);
    cv.split(img2, rgba2);

    let resultChannels = new cv.MatVector();

    for (let i = 0; i < 3; i++) {
        let c1 = new cv.Mat(), c2 = new cv.Mat(), blended = new cv.Mat();
        rgba1.get(i).convertTo(c1, cv.CV_32F);
        rgba2.get(i).convertTo(c2, cv.CV_32F);

        cv.multiply(c1, maskFloat, c1);
        cv.multiply(c2, invMaskFloat, c2);
        cv.add(c1, c2, blended);

        blended.convertTo(blended, cv.CV_8U);
        resultChannels.push_back(blended);
        c1.delete(); c2.delete(); blended.delete();
    }
    
    resultChannels.push_back(rgba1.get(3)); // Alpha channel
    let res = new cv.Mat();
    cv.merge(resultChannels, res);

    // Cleanup
    rgba1.delete(); rgba2.delete(); resultChannels.delete();
    mats.forEach(m => m.delete());

    return res;
}

// --- UI Event Handlers ---
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
        status.innerText = "Object captured! Now focus on background.";
        snapBtn.innerText = "Take Background Photo";
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

    const focusMode = capabilities.focusMode?.includes('manual') ? 'manual' : 
                     capabilities.focusMode?.includes('single-shot') ? 'single-shot' : 'continuous';

    try {
        await track.applyConstraints({
            advanced: [{
                focusMode: focusMode,
                pointsOfInterest: [{x, y}]
            }]
        });
        status.innerText = `Focus Locked at ${Math.round(x*100)}%`;
        if (navigator.vibrate) navigator.vibrate(50);
    } catch (err) {
        console.warn("Focus lock failed:", err);
    }
};

resetBtn.onclick = () => location.reload();