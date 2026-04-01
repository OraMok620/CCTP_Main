// Notes 1: document.getElementById is used to access HTML elements by their ID.
const video = document.getElementById('video');
const snapBtn = document.getElementById('snapBtn');
const resetBtn = document.getElementById('resetBtn');
const status = document.getElementById('status');
const previews = document.getElementById('previews');
const hiddenCanvas = document.getElementById('hiddenCanvas');

// Notes 2: capturedMats holds the OpenCV Mat objects for the captured images.
let capturedMats = [];
const MAX_WIDTH = 1024;

// Notes 3: Wait for OpenCV.js to load before initializing the app.
window.onload = () => {
    let checkCV = setInterval(() => {
        if (typeof cv !== 'undefined' && cv.Mat) {
            clearInterval(checkCV);
            onOpenCvReady();
        }
    }, 500);
};

// Notes 4: onOpenCvReady initializes the app once OpenCV is ready, enabling the snap button and starting the camera.
function onOpenCvReady() {
    status.innerText = "OpenCV Ready. Take your two photos.";
    snapBtn.disabled = false;
    snapBtn.innerText = "📸 1. Take First Photo";
    startCamera();
}

// Notes 5: startCamera uses the MediaDevices API to access the user's camera, requesting the environment-facing camera for better quality.
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' }, //Additional hint: There are 2 facing modes: 'user' (front) and 'environment' (back / rear). 
            audio: false  // We don't need audio for this app.
        });
        video.srcObject = stream;
    } catch (err) {
        status.innerText = "Error: Camera access denied."; // Message shown if user denies camera access or if there's an error.
    }
}

//Notes 6: ORB is a popular feature detection and matching algorithm in computer vision. 
//It is used to find key points and descriptors in images, which can then be used for tasks like image alignment, object recognition, and more. 
//In this code, we use ORB to align two images by finding matching key points between them and computing a homography transformation.
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
    if (goodMatches.length < 8) throw "ORB Failed";

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
    [orb, kp1, kp2, des1, des2, bf, matches, srcMat, dstMat, h].forEach(obj => { if (obj && obj.delete) obj.delete(); });
    return aligned;
}

// Notes 7: AKAZE is another feature detection and matching algorithm that is designed to be faster and more efficient than ORB, especially for larger images.
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

    if (goodMatches.length < 4) throw "AKAZE Failed";

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

    [akaze, kp1, kp2, des1, des2, bf, matches, srcMat, dstMat, h].forEach(obj => { if (obj && obj.delete) obj.delete(); });
    return aligned;
}

// Notes 8: Template Matching is a technique in computer vision used to find a smaller image (template) within a larger image (target). 
// It works by sliding the template over the target image and comparing the template with the overlapping region of the target image using a similarity metric. 
// In this code, we use normalized cross-correlation (cv.TM_CCOEFF_NORMED) as the similarity metric to find the best match of the reference image within the target image, and then align it accordingly.   
function alignWithTemplateMatching(refMat, targetMat) {
    let grayRef = new cv.Mat(), grayTarget = new cv.Mat();
    cv.cvtColor(refMat, grayRef, cv.COLOR_RGBA2GRAY);
    cv.cvtColor(targetMat, grayTarget, cv.COLOR_RGBA2GRAY);
    let result = new cv.Mat();
    let mask = new cv.Mat();
    cv.matchTemplate(grayTarget, grayRef, result, cv.TM_CCOEFF_NORMED, mask);
    let minMax = cv.minMaxLoc(result);
    let maxLoc = minMax.maxLoc;
    let M = cv.matFromArray(2, 3, cv.CV_64F, [1, 0, -maxLoc.x, 0, 1, -maxLoc.y]);
    let aligned = new cv.Mat();
    cv.warpAffine(targetMat, aligned, M, new cv.Size(refMat.cols, refMat.rows));

    [grayRef, grayTarget, result, mask, M].forEach(m => m.delete());
    return aligned;
}

// Notes 9: robustAlignment is a function that tries multiple alignment strategies in sequence (ORB, AKAZE, Template Matching) and falls back to the raw capture if all else fails.
function robustAlignment(refMat, targetMat) {
    let strategies = [
        () => alignWithORB(refMat, targetMat),
        () => alignWithAKAZE(refMat, targetMat),
        () => alignWithTemplateMatching(refMat, targetMat),
        () => { 
            console.warn("Using raw capture fallback."); 
            return targetMat.clone(); 
        }
    ];
    for (let strategy of strategies) {
        try {
            return strategy();
        } catch (e) {
            console.warn("Next strategy...");
        }
    }
}

// Notes 10: fastAlphaBlend performs a weighted blending of two images based on a mask, where the mask determines the contribution of each image to the final result.
function fastAlphaBlend(img1, img2, mask) {
    let m = new cv.Mat();
    mask.convertTo(m, cv.CV_32F, 1/255);
    let f1 = new cv.Mat(), f2 = new cv.Mat();
    img1.convertTo(f1, cv.CV_32F);
    img2.convertTo(f2, cv.CV_32F);
    let channels1 = new cv.MatVector(), channels2 = new cv.MatVector();
    cv.split(f1, channels1); cv.split(f2, channels2);
    for(let i=0; i<3; i++) {
        let t1 = new cv.Mat(), t2 = new cv.Mat(), invM = new cv.Mat(m.rows, m.cols, cv.CV_32F, new cv.Scalar(1.0));
        cv.subtract(invM, m, invM);
        cv.multiply(channels1.get(i), invM, t1);
        cv.multiply(channels2.get(i), m, t2);
        cv.add(t1, t2, channels2.get(i));
        [t1, t2, invM].forEach(x => x.delete());
    }
    let result = new cv.Mat();
    cv.merge(channels2, result);
    result.convertTo(result, cv.CV_8U);
    [m, f1, f2, channels1, channels2].forEach(x => { if(x.delete) x.delete(); });
    return result;
}

// Notes 11: processStacking is the main function that handles the image alignment and blending process. 
// It updates the status messages, performs the alignment using the robustAlignment function, optimizes the quality by creating a mask based on Laplacian edge detection, and then blends the images using fastAlphaBlend. 
// Finally, it displays the result and sets up the download button.
function processStacking() {
    try {
        status.innerText = "Aligning images...";
        const img1 = capturedMats[0];
        const img2Raw = capturedMats[1];
        const img2 = robustAlignment(img1, img2Raw);

        status.innerText = "Optimizing quality...";
        let gray1 = new cv.Mat(), gray2 = new cv.Mat();
        cv.cvtColor(img1, gray1, cv.COLOR_RGBA2GRAY);
        cv.cvtColor(img2, gray2, cv.COLOR_RGBA2GRAY);

        let lap1 = new cv.Mat(), lap2 = new cv.Mat();
        cv.Laplacian(gray1, lap1, cv.CV_64F);
        cv.Laplacian(gray2, lap2, cv.CV_64F);

        let abs1 = new cv.Mat(), abs2 = new cv.Mat();
        cv.convertScaleAbs(lap1, abs1);
        cv.convertScaleAbs(lap2, abs2);

        let mask = new cv.Mat();
        cv.compare(abs1, abs2, mask, cv.CMP_GT);
        
        let softMask = new cv.Mat();
        cv.GaussianBlur(mask, softMask, new cv.Size(31, 31), 0);

        let result = fastAlphaBlend(img1, img2, softMask);
        
        let resCanvas = document.getElementById('resCanvas') || document.createElement('canvas');
        resCanvas.id = 'resCanvas';
        if (!document.getElementById('resCanvas')) document.body.appendChild(resCanvas);
        
        cv.imshow(resCanvas, result);
        setupDownload(resCanvas);
        
        status.innerText = "Completed.";
        [img2, gray1, gray2, lap1, lap2, abs1, abs2, mask, softMask, result].forEach(m => { if(m.delete) m.delete(); });

    } catch (e) {
        console.error(e);
        status.innerText = "Process failed. Please try again.";
    }
}

// Notes 12: The snap button's onclick event captures the current video frame, creates an OpenCV Mat from it, and adds it to the capturedMats array. 
// It also updates the UI to show the captured image and changes the button text for the next capture. 
// Once two images are captured, it disables the snap button and starts the processing. 
// The reset button simply reloads the page to start over.
snapBtn.onclick = () => {
    const canvas = document.createElement('canvas');
    canvas.width = Math.min(video.videoWidth, MAX_WIDTH);
    canvas.height = (canvas.width / video.videoWidth) * video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    let mat = cv.imread(canvas);
    capturedMats.push(mat);

    const img = new Image();
    img.src = canvas.toDataURL('image/jpeg');
    img.className = "preview-img";
    previews.appendChild(img);

    if (capturedMats.length === 1) {
        snapBtn.innerText = "📸 2. Take Second Photo";
    } else if (capturedMats.length === 2) {
        snapBtn.disabled = true;
        snapBtn.innerText = "Processing...";
        resetBtn.style.display = "inline-block";
        setTimeout(processStacking, 100);
    }
};

// Notes 13: setupDownload creates a download button that allows the user to save the resulting blended image as a JPEG file.
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

// Notes 14: The reset button's onclick event simply reloads the page, allowing the user to start the process over with new captures.
resetBtn.onclick = () => { location.reload(); };

// Notes 15: The video element's onclick event allows the user to tap on the video feed to set a focus point. 
// It calculates the relative x and y coordinates of the click within the video element and applies constraints to the video track to set the focus point accordingly. 
// If successful, it updates the status message to indicate that focus is locked.   
video.onclick = async (e) => {
    const track = video.srcObject.getVideoTracks()[0];
    const capabilities = track.getCapabilities();
    const rect = video.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    try {
        await track.applyConstraints({
            advanced: [{ pointsOfInterest: [{x, y}] }]
        });
        status.innerText = "Focus Locked.";
    } catch (err) { console.warn(err); }
};