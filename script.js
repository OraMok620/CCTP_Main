// Notes 1: document.getElementById is used to access HTML elements by their ID.
const video = document.getElementById('video');
const captureBtn = document.getElementById('capture');
const restartBtn = document.getElementById('restart');
const status = document.getElementById('status');
const previews = document.getElementById('previews');
const hiddenCanvas = document.getElementById('hiddenCanvas');

// Notes 2: capturedMats holds the OpenCV Mat objects for the captured images, 
// MAX_WIDTH is a constant that limits the width of the captured images to ensure better performance during processing.
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

// Notes 4: onOpenCvReady initializes the app once OpenCV is ready, enabling the button and starting the camera.
function onOpenCvReady() {
    status.innerText = "It's ready! Follow the instructions below.";
    captureBtn.disabled = false;
    captureBtn.innerText = "Take photo focus on object.";
    startCamera();
}

// Notes 5: startCamera uses the MediaDevices API to access the user's camera, requesting the environment-facing camera for better quality.
// New learning : async allows a program to start a potentially long-running operation and still be able to respond to other events while waiting for the operation to complete.
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
//In this code, ORB is used to align two images by finding matching key points between them and computing a homography transformation.
function alignWithORB(refMat, targetMat) {
    let orb = new cv.ORB(); //cv.orb is a class in OpenCV that implements  which the ORB stand for *O*riented FAST and *R*otated *B*RIEF feature detection and description algorithm.
    let keyPoint_1 = new cv.KeyPointVector(), keyPoint_2 = new cv.KeyPointVector(); //cv.keyPointVector is a data structure used in OpenCV to store key points detected in an image. Here, keyPoint_1 and keyPoint_2 will hold the key points for the reference and target images, respectively.
    let baseFeatures = new cv.Mat(), targetFeatures = new cv.Mat(); //cv.mat is a matrix data structure used in OpenCV to store image data, descriptors, and other information. Here, baseFeatures and targetFeatures will hold the descriptors for the key points detected in the reference and target images, respectively.
    //orb.detectAndCompute is a method that detects key points and computes their descriptors in a single step. 
    //(refMat, new cv.Mat() are the input image and mask for the reference image, while keyPoint_1 and baseFeatures are the output key points and descriptors.
    orb.detectAndCompute(refMat, new cv.Mat(), keyPoint_1, baseFeatures); //refMat is the input image for which key points and descriptors are to be computed.
    orb.detectAndCompute(targetMat, new cv.Mat(), keyPoint_2, targetFeatures); //targetMat is the input image for which key points and descriptors are to be computed.
    //cv.BFMatcher is a brute-force matcher that matches descriptors between two sets. 
    // cv.NORM_HAMMING indicates that the Hamming distance is used for matching, which is suitable for binary descriptors like those produced by ORB. The second parameter 'false' indicates that we want to find the best match for each descriptor.
    let bf = new cv.BFMatcher(cv.NORM_HAMMING, false); 
    let matches = new cv.DMatchVector();
    bf.match(baseFeatures, targetFeatures, matches);
    let goodMatches = [];

    for (let i = 0; i < matches.size(); i++) goodMatches.push(matches.get(i));
    goodMatches.sort((a, b) => a.distance - b.distance);
    if (goodMatches.length < 20) throw "ORB Failed";

    let srcPts = [], dstPts = [];

    for (let i = 0; i < Math.min(goodMatches.length, 50); i++) {
        let coordinate_2 = keyPoint_2.get(goodMatches[i].trainIdx).pt;
        let coordinate_1 = keyPoint_1.get(goodMatches[i].queryIdx).pt;
        srcPts.push(coordinate_2.x, coordinate_2.y);
        dstPts.push(coordinate_1.x, coordinate_1.y);
    }

    let srcMat = cv.matFromArray(srcPts.length / 2, 1, cv.CV_32FC2, srcPts);
    let dstMat = cv.matFromArray(dstPts.length / 2, 1, cv.CV_32FC2, dstPts);
    let h = cv.findHomography(srcMat, dstMat, cv.RANSAC, 3);
    let aligned = new cv.Mat();

    cv.warpPerspective(targetMat, aligned, h, new cv.Size(refMat.cols, refMat.rows));
    [orb, keyPoint_1, keyPoint_2, baseFeatures, targetFeatures, bf, matches, srcMat, dstMat, h].forEach(obj => { if (obj && obj.delete) obj.delete(); });
    return aligned;
}

// Notes 7: AKAZE is another feature detection and matching algorithm that is designed to be faster and more efficient than ORB, especially for larger images.
function alignWithAKAZE(refMat, targetMat) {
    let akaze = cv.AKAZE.create();
    let keyPoint_1 = new cv.KeyPointVector(), keyPoint_2 = new cv.KeyPointVector();
    let baseFeatures = new cv.Mat(), targetFeatures = new cv.Mat();
    akaze.detectAndCompute(refMat, new cv.Mat(), keyPoint_1, baseFeatures);
    akaze.detectAndCompute(targetMat, new cv.Mat(), keyPoint_2, targetFeatures);
    let bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
    let matches = new cv.DMatchVector();
    bf.match(baseFeatures, targetFeatures, matches);
    let goodMatches = [];

    for (let i = 0; i < matches.size(); i++) goodMatches.push(matches.get(i));
    goodMatches.sort((a, b) => a.distance - b.distance);

    if (goodMatches.length < 4) throw "AKAZE Failed";

    let srcPts = [], dstPts = [];

    for (let i = 0; i < Math.min(goodMatches.length, 40); i++) {
        let coordinate_2 = keyPoint_2.get(goodMatches[i].trainIdx).pt;
        let coordinate_1 = keyPoint_1.get(goodMatches[i].queryIdx).pt;
        srcPts.push(coordinate_2.x, coordinate_2.y);
        dstPts.push(coordinate_1.x, coordinate_1.y);
    }

    let srcMat = cv.matFromArray(srcPts.length / 2, 1, cv.CV_32FC2, srcPts);
    let dstMat = cv.matFromArray(dstPts.length / 2, 1, cv.CV_32FC2, dstPts);
    let h = cv.findHomography(srcMat, dstMat, cv.RANSAC, 3);
    let aligned = new cv.Mat();
    cv.warpPerspective(targetMat, aligned, h, new cv.Size(refMat.cols, refMat.rows));

    [akaze, keyPoint_1, keyPoint_2, baseFeatures, targetFeatures, bf, matches, srcMat, dstMat, h].forEach(obj => { if (obj && obj.delete) obj.delete(); });
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
        let diff = new cv.Mat();
        cv.subtract(abs1, abs2, diff);
        cv.threshold(diff, mask, 5, 255, cv.THRESH_BINARY);
        
        let softMask = new cv.Mat();
        cv.GaussianBlur(mask, softMask, new cv.Size(15, 15), 0);

        let result = fastAlphaBlend(img1, img2, softMask);
        
        let resCanvas = document.getElementById('resCanvas') || document.createElement('canvas');
        resCanvas.id = 'resCanvas';
        if (!document.getElementById('resCanvas')) previews.appendChild(resCanvas);
        
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
// The restart button simply reloads the page to start over.
captureBtn.onclick = () => {
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
        captureBtn.innerText = "Take photo focus focus on background.";
    } else if (capturedMats.length === 2) {
        captureBtn.disabled = true;
        captureBtn.innerText = "Download your photo below or try again.";
        restartBtn.style.display = "inline-block";
        setTimeout(processStacking, 100);
    }
};

// Notes 13: setupDownload creates a download button that allows the user to save the resulting blended image as a JPEG file.
function setupDownload(canvas) {
    const downloadBtn = document.getElementById('download');
    downloadBtn.style.display = "inline-block";
    downloadBtn.onclick = () => {
        const link = document.createElement('a');
        link.download = 'stacked-photo.jpg';
        link.href = canvas.toDataURL('image/jpeg', 0.95);
        link.click();
    };
}

// Notes 14: The restart button's onclick event simply reloads the page, allowing the user to start the process over with new captures.
restartBtn.onclick = () => { location.reload(); };

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
        if (capturedMats.length === 0) {
            status.innerText = "Foreground focus set. Now take Photo 1.";
        } else {
            status.innerText = "Background focus set. Now take Photo 2.";
        }
    } catch (err) { console.warn(err); }
};