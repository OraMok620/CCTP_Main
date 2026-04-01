const video = document.getElementById('video');
const snapBtn = document.getElementById('snapBtn');
const resetBtn = document.getElementById('resetBtn');
const status = document.getElementById('status');
const previews = document.getElementById('previews');
const hiddenCanvas = document.getElementById('hiddenCanvas');

let capturedMats = [];
const MAX_WIDTH = 1024; 

//Check if OpenCV.js is loaded
window.onload = () => {
    let checkCV = setInterval(() => {
        if (typeof cv !== 'undefined' && cv.Mat) {
            clearInterval(checkCV);
            onOpenCvReady();
        }
    }, 500);
};

function onOpenCvReady() {
    status.innerText = "Please point the camera at the 'object' to take a photo.";
    snapBtn.disabled = false;
    snapBtn.innerText = "📸 First Photo for your object";
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
        status.innerText = "Error: Unable to access the camera";
    }
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
        status.innerText = "First photo taken! Now please point the camera at the 'background' to take a photo.";
        snapBtn.innerText = "📸 Second Photo for the background";
    } else if (capturedMats.length === 2) {
        snapBtn.disabled = true;
        snapBtn.innerText = "Processing...";
        resetBtn.style.display = "inline-block";
        setTimeout(processStacking, 100);
    }
};

function processStacking() {
    try {
        status.innerText = "Processing... This may take a few seconds.";
        
        let img1 = capturedMats[0]; 
        let img2 = capturedMats[1]; 

        let gray1 = new cv.Mat();
        let gray2 = new cv.Mat();
        cv.cvtColor(img1, gray1, cv.COLOR_RGBA2GRAY);
        cv.cvtColor(img2, gray2, cv.COLOR_RGBA2GRAY);

        let blur1 = new cv.Mat();
        let blur2 = new cv.Mat();
        cv.GaussianBlur(gray1, blur1, new cv.Size(5, 5), 0);
        cv.GaussianBlur(gray2, blur2, new cv.Size(5, 5), 0);

        let lap1 = new cv.Mat();
        let lap2 = new cv.Mat();
        cv.Laplacian(blur1, lap1, cv.CV_64F);
        cv.Laplacian(blur2, lap2, cv.CV_64F);

        let abs1 = new cv.Mat();
        let abs2 = new cv.Mat();
        cv.convertScaleAbs(lap1, abs1);
        cv.convertScaleAbs(lap2, abs2);

        let mask = new cv.Mat();
        cv.compare(abs1, abs2, mask, cv.CMP_GT);

        let result = img2.clone(); 
        img1.copyTo(result, mask); 

        let resCanvas = document.getElementById('resCanvas');
        if(!resCanvas) {
            resCanvas = document.createElement('canvas');
            resCanvas.id = 'resCanvas';
            document.body.appendChild(resCanvas);
        }
        cv.imshow(resCanvas, result);
        const downloadBtn = document.getElementById('downloadBtn');
        downloadBtn.style.display = "inline-block";

        downloadBtn.onclick = () => {
            const resCanvas = document.getElementById('resCanvas');
            const link = document.createElement('a');
            link.download = 'stacked-photo.jpg';
            // Convert canvas to a data URL (the actual image data)
            link.href = resCanvas.toDataURL('image/jpeg', 0.9); 
            link.click();
        };
        
        status.innerText = "🎉 Processing successful!";

        [gray1, gray2, blur1, blur2, lap1, lap2, abs1, abs2, mask, result].forEach(m => m.delete());
        
    } catch (err) {
        console.error(err);
        status.innerText = "Processing failed: " + err;
    }
}

resetBtn.onclick = () => {
    capturedMats.forEach(m => m.delete());
    capturedMats = [];
    previews.innerHTML = "";
    const oldRes = document.getElementById('resCanvas');
    if(oldRes) oldRes.remove();
    snapBtn.disabled = false;
    snapBtn.innerText = "📸 First Photo for your object";
    status.innerText = "Please point the camera at the 'object' to take a photo.";
    resetBtn.style.display = "none";
};