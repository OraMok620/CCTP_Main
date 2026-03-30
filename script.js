const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snap = document.getElementById('snap');
const autoBtn = document.getElementById('auto-btn');

async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: "environment" }, // Prefers the back camera
            audio: false 
        });
        video.srcObject = stream;
    } catch (err) {
        console.error("Error accessing camera: ", err);
        alert("Could not access camera. Please ensure you are using HTTPS and have granted permissions.");
    }
}

snap.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = 'captured-photo.png';
    link.href = dataURL;
    link.click();
});

autoBtn.addEventListener('click', () => {
    console.log("Auto exposure logic will be implemented here.");
});

initCamera();