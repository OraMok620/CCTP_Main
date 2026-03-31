// NOTES 1: Get references to HTML elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snap = document.getElementById('snap');
const autoBtn = document.getElementById('auto-btn');
let videoTrack;

// NOTES 2: Initialize camera stream
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: "environment" }, // facingMode: "environment" for rear camera on mobile devices, "user" for front camera.
            audio: false 
        });
        video.srcObject = stream;
        videoTrack = stream.getVideoTracks()[0];
        setTimeout(logExposureInfo, 1000);
    } catch (err) {
        console.error("Error accessing camera: ", err);
        alert("Could not access camera. Please ensure you are using HTTPS and have granted permissions.");
    }
}

//NOTES 3: Function to log camera capabilities and current settings, including exposure information if available.
function logExposureInfo() {
    if (!videoTrack) return;

    const capabilities = videoTrack.getCapabilities();
    const settings = videoTrack.getSettings();

    console.log("--- Camera Capabilities ---", capabilities);
    console.log("--- Current Settings ---", settings);

    if (settings.exposureMode) {
        console.log("Current Exposure Mode:", settings.exposureMode);
        console.log("Current Exposure Compensation:", settings.exposureCompensation);
    } else {
        console.log("Exposure control is not supported on this browser/device.");
    }
}

// NOTES 4: Capture photo on button click (Button functionality to capture the current frame from the video stream and save it as an image)
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

// NOTES 5: Set auto-exposure on button click (Button functionality to set the camera's exposure mode to 'continuous', which is typically the auto-exposure mode. It checks for support and applies the constraint if possible.)
autoBtn.addEventListener('click', async () => {
    if (!videoTrack) {
        console.error("Camera track not found.");
        return;
    }
    // 1. Get current capabilities and settings
    const capabilities = videoTrack.getCapabilities();
    const settings = videoTrack.getSettings();
    // 2. Check if the device actually supports auto-exposure
    if (!capabilities.exposureMode || !capabilities.exposureMode.includes('continuous')) {
        console.warn("This device does not support continuous (auto) exposure.");
        return;
    }
    try {
        // 3. Apply the 'continuous' constraint
        await videoTrack.applyConstraints({
            advanced: [{ exposureMode: 'continuous' }]
        });
        console.log("✅ Exposure set to AUTO (continuous)");
        // 4. Log the new settings to the console so you can verify the change
        const newSettings = videoTrack.getSettings();
        console.log("New Camera Settings:", newSettings);
    } catch (err) {
        console.error("Error applying auto exposure:", err);
    }
});


setInterval(() => {
    if (videoTrack) {
        const settings = videoTrack.getSettings();
        // exposureTime is usually in milliseconds or fractional seconds
        console.log("Live Exposure Time:", settings.exposureTime);
    }
}, 1000);

initCamera();