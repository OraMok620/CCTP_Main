const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snap = document.getElementById('snap');
const autoBtn = document.getElementById('auto-btn');
let videoTrack;

async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: "environment" }, 
            audio: false 
        });
        video.srcObject = stream;
        videoTrack = stream.getVideoTracks()[0];
    } catch (err) {
        console.error("Camera access error:", err);
    }
}

autoBtn.addEventListener('click', async () => {
    if (!videoTrack) return;

    const capabilities = videoTrack.getCapabilities();
    const settings = videoTrack.getSettings();

    if (!capabilities.exposureCompensation) {
        console.warn("Exposure compensation not supported.");
        return;
    }

    // 1. Quick Brightness Analysis
    const tempCanvas = document.createElement('canvas');
    const ctx = tempCanvas.getContext('2d');
    tempCanvas.width = 40; // Extremely small for speed
    tempCanvas.height = 30;
    ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    const data = ctx.getImageData(0, 0, tempCanvas.width, tempCanvas.height).data;

    let totalBrightness = 0;
    for (let i = 0; i < data.length; i += 4) {
        totalBrightness += (0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    }
    const avgBrightness = totalBrightness / (data.length / 4);

    // 2. Calculate New Target
    let currentComp = settings.exposureCompensation || 0;
    const step = capabilities.exposureCompensation.step || 0.5;
    let targetComp = currentComp;

    if (avgBrightness > 190) { 
        targetComp -= (step * 2); // Decrease brightness
    } else if (avgBrightness < 70) { 
        targetComp += (step * 2); // Increase brightness
    }

    // 3. Clamp to hardware limits
    targetComp = Math.max(capabilities.exposureCompensation.min, 
                 Math.min(capabilities.exposureCompensation.max, targetComp));

    // 4. Apply
    try {
        await videoTrack.applyConstraints({
            advanced: [
                { exposureMode: 'manual' }, 
                { exposureCompensation: targetComp }
            ]
        });
        console.log(`Brightness: ${Math.round(avgBrightness)} | New Compensation: ${targetComp}`);
    } catch (err) {
        console.error("Failed to adjust hardware:", err);
    }
});

snap.addEventListener('click', () => {
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const link = document.createElement('a');
    link.download = 'photo.png';
    link.href = canvas.toDataURL('image/png');
    link.click();
});

initCamera();