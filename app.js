const socket = io('http://localhost:5000');
const video = document.getElementById('videoFeed');
const canvas = document.getElementById('canvasFeed');
const ctx = canvas.getContext('2d');
const startButton = document.getElementById('startCamera');
const stopButton = document.getElementById('stopCamera');
const processButton = document.getElementById('enableProcessing');
const statusConnection = document.getElementById('statusConnection');
const statusCamera = document.getElementById('statusCamera');
const statusProcessing = document.getElementById('statusProcessing');
const threatsDetected = document.getElementById('threatsDetected');
const alertsTriggered = document.getElementById('alertsTriggered');
const framesProcessed = document.getElementById('framesProcessed');
const sessionTime = document.getElementById('sessionTime');
const recentAlerts = document.getElementById('recentAlerts');
const videoUpload = document.getElementById('videoUpload');
const uploadButton = document.getElementById('uploadButton');

let stream = null;
let isProcessing = false;
let frameCount = 0;
let threatCount = 0;
let alertCount = 0;
let startTime = null;

// SocketIO event handlers
socket.on('connect', () => {
    statusConnection.textContent = 'Connected';
    statusConnection.classList.remove('text-red-500');
    statusConnection.classList.add('text-green-500');
});

socket.on('disconnect', () => {
    statusConnection.textContent = 'Disconnected';
    statusConnection.classList.remove('text-green-500');
    statusConnection.classList.add('text-red-500');
});

socket.on('server_status', (data) => {
    console.log('Server status:', data.status);
});

socket.on('processed_frame', (data) => {
    if (isProcessing) {
        // Update canvas with processed frame
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        img.src = data.image;

        // Update stats
        frameCount++;
        framesProcessed.textContent = frameCount;
        if (data.threat_detected) {
            threatCount++;
            threatsDetected.textContent = threatCount;
        }
        if (data.alerts && data.alerts.length > 0) {
            alertCount += data.alerts.length;
            alertsTriggered.textContent = alertCount;
            updateRecentAlerts(data.alerts);
        }
    }
});

socket.on('processing_error', (data) => {
    console.error('Processing error:', data.message);
    alert('Error: ' + data.message);
});

// Update recent alerts list
function updateRecentAlerts(alerts) {
    recentAlerts.innerHTML = '';
    alerts.forEach(alert => {
        const li = document.createElement('li');
        li.textContent = `${new Date().toLocaleTimeString()}: ${alert}`;
        recentAlerts.appendChild(li);
    });
}

// Session timer
function updateSessionTime() {
    if (startTime) {
        const now = new Date();
        const diff = Math.floor((now - startTime) / 1000);
        const hours = Math.floor(diff / 3600).toString().padStart(2, '0');
        const minutes = Math.floor((diff % 3600) / 60).toString().padStart(2, '0');
        const seconds = (diff % 60).toString().padStart(2, '0');
        sessionTime.textContent = `${hours}:${minutes}:${seconds}`;
    }
}
setInterval(updateSessionTime, 1000);

// Start camera
startButton.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        statusCamera.textContent = 'Camera Active';
        statusCamera.classList.remove('text-red-500');
        statusCamera.classList.add('text-green-500');
        startButton.disabled = true;
        stopButton.disabled = false;
        startTime = new Date();
        sendFrames();
    } catch (err) {
        console.error('Error accessing camera:', err);
        alert('Failed to access camera. Please check permissions.');
    }
});

// Stop camera
stopButton.addEventListener('click', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        statusCamera.textContent = 'Camera Inactive';
        statusCamera.classList.remove('text-green-500');
        statusCamera.classList.add('text-red-500');
        startButton.disabled = false;
        stopButton.disabled = true;
        isProcessing = false;
        statusProcessing.textContent = 'Processing Disabled';
        statusProcessing.classList.remove('text-green-500');
        statusProcessing.classList.add('text-red-500');
        processButton.textContent = 'Enable Processing';
        startTime = null;
    }
});

// Toggle processing
processButton.addEventListener('click', () => {
    if (!stream) {
        alert('Camera must be started before enabling processing.');
        return;
    }
    isProcessing = !isProcessing;
    statusProcessing.textContent = isProcessing ? 'Processing Enabled' : 'Processing Disabled';
    statusProcessing.classList.toggle('text-green-500', isProcessing);
    statusProcessing.classList.toggle('text-red-500', !isProcessing);
    processButton.textContent = isProcessing ? 'Disable Processing' : 'Enable Processing';
});

// Send video frames to backend
function sendFrames() {
    if (!isProcessing || !stream) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
    socket.emit('video_frame', { image: dataUrl });
    requestAnimationFrame(sendFrames);
}

// Video file upload
uploadButton.addEventListener('click', () => {
    const file = videoUpload.files[0];
    if (!file) {
        alert('Please select a video file.');
        return;
    }
    const formData = new FormData();
    formData.append('video', file);
    fetch('/upload_video', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        alert(data.message);
        console.log('Upload response:', data);
    })
    .catch(err => {
        console.error('Upload error:', err);
        alert('Failed to upload video.');
    });
});

// Initialize UI
stopButton.disabled = true;