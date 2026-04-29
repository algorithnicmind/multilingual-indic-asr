let mediaRecorder;
let audioChunks = [];
let audioContext;
let analyser;
let dataArray;
let animationId;

const recordBtn = document.getElementById('record-btn');
const statusText = document.getElementById('status-text');
const transcriptText = document.getElementById('transcript');
const langBadge = document.getElementById('language-badge');
const confidenceText = document.getElementById('confidence');
const visualizer = document.getElementById('visualizer');
const canvasCtx = visualizer.getContext('2d');

// Initialize Visualizer
function initVisualizer(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    
    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    dataArray = new Uint8Array(bufferLength);
    
    draw();
}

function draw() {
    animationId = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(dataArray);
    
    canvasCtx.clearRect(0, 0, visualizer.width, visualizer.height);
    
    const barWidth = (visualizer.width / dataArray.length) * 2.5;
    let barHeight;
    let x = 0;
    
    for (let i = 0; i < dataArray.length; i++) {
        barHeight = dataArray[i] / 2;
        
        const gradient = canvasCtx.createLinearGradient(0, visualizer.height, 0, 0);
        gradient.addColorStop(0, '#4f46e5');
        gradient.addColorStop(1, '#ec4899');
        
        canvasCtx.fillStyle = gradient;
        canvasCtx.fillRect(x, visualizer.height - barHeight, barWidth, barHeight);
        
        x += barWidth + 1;
    }
}

// Recording Logic
recordBtn.addEventListener('click', async () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        recordBtn.classList.remove('recording');
        statusText.innerText = "Processing speech...";
        cancelAnimationFrame(animationId);
    } else {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            initVisualizer(stream);
            
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                sendAudioToServer(audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };
            
            mediaRecorder.start();
            recordBtn.classList.add('recording');
            statusText.innerText = "Listening...";
        } catch (err) {
            console.error("Error accessing microphone:", err);
            statusText.innerText = "Microphone access denied";
        }
    }
});

async function sendAudioToServer(blob) {
    const formData = new FormData();
    formData.append('audio', blob, 'recording.wav');
    
    try {
        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            transcriptText.innerText = data.text;
            langBadge.innerText = data.language;
            langBadge.classList.remove('hidden');
            confidenceText.innerText = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
            statusText.innerText = "Transcription complete";
        } else {
            statusText.innerText = "Error: " + data.error;
        }
    } catch (err) {
        console.error("Error sending audio:", err);
        statusText.innerText = "Connection failed";
    }
}

// Copy functionality
document.getElementById('copy-btn').addEventListener('click', () => {
    navigator.clipboard.writeText(transcriptText.innerText);
    const originalIcon = document.querySelector('#copy-btn i').className;
    document.querySelector('#copy-btn i').className = 'fas fa-check';
    setTimeout(() => {
        document.querySelector('#copy-btn i').className = 'far fa-copy';
    }, 2000);
});

// Set canvas dimensions
function resizeCanvas() {
    visualizer.width = visualizer.clientWidth;
    visualizer.height = visualizer.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();
