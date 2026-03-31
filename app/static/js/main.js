const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadForm = document.getElementById('upload-form');
const resultsSection = document.getElementById('results');
const loader = document.getElementById('loader');
const resultContent = document.getElementById('result-content');

// Prevent defaults
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// Hover effects
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

// Handle drop and click
dropZone.addEventListener('drop', handleDrop, false);
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFiles(fileInput.files);
});

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFiles(files) {
    const file = files[0];
    if (file && file.type.startsWith('image/')) {
        uploadFile(file);
    } else {
        alert('Please upload a valid image file (PNG/JPG).');
    }
}

async function uploadFile(file) {
    // UI Transitions
    document.querySelector('.upload-section').classList.add('hidden');
    resultsSection.classList.remove('hidden');
    loader.classList.remove('hidden');
    resultContent.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            throw new Error(data.error || 'Prediction failed');
        }
    } catch (error) {
        alert('Error: ' + error.message);
        resetApp();
    }
}

function displayResults(data) {
    loader.classList.add('hidden');
    resultContent.classList.remove('hidden');

    // Update Image
    document.getElementById('result-image').src = data.result_image_url;

    // Update Metrics
    document.getElementById('detection-count').innerText = data.detections;

    // Update Badge
    const badge = document.getElementById('status-badge');
    if (data.has_fire) {
        badge.innerText = '🔥 Wildfire Detected';
        badge.className = 'status-badge status-fire';
    } else {
        badge.innerText = '✅ Safe / No Fire';
        badge.className = 'status-badge status-safe';
    }
}

function resetApp() {
    resultsSection.classList.add('hidden');
    document.querySelector('.upload-section').classList.remove('hidden');
    fileInput.value = '';
}
