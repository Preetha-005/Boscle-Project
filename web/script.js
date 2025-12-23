// ===================================
// STATE MANAGEMENT
// ===================================
const state = {
    inputType: 'local',
    selectedFile: null,
    videoUrl: null,
    processing: false,
    currentStep: 0,
    progress: 0,
    customOutputDirectory: null,
    platform: 'Unknown',
    pathSeparator: '/',
    defaultOutputDir: '',
    // Password protection state
    passwordProtection: false,
    filePassword: null
};

// ===================================
// DOM ELEMENTS
// ===================================
const elements = {
    inputTypeBtns: document.querySelectorAll('.input-type-btn'),
    fileUploadArea: document.getElementById('fileUploadArea'),
    urlInputArea: document.getElementById('urlInputArea'),
    fileInput: document.getElementById('fileInput'),
    urlInput: document.getElementById('urlInput'),
    selectedFile: document.getElementById('selectedFile'),
    fileName: document.getElementById('fileName'),
    fileSize: document.getElementById('fileSize'),
    removeFileBtn: document.getElementById('removeFileBtn'),
    processBtn: document.getElementById('processBtn'),
    progressSection: document.getElementById('progressSection'),
    progressBar: document.getElementById('progressBar'),
    progressPercentage: document.getElementById('progressPercentage'),
    logToggle: document.getElementById('logToggle'),
    logContent: document.getElementById('logContent'),
    logMessages: document.getElementById('logMessages'),
    resultsSection: document.getElementById('resultsSection'),
    resultsGrid: document.getElementById('resultsGrid'),
    processAnotherBtn: document.getElementById('processAnotherBtn')
};

// ===================================
// INPUT TYPE SELECTION
// ===================================
elements.inputTypeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        elements.inputTypeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        state.inputType = btn.dataset.type;

        if (state.inputType === 'local') {
            elements.fileUploadArea.classList.remove('hidden');
            elements.urlInputArea.classList.add('hidden');
        } else {
            elements.fileUploadArea.classList.add('hidden');
            elements.urlInputArea.classList.remove('hidden');
        }

        resetFileSelection();
    });
});

// ===================================
// FILE UPLOAD HANDLING
// ===================================
elements.fileUploadArea.addEventListener('click', () => {
    elements.fileInput.click();
});

elements.fileUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.fileUploadArea.classList.add('drag-over');
});

elements.fileUploadArea.addEventListener('dragleave', () => {
    elements.fileUploadArea.classList.remove('drag-over');
});

elements.fileUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.fileUploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelection(files[0]);
    }
});

elements.fileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        handleFileSelection(files[0]);
    }
});

function handleFileSelection(file) {
    // Validate file type - check both MIME type and file extension
    // Some browsers don't correctly identify MIME types for all video formats
    const validTypes = [
        'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska',
        'video/webm', 'video/x-ms-wmv', 'video/x-flv', 'video/avi', 'video/x-m4v',
        'video/3gpp', 'video/ogg', 'application/octet-stream' // Some browsers report this for video files
    ];
    const validExtensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.wmv', '.flv', '.m4v', '.3gp'];

    // Get file extension
    const fileName = file.name.toLowerCase();
    const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext));
    const hasValidType = validTypes.includes(file.type) || file.type === '';

    if (!hasValidExtension && !hasValidType) {
        showNotification('Please select a valid video file (MP4, MOV, AVI, MKV, WebM, WMV, FLV)', 'error');
        return;
    }

    // Validate file size (max 2GB)
    const maxSize = 2 * 1024 * 1024 * 1024;
    if (file.size > maxSize) {
        showNotification('File size must be less than 2GB', 'error');
        return;
    }

    state.selectedFile = file;

    // Update UI
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = formatFileSize(file.size);
    elements.fileUploadArea.classList.add('hidden');
    elements.selectedFile.classList.remove('hidden');
}

elements.removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetFileSelection();
});

function resetFileSelection() {
    state.selectedFile = null;
    state.videoUrl = null;
    elements.fileInput.value = '';
    elements.urlInput.value = '';
    elements.selectedFile.classList.add('hidden');

    if (state.inputType === 'local') {
        elements.fileUploadArea.classList.remove('hidden');
    }
}

// ===================================
// PROCESS BUTTON
// ===================================
elements.processBtn.addEventListener('click', () => {
    if (state.processing) return;

    // Validate input
    if (state.inputType === 'local' && !state.selectedFile) {
        showNotification('Please select a video file', 'error');
        return;
    }

    if (state.inputType !== 'local') {
        const url = elements.urlInput.value.trim();
        if (!url) {
            showNotification('Please enter a video URL', 'error');
            return;
        }
        state.videoUrl = url;
    }

    // Show password protection modal instead of starting directly
    showPasswordModal();
});

// ===================================
// PROCESSING LOGIC
// ===================================
async function startProcessing() {
    state.processing = true;
    state.currentStep = 0;
    state.progress = 0;

    // Update UI
    elements.processBtn.disabled = true;
    elements.processBtn.querySelector('.btn-text').textContent = 'Processing...';
    elements.progressSection.classList.remove('hidden');
    elements.resultsSection.classList.add('hidden');

    // Scroll to progress section
    elements.progressSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Clear previous logs
    elements.logMessages.innerHTML = '';

    try {
        let videoPath = null;

        // Handle file upload for local files
        if (state.inputType === 'local' && state.selectedFile) {
            addLogMessage('Uploading file to server...', 'info');

            // Create form data for file upload
            const formData = new FormData();
            formData.append('file', state.selectedFile);

            // Upload file
            const uploadResponse = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) {
                const error = await uploadResponse.json();
                throw new Error(error.error || 'File upload failed');
            }

            const uploadResult = await uploadResponse.json();
            videoPath = uploadResult.filepath;
            addLogMessage(`File uploaded: ${uploadResult.filename}`, 'success');
        } else {
            videoPath = state.videoUrl;
        }

        // Prepare request data
        const requestData = {
            inputType: state.inputType,
            videoPath: videoPath,
            options: {
                burnCaptions: document.getElementById('burnCaptions').checked,
                generateReport: document.getElementById('generateReport').checked,
                passwordProtection: state.passwordProtection,
                filePassword: state.filePassword
            }
        };

        addLogMessage('Starting video processing...', 'info');

        // Start processing on backend
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const { processId } = await response.json();
        addLogMessage(`Processing started (ID: ${processId})`, 'success');

        // Poll for status updates
        await pollProcessingStatus(processId);

    } catch (error) {
        addLogMessage(`Error: ${error.message}`, 'error');
        // Show error modal for technical glitches
        showErrorModal();
    }
}

async function pollProcessingStatus(processId) {
    const pollInterval = 1000; // Poll every second

    while (state.processing) {
        try {
            const response = await fetch(`/api/status/${processId}`);

            if (!response.ok) {
                throw new Error('Failed to get status');
            }

            const status = await response.json();

            // Update progress
            updateProgressBar(status.progress || 0);

            // Update step
            if (status.step) {
                addLogMessage(status.step, 'info');
                updateProgressStepByName(status.step);
            }

            // Check if complete
            if (status.status === 'completed') {
                addLogMessage('Processing completed successfully!', 'success');
                state.results = status.results;
                completeProcessing();
                break;
            } else if (status.status === 'error') {
                throw new Error(status.error || 'Processing failed');
            }

            // Wait before next poll
            await new Promise(resolve => setTimeout(resolve, pollInterval));

        } catch (error) {
            addLogMessage(`Error: ${error.message}`, 'error');
            // Show error modal for technical glitches
            showErrorModal();
            break;
        }
    }
}

function updateProgressStepByName(stepName) {
    const stepMap = {
        'Initializing': 1,
        'Processing input': 1,
        'Extracting audio': 2,
        'Transcribing': 3,
        'Generating captions': 4,
        'Creating report': 5,
        'Complete': 5
    };

    const stepNumber = stepMap[stepName] || 1;
    updateProgressStep(stepNumber, 'active');

    // Mark previous steps as completed
    for (let i = 1; i < stepNumber; i++) {
        updateProgressStep(i, 'completed');
    }
}

function simulateProcessing(duration, targetProgress) {
    return new Promise(resolve => {
        const startProgress = state.progress;
        const progressDiff = targetProgress - startProgress;
        const steps = 50;
        const stepDuration = duration / steps;
        const progressStep = progressDiff / steps;

        let currentStep = 0;
        const interval = setInterval(() => {
            currentStep++;
            state.progress = startProgress + (progressStep * currentStep);
            updateProgressBar(state.progress);

            if (currentStep >= steps) {
                clearInterval(interval);
                resolve();
            }
        }, stepDuration);
    });
}

function updateProgressBar(progress) {
    elements.progressBar.style.width = `${progress}%`;
    elements.progressPercentage.textContent = `${Math.round(progress)}%`;
}

function updateProgressStep(stepNumber, status) {
    const steps = document.querySelectorAll('.progress-step');
    const step = steps[stepNumber - 1];

    if (!step) return;

    if (status === 'active') {
        step.classList.add('active');
        step.classList.remove('completed');
    } else if (status === 'completed') {
        step.classList.remove('active');
        step.classList.add('completed');
    }
}

function addLogMessage(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const logMessage = document.createElement('div');
    logMessage.className = `log-message ${type}`;
    logMessage.innerHTML = `
        <span class="timestamp">[${timestamp}]</span>
        <span>${message}</span>
    `;
    elements.logMessages.appendChild(logMessage);
    elements.logMessages.scrollTop = elements.logMessages.scrollHeight;
}

// ===================================
// COMPLETION
// ===================================
function completeProcessing() {
    state.processing = false;

    // Update UI
    elements.processBtn.disabled = false;
    elements.processBtn.querySelector('.btn-text').textContent = 'Start Processing';

    // Show results
    setTimeout(() => {
        elements.progressSection.classList.add('hidden');
        elements.resultsSection.classList.remove('hidden');
        displayResults();
        elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 1000);
}

function displayResults() {
    // Use actual results from backend if available
    if (state.results && state.results.zipFile) {
        const isProtected = state.results.passwordProtected;
        const filename = state.results.zipFilename;

        // Build the download URL
        const downloadUrl = `/api/download?filename=${encodeURIComponent(filename)}`;

        elements.resultsGrid.innerHTML = `
            <div class="result-card download-card ${isProtected ? 'protected-card' : ''}">
                <div class="result-header">
                    <div class="result-icon ${isProtected ? 'protected-icon' : ''}">
                        ${isProtected ?
                `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12 15V17M6 21H18C19.1046 21 20 20.1046 20 19V13C20 11.8954 19.1046 11 18 11H6C4.89543 11 4 11.8954 4 13V19C4 20.1046 4.89543 21 6 21ZM16 11V7C16 4.79086 14.2091 3 12 3C9.79086 3 8 4.79086 8 7V11H16Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>` :
                `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M21 15V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V15M7 10L12 15M12 15L17 10M12 15V3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>`
            }
                    </div>
                    <h3 class="result-title">${isProtected ? '🔒 Password Protected Files' : '📦 Your Files Are Ready!'}</h3>
                </div>
                
                <p class="result-description">
                    ${isProtected ?
                'Your files are encrypted and secured. You will need your password to extract them.' :
                'All your processed files are bundled and ready for download.'
            }
                </p>
                
                <div class="download-contents">
                    <p class="contents-title">📁 Package contains:</p>
                    <ul class="contents-list">
                        <li>🎬 Captioned Video (with burned-in captions)</li>
                        <li>📄 Meeting Report (PDF)</li>
                        <li>📝 Transcript (TXT)</li>
                    </ul>
                </div>
                
                ${isProtected ? '<p class="password-reminder">💡 Remember: You\'ll need your password to extract files</p>' : ''}
                
                <a href="${downloadUrl}" class="download-btn" download="${filename}">
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M21 15V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V15M7 10L12 15M12 15L17 10M12 15V3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    Download ${isProtected ? 'Protected ' : ''}Files
                </a>
                
                <p class="filename-display">${filename}</p>
            </div>
        `;
    } else if (state.results) {
        // Fallback for individual files (if zip creation failed)
        const results = [];

        if (state.results.captionedVideo) {
            results.push({ title: 'Captioned Video', path: state.results.captionedVideo });
        }
        if (state.results.report) {
            results.push({ title: 'Meeting Report', path: state.results.report });
        }
        if (state.results.transcript) {
            results.push({ title: 'Transcript', path: state.results.transcript });
        }

        if (results.length > 0) {
            elements.resultsGrid.innerHTML = `
                <div class="result-card">
                    <h3 class="result-title">⚠️ Files Created (Not Zipped)</h3>
                    <p class="result-description">Files were created but zip creation failed. Contact support if this persists.</p>
                    <ul class="contents-list">
                        ${results.map(r => `<li>${r.title}: ${r.path}</li>`).join('')}
                    </ul>
                </div>
            `;
        } else {
            elements.resultsGrid.innerHTML = `
                <div class="result-card">
                    <p>Processing completed! Please check the server logs if files are missing.</p>
                </div>
            `;
        }
    } else {
        // Fallback message
        elements.resultsGrid.innerHTML = `
            <div class="result-card">
                <p>Processing completed! Please check the server logs for details.</p>
            </div>
        `;
    }
}

async function openFile(filepath) {
    try {
        const response = await fetch('/api/open-file', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ filepath: filepath })
        });

        const result = await response.json();

        if (response.ok) {
            showNotification(result.message, 'success');
        } else {
            showNotification(`Error: ${result.error}`, 'error');
        }
    } catch (error) {
        showNotification(`Failed to open file: ${error.message}`, 'error');
        console.error('Error opening file:', error);
    }
}

// ===================================
// LOG TOGGLE
// ===================================
elements.logToggle.addEventListener('click', () => {
    elements.logContent.classList.toggle('hidden');
    elements.logToggle.classList.toggle('open');
});

// ===================================
// PROCESS ANOTHER VIDEO
// ===================================
elements.processAnotherBtn.addEventListener('click', () => {
    resetFileSelection();
    elements.resultsSection.classList.add('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// ===================================
// UTILITY FUNCTIONS
// ===================================
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'error' ? 'rgba(245, 87, 108, 0.9)' : 'rgba(79, 172, 254, 0.9)'};
        color: white;
        padding: 16px 24px;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(20px);
        z-index: 1000;
        animation: slideInRight 0.3s ease;
        font-weight: 600;
        max-width: 400px;
        word-wrap: break-word;
    `;
    notification.textContent = message;

    document.body.appendChild(notification);

    // Remove after 4 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            if (notification.parentNode) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 4000);
}

// Add notification animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ===================================
// OUTPUT DIRECTORY MANAGEMENT
// ===================================
function changeOutputDirectory() {
    const currentPath = state.customOutputDirectory || state.defaultOutputDir;

    // Platform-specific examples
    let examples = '';
    if (state.platform === 'Windows') {
        examples = '  • C:\\Users\\YourName\\Videos\\MeetingVideos\n' +
            '  • D:\\Projects\\ProcessedVideos\n' +
            '  • C:\\MyVideos';
    } else if (state.platform === 'Darwin') {
        examples = '  • /Users/YourName/Movies/MeetingVideos\n' +
            '  • ~/Documents/ProcessedVideos\n' +
            '  • /Volumes/ExternalDrive/Videos';
    } else {
        examples = '  • /home/username/Videos/MeetingVideos\n' +
            '  • ~/Documents/ProcessedVideos\n' +
            '  • /mnt/storage/Videos';
    }

    const newPath = prompt(
        'Enter the full path where you want to save processed files:\n\n' +
        'Examples:\n' + examples + '\n\n' +
        'Current path:',
        currentPath
    );

    if (newPath && newPath.trim() !== '') {
        const trimmedPath = newPath.trim();

        // Validate path format against server platform
        const isWindowsPath = /^[A-Za-z]:\\/.test(trimmedPath) || trimmedPath.includes(':\\');
        const isUnixPath = trimmedPath.startsWith('/') || trimmedPath.startsWith('~');

        // Warn if Windows path on non-Windows server
        if (state.platform !== 'Windows' && isWindowsPath) {
            showNotification(
                '⚠️ Warning: You entered a Windows path (e.g., C:\\...) but the server is running on ' + state.platform + '. ' +
                'The server will use its default directory instead. Please use Unix-style paths (e.g., /home/user/...).',
                'error'
            );
            return;
        }

        // Warn if Unix path on Windows server
        if (state.platform === 'Windows' && isUnixPath && !trimmedPath.startsWith('~')) {
            showNotification(
                '⚠️ Warning: You entered a Unix path but the server is running on Windows. ' +
                'Please use Windows-style paths (e.g., C:\\Users\\...).',
                'error'
            );
            return;
        }

        state.customOutputDirectory = trimmedPath;
        document.getElementById('outputPath').textContent = trimmedPath;
        showNotification('Output directory updated!', 'success');
    }
}

// ===================================
// FETCH CONFIGURATION FROM SERVER
// ===================================
async function fetchConfig() {
    try {
        const response = await fetch('/api/config');
        if (response.ok) {
            const config = await response.json();
            state.platform = config.platform || 'Unknown';
            state.pathSeparator = config.pathSeparator || '/';
            state.defaultOutputDir = config.outputDir || '';

            // Update output path display
            const outputPathElement = document.getElementById('outputPath');
            if (outputPathElement && state.defaultOutputDir) {
                outputPathElement.textContent = state.defaultOutputDir;
            }

            console.log(`Platform detected: ${state.platform}`);
            console.log(`Default output directory: ${state.defaultOutputDir}`);
        }
    } catch (error) {
        console.warn('Could not fetch config from server:', error);
        // Use fallback values
        setFallbackOutputDirectory();
    }
}

// ===================================
// SET FALLBACK OUTPUT DIRECTORY
// ===================================
function setFallbackOutputDirectory() {
    const outputPathElement = document.getElementById('outputPath');
    if (outputPathElement) {
        // Try to detect platform from user agent
        const userAgent = navigator.userAgent.toLowerCase();
        let outputPath = '~/Videos/MeetingCaptioning';

        if (userAgent.includes('win')) {
            outputPath = '%USERPROFILE%\\Videos\\MeetingCaptioning';
            state.platform = 'Windows';
            state.pathSeparator = '\\';
        } else if (userAgent.includes('mac')) {
            outputPath = '~/Movies/MeetingCaptioning';
            state.platform = 'Darwin';
            state.pathSeparator = '/';
        } else {
            outputPath = '~/Videos/MeetingCaptioning';
            state.platform = 'Linux';
            state.pathSeparator = '/';
        }

        state.defaultOutputDir = outputPath;
        outputPathElement.textContent = outputPath;
    }
}

// ===================================
// INITIALIZATION
// ===================================
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Meeting Captioning Studio initializing...');

    // Fetch configuration from server
    await fetchConfig();

    // Initialize password modal handlers
    initPasswordModal();

    console.log('Meeting Captioning Studio initialized');
    console.log('Ready to process videos!');
});

// ===================================
// PASSWORD PROTECTION MODAL
// ===================================
function showPasswordModal() {
    const modal = document.getElementById('passwordModal');
    const protectionQuestion = document.getElementById('protectionQuestion');
    const passwordOptions = document.getElementById('passwordOptions');
    const generatedPasswordDisplay = document.getElementById('generatedPasswordDisplay');
    const customPasswordInput = document.getElementById('customPasswordInput');

    // Reset modal state
    protectionQuestion.classList.remove('hidden');
    passwordOptions.classList.add('hidden');
    generatedPasswordDisplay.classList.add('hidden');
    customPasswordInput.classList.add('hidden');

    // Reset password state
    state.passwordProtection = false;
    state.filePassword = null;

    // Show modal
    modal.classList.remove('hidden');
}

function hidePasswordModal() {
    const modal = document.getElementById('passwordModal');
    modal.classList.add('hidden');
}

function generateRandomPassword(length = 12) {
    const charset = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789';
    let password = '';
    const randomValues = new Uint32Array(length);
    crypto.getRandomValues(randomValues);
    for (let i = 0; i < length; i++) {
        password += charset[randomValues[i] % charset.length];
    }
    return password;
}

function initPasswordModal() {
    // Get modal elements
    const noProtectionBtn = document.getElementById('noProtectionBtn');
    const yesProtectionBtn = document.getElementById('yesProtectionBtn');
    const backToQuestionBtn = document.getElementById('backToQuestionBtn');
    const generatePasswordOption = document.getElementById('generatePasswordOption');
    const customPasswordOption = document.getElementById('customPasswordOption');
    const backToOptionsBtn1 = document.getElementById('backToOptionsBtn1');
    const backToOptionsBtn2 = document.getElementById('backToOptionsBtn2');
    const confirmGeneratedPasswordBtn = document.getElementById('confirmGeneratedPasswordBtn');
    const confirmCustomPasswordBtn = document.getElementById('confirmCustomPasswordBtn');
    const copyPasswordBtn = document.getElementById('copyPasswordBtn');
    const togglePassword1 = document.getElementById('togglePassword1');
    const togglePassword2 = document.getElementById('togglePassword2');
    const closeErrorModalBtn = document.getElementById('closeErrorModalBtn');
    const retryProcessingBtn = document.getElementById('retryProcessingBtn');

    // Modal content sections
    const protectionQuestion = document.getElementById('protectionQuestion');
    const passwordOptions = document.getElementById('passwordOptions');
    const generatedPasswordDisplay = document.getElementById('generatedPasswordDisplay');
    const customPasswordInput = document.getElementById('customPasswordInput');

    // No protection - continue without password
    noProtectionBtn.addEventListener('click', () => {
        state.passwordProtection = false;
        state.filePassword = null;
        hidePasswordModal();
        startProcessing();
    });

    // Yes protection - show options
    yesProtectionBtn.addEventListener('click', () => {
        protectionQuestion.classList.add('hidden');
        passwordOptions.classList.remove('hidden');
    });

    // Back to question
    backToQuestionBtn.addEventListener('click', () => {
        passwordOptions.classList.add('hidden');
        protectionQuestion.classList.remove('hidden');
    });

    // Generate random password option
    generatePasswordOption.addEventListener('click', () => {
        const password = generateRandomPassword(12);
        state.filePassword = password;
        state.passwordProtection = true;

        document.getElementById('generatedPasswordValue').textContent = password;
        passwordOptions.classList.add('hidden');
        generatedPasswordDisplay.classList.remove('hidden');
    });

    // Custom password option
    customPasswordOption.addEventListener('click', () => {
        document.getElementById('customPasswordField').value = '';
        document.getElementById('confirmPasswordField').value = '';
        document.getElementById('passwordError').classList.add('hidden');
        passwordOptions.classList.add('hidden');
        customPasswordInput.classList.remove('hidden');
    });

    // Back to options from generated password
    backToOptionsBtn1.addEventListener('click', () => {
        generatedPasswordDisplay.classList.add('hidden');
        passwordOptions.classList.remove('hidden');
    });

    // Back to options from custom password
    backToOptionsBtn2.addEventListener('click', () => {
        customPasswordInput.classList.add('hidden');
        passwordOptions.classList.remove('hidden');
    });

    // Confirm generated password
    confirmGeneratedPasswordBtn.addEventListener('click', () => {
        hidePasswordModal();
        showNotification('🔒 Files will be protected with your password', 'success');
        startProcessing();
    });

    // Confirm custom password
    confirmCustomPasswordBtn.addEventListener('click', () => {
        const password = document.getElementById('customPasswordField').value;
        const confirmPassword = document.getElementById('confirmPasswordField').value;
        const errorEl = document.getElementById('passwordError');

        // Validate
        if (password.length < 6) {
            errorEl.textContent = 'Password must be at least 6 characters long';
            errorEl.classList.remove('hidden');
            return;
        }

        if (password !== confirmPassword) {
            errorEl.textContent = 'Passwords do not match';
            errorEl.classList.remove('hidden');
            return;
        }

        // Valid password
        state.filePassword = password;
        state.passwordProtection = true;
        hidePasswordModal();
        showNotification('🔒 Files will be protected with your password', 'success');
        startProcessing();
    });

    // Copy password to clipboard
    copyPasswordBtn.addEventListener('click', async () => {
        const password = document.getElementById('generatedPasswordValue').textContent;
        try {
            await navigator.clipboard.writeText(password);
            copyPasswordBtn.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 12L11 14L15 10M21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                Copied!
            `;
            showNotification('Password copied to clipboard!', 'success');
            setTimeout(() => {
                copyPasswordBtn.innerHTML = `
                    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M8 5H6C5.46957 5 4.96086 5.21071 4.58579 5.58579C4.21071 5.96086 4 6.46957 4 7V19C4 19.5304 4.21071 20.0391 4.58579 20.4142C4.96086 20.7893 5.46957 21 6 21H16C16.5304 21 17.0391 20.7893 17.4142 20.4142C17.7893 20.0391 18 19.5304 18 19V18M8 5C8 5.53043 8.21071 6.03914 8.58579 6.41421C8.96086 6.78929 9.46957 7 10 7H12C12.5304 7 13.0391 6.78929 13.4142 6.41421C13.7893 6.03914 14 5.53043 14 5M8 5C8 4.46957 8.21071 3.96086 8.58579 3.58579C8.96086 3.21071 9.46957 3 10 3H12C12.5304 3 13.0391 3.21071 13.4142 3.58579C13.7893 3.96086 14 4.46957 14 5M14 5H16C16.5304 5 17.0391 5.21071 17.4142 5.58579C17.7893 5.96086 18 6.46957 18 7V10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    Copy
                `;
            }, 2000);
        } catch (err) {
            showNotification('Failed to copy password', 'error');
        }
    });

    // Toggle password visibility
    togglePassword1.addEventListener('click', () => {
        const input = document.getElementById('customPasswordField');
        input.type = input.type === 'password' ? 'text' : 'password';
    });

    togglePassword2.addEventListener('click', () => {
        const input = document.getElementById('confirmPasswordField');
        input.type = input.type === 'password' ? 'text' : 'password';
    });

    // Error modal handlers
    closeErrorModalBtn.addEventListener('click', () => {
        hideErrorModal();
    });

    retryProcessingBtn.addEventListener('click', () => {
        hideErrorModal();
        showPasswordModal();
    });
}

// ===================================
// ERROR MODAL
// ===================================
function showErrorModal() {
    const modal = document.getElementById('errorModal');
    modal.classList.remove('hidden');

    // Reset processing state
    state.processing = false;
    elements.processBtn.disabled = false;
    elements.processBtn.querySelector('.btn-text').textContent = 'Start Processing';
    elements.progressSection.classList.add('hidden');
}

function hideErrorModal() {
    const modal = document.getElementById('errorModal');
    modal.classList.add('hidden');
}
window.changeOutputDirectory = changeOutputDirectory;