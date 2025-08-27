/**
 * Stream Manager
 * Handles YouTube stream processing and video display
 */
class StreamManager {
    constructor() {
        this.isActive = false;
        this.videoCanvas = null;
        this.detectionCanvas = null;
        this.videoContext = null;
        this.detectionContext = null;
        this.currentStream = null;
        this.detectionInterval = null;
        this.statusInterval = null;
        
        this.initializeElements();
        this.setupEventListeners();
    }

    /**
     * Initialize canvas elements and contexts
     */
    initializeElements() {
        this.videoCanvas = document.getElementById('videoCanvas');
        this.detectionCanvas = document.getElementById('detectionCanvas');
        
        if (this.videoCanvas) {
            this.videoContext = this.videoCanvas.getContext('2d');
        }
        
        if (this.detectionCanvas) {
            this.detectionContext = this.detectionCanvas.getContext('2d');
        }
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Start button
        const startBtn = document.getElementById('startBtn');
        if (startBtn) {
            startBtn.addEventListener('click', () => this.startStream());
        }

        // Stop button
        const stopBtn = document.getElementById('stopBtn');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopStream());
        }

        // URL input enter key
        const urlInput = document.getElementById('youtubeUrl');
        if (urlInput) {
            urlInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.startStream();
                }
            });
        }
    }

    /**
     * Start stream processing
     */
    async startStream() {
        const urlInput = document.getElementById('youtubeUrl');
        const url = urlInput ? urlInput.value.trim() : '';

        if (!url) {
            this.showAlert('Please enter a YouTube URL', 'danger');
            return;
        }

        if (!this.validateYouTubeUrl(url)) {
            this.showAlert('Please enter a valid YouTube URL', 'danger');
            return;
        }

        // Check if model is loaded
        if (!window.detectionManager || !window.detectionManager.isModelLoaded) {
            this.showAlert('AI model is still loading. Please wait...', 'info');
            return;
        }

        try {
            this.showLoading(true);
            this.updateButtonStates(true);

            // Send request to start stream
            const response = await fetch('/start_stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: url })
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Failed to start stream');
            }

            // Start video processing
            await this.initializeVideoDisplay();
            this.startDetection();
            this.startStatusMonitoring();
            
            this.isActive = true;
            this.showAlert('Stream started successfully!', 'success');
            
        } catch (error) {
            console.error('Error starting stream:', error);
            this.showAlert(error.message, 'danger');
            this.updateButtonStates(false);
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Stop stream processing
     */
    async stopStream() {
        try {
            // Stop local processing
            this.stopDetection();
            this.stopStatusMonitoring();
            
            // Send request to stop stream
            const response = await fetch('/stop_stream', {
                method: 'POST'
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Failed to stop stream');
            }

            this.isActive = false;
            this.hideVideoDisplay();
            this.updateButtonStates(false);
            this.clearDetectionStats();
            
            this.showAlert('Stream stopped', 'info');
            
        } catch (error) {
            console.error('Error stopping stream:', error);
            this.showAlert(error.message, 'danger');
        }
    }

    /**
     * Initialize video display
     */
    async initializeVideoDisplay() {
        const placeholder = document.getElementById('videoPlaceholder');
        const streamStatus = document.getElementById('streamStatus');
        
        // Hide placeholder, show canvas and status
        if (placeholder) placeholder.classList.add('d-none');
        if (this.videoCanvas) this.videoCanvas.classList.remove('d-none');
        if (this.detectionCanvas) this.detectionCanvas.classList.remove('d-none');
        if (streamStatus) streamStatus.classList.remove('d-none');

        // Create video element for processing
        this.videoElement = document.createElement('video');
        this.videoElement.crossOrigin = 'anonymous';
        this.videoElement.muted = true;
        this.videoElement.autoplay = true;
        this.videoElement.style.display = 'none';
        document.body.appendChild(this.videoElement);

        // Set video source to our stream endpoint
        this.videoElement.src = '/video_feed';
        
        // Wait for video to load
        return new Promise((resolve) => {
            this.videoElement.addEventListener('loadedmetadata', () => {
                this.setupCanvases();
                resolve();
            });
        });
    }

    /**
     * Setup canvas dimensions
     */
    setupCanvases() {
        if (!this.videoElement || !this.videoCanvas || !this.detectionCanvas) return;

        const containerWidth = this.videoCanvas.parentElement.offsetWidth;
        const videoAspectRatio = this.videoElement.videoWidth / this.videoElement.videoHeight;
        
        const canvasWidth = Math.min(containerWidth, 800);
        const canvasHeight = canvasWidth / videoAspectRatio;

        // Set canvas dimensions
        this.videoCanvas.width = canvasWidth;
        this.videoCanvas.height = canvasHeight;
        this.detectionCanvas.width = canvasWidth;
        this.detectionCanvas.height = canvasHeight;

        // Set CSS dimensions
        this.videoCanvas.style.width = canvasWidth + 'px';
        this.videoCanvas.style.height = canvasHeight + 'px';
        this.detectionCanvas.style.width = canvasWidth + 'px';
        this.detectionCanvas.style.height = canvasHeight + 'px';
    }

    /**
     * Start object detection processing
     */
    startDetection() {
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
        }

        this.detectionInterval = setInterval(async () => {
            if (!this.isActive || !this.videoElement || !window.detectionManager) return;

            try {
                // Draw current video frame to canvas
                if (this.videoContext && this.videoElement.videoWidth > 0) {
                    this.videoContext.drawImage(
                        this.videoElement, 
                        0, 0, 
                        this.videoCanvas.width, 
                        this.videoCanvas.height
                    );
                }

                // Perform object detection
                const predictions = await window.detectionManager.detectObjects(this.videoElement);
                
                // Draw detection results
                this.drawDetections(predictions);
                
            } catch (error) {
                console.error('Detection error:', error);
            }
        }, 100); // 10 FPS detection rate
    }

    /**
     * Draw detection bounding boxes and labels
     */
    drawDetections(predictions) {
        if (!this.detectionContext || !predictions) return;

        // Clear previous detections
        this.detectionContext.clearRect(0, 0, this.detectionCanvas.width, this.detectionCanvas.height);

        predictions.forEach(prediction => {
            const [x, y, width, height] = prediction.bbox;
            const confidence = (prediction.score * 100).toFixed(1);
            
            // Scale coordinates to canvas size
            const scaleX = this.detectionCanvas.width / this.videoElement.videoWidth;
            const scaleY = this.detectionCanvas.height / this.videoElement.videoHeight;
            
            const scaledX = x * scaleX;
            const scaledY = y * scaleY;
            const scaledWidth = width * scaleX;
            const scaledHeight = height * scaleY;

            // Draw bounding box
            this.detectionContext.strokeStyle = '#471580';
            this.detectionContext.lineWidth = 3;
            this.detectionContext.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

            // Draw label background
            const label = `${prediction.class} (${confidence}%)`;
            this.detectionContext.font = '14px Arial';
            const textWidth = this.detectionContext.measureText(label).width;
            
            this.detectionContext.fillStyle = '#471580';
            this.detectionContext.fillRect(scaledX, scaledY - 25, textWidth + 10, 25);

            // Draw label text
            this.detectionContext.fillStyle = 'white';
            this.detectionContext.fillText(label, scaledX + 5, scaledY - 8);
        });
    }

    /**
     * Start monitoring stream status
     */
    startStatusMonitoring() {
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
        }

        this.statusInterval = setInterval(async () => {
            try {
                const response = await fetch('/stream_status');
                const status = await response.json();
                
                this.updateStreamStatus(status);
                
                if (!status.active && this.isActive) {
                    // Stream stopped unexpectedly
                    this.handleStreamError('Stream connection lost');
                }
            } catch (error) {
                console.error('Status monitoring error:', error);
            }
        }, 1000);
    }

    /**
     * Update stream status display
     */
    updateStreamStatus(status) {
        const fpsElement = document.getElementById('streamFps');
        const frameCountElement = document.getElementById('frameCount');

        if (fpsElement) {
            fpsElement.textContent = status.fps ? status.fps.toFixed(1) : '0.0';
        }
        
        if (frameCountElement) {
            frameCountElement.textContent = status.frame_count || 0;
        }
    }

    /**
     * Stop detection processing
     */
    stopDetection() {
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
    }

    /**
     * Stop status monitoring
     */
    stopStatusMonitoring() {
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
            this.statusInterval = null;
        }
    }

    /**
     * Hide video display and show placeholder
     */
    hideVideoDisplay() {
        const placeholder = document.getElementById('videoPlaceholder');
        const streamStatus = document.getElementById('streamStatus');
        
        if (placeholder) placeholder.classList.remove('d-none');
        if (this.videoCanvas) this.videoCanvas.classList.add('d-none');
        if (this.detectionCanvas) this.detectionCanvas.classList.add('d-none');
        if (streamStatus) streamStatus.classList.add('d-none');

        // Remove video element
        if (this.videoElement) {
            this.videoElement.remove();
            this.videoElement = null;
        }
    }

    /**
     * Update button states
     */
    updateButtonStates(isStarted) {
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');

        if (startBtn && stopBtn) {
            if (isStarted) {
                startBtn.classList.add('d-none');
                stopBtn.classList.remove('d-none');
            } else {
                startBtn.classList.remove('d-none');
                stopBtn.classList.add('d-none');
            }
        }
    }

    /**
     * Clear detection statistics
     */
    clearDetectionStats() {
        if (window.detectionManager) {
            window.detectionManager.clearStats();
        }
    }

    /**
     * Handle stream errors
     */
    handleStreamError(message) {
        this.stopDetection();
        this.stopStatusMonitoring();
        this.isActive = false;
        this.hideVideoDisplay();
        this.updateButtonStates(false);
        this.showAlert(message, 'danger');
    }

    /**
     * Validate YouTube URL
     */
    validateYouTubeUrl(url) {
        const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+/;
        return youtubeRegex.test(url);
    }

    /**
     * Show loading modal
     */
    showLoading(show) {
        const modal = document.getElementById('loadingModal');
        if (modal) {
            const modalInstance = bootstrap.Modal.getOrCreateInstance(modal);
            if (show) {
                modalInstance.show();
            } else {
                modalInstance.hide();
            }
        }
    }

    /**
     * Show alert message
     */
    showAlert(message, type) {
        const alertContainer = document.getElementById('alertContainer');
        if (!alertContainer) return;

        const alertId = 'alert_' + Date.now();
        const alertHTML = `
            <div class="alert alert-${type} alert-dismissible fade show" id="${alertId}" role="alert">
                <i class="fas fa-${this.getAlertIcon(type)} me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;

        alertContainer.insertAdjacentHTML('beforeend', alertHTML);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alertElement = document.getElementById(alertId);
            if (alertElement) {
                const alert = bootstrap.Alert.getOrCreateInstance(alertElement);
                alert.close();
            }
        }, 5000);
    }

    /**
     * Get icon for alert type
     */
    getAlertIcon(type) {
        const icons = {
            success: 'check-circle',
            danger: 'exclamation-triangle',
            warning: 'exclamation-circle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    }
}

// Dark mode functionality
class ThemeManager {
    constructor() {
        this.isDarkMode = localStorage.getItem('darkMode') === 'true';
        this.setupDarkMode();
        this.setupToggle();
    }

    setupDarkMode() {
        if (this.isDarkMode) {
            document.documentElement.setAttribute('data-theme', 'dark');
            this.updateToggleButton();
        }
    }

    setupToggle() {
        const toggle = document.getElementById('darkModeToggle');
        if (toggle) {
            toggle.addEventListener('click', () => this.toggleDarkMode());
        }
    }

    toggleDarkMode() {
        this.isDarkMode = !this.isDarkMode;
        localStorage.setItem('darkMode', this.isDarkMode);
        
        if (this.isDarkMode) {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }
        
        this.updateToggleButton();
    }

    updateToggleButton() {
        const toggle = document.getElementById('darkModeToggle');
        if (toggle) {
            if (this.isDarkMode) {
                toggle.innerHTML = '<i class="fas fa-sun"></i> Light Mode';
            } else {
                toggle.innerHTML = '<i class="fas fa-moon"></i> Dark Mode';
            }
        }
    }
}

// Initialize managers when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initialize stream manager
    window.streamManager = new StreamManager();
    
    // Initialize theme manager
    window.themeManager = new ThemeManager();
    
    console.log('Stream application initialized');
});
