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
        
        console.log('Canvas elements found:', {
            videoCanvas: !!this.videoCanvas,
            detectionCanvas: !!this.detectionCanvas
        });
        
        if (this.videoCanvas) {
            this.videoContext = this.videoCanvas.getContext('2d');
            console.log('Video context initialized:', !!this.videoContext);
        } else {
            console.error('Video canvas not found!');
        }
        
        if (this.detectionCanvas) {
            this.detectionContext = this.detectionCanvas.getContext('2d');
            console.log('Detection context initialized:', !!this.detectionContext);
        } else {
            console.error('Detection canvas not found!');
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

        try {
            // Start backend stream processing
            const response = await fetch('/start_stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            console.log('Stream started:', result);
            
            // Update UI state
            this.isActive = true;
            this.currentStream = url;
            this.frameCount = 0;
            
            // Update button states
            this.updateButtonStates();
            
            // Start status monitoring
            this.startStatusMonitoring();
            
            // Setup video display
            await this.setupVideoDisplay();
            
            // Wait a moment for backend to be ready, then start fetching
            setTimeout(() => {
                if (this.isActive) {
                    this.startFrameFetching();
                }
            }, 2000);
            
        } catch (error) {
            console.error('Failed to start stream:', error);
            this.handleStreamError(`Failed to start stream: ${error.message}`);
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
    async setupVideoDisplay() {
        const placeholder = document.getElementById('videoPlaceholder');
        const streamStatus = document.getElementById('streamStatus');
        
        // Hide placeholder, show canvas and status
        if (placeholder) placeholder.classList.add('d-none');
        if (this.videoCanvas) this.videoCanvas.classList.remove('d-none');
        if (this.detectionCanvas) this.detectionCanvas.classList.remove('d-none');
        if (streamStatus) streamStatus.classList.remove('d-none');

        // Initialize canvas with default size
        this.setupCanvases(800, 450); // 16:9 aspect ratio default
        
        return Promise.resolve();
    }

    /**
     * Setup canvas dimensions
     */
    setupCanvases(defaultWidth = 800, defaultHeight = 450) {
        if (!this.videoCanvas || !this.detectionCanvas) return;

        const containerWidth = this.videoCanvas.parentElement.offsetWidth;
        const canvasWidth = Math.min(containerWidth, defaultWidth);
        const canvasHeight = (canvasWidth / defaultWidth) * defaultHeight;

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
     * Start frame fetching from backend
     */
    startFrameFetching() {
        console.log('Starting frame fetching...');
        
        this.frameLoadCount = 0;
        this.frameErrorCount = 0;
        
        // Start fetching frames
        this.fetchNextFrame();
    }
    
    /**
     * Fetch next frame from backend
     */
    fetchNextFrame() {
        if (!this.isActive) return;
        
        // Create a new Image object for each frame to avoid caching issues
        this.frameImage = new Image();
        this.frameImage.crossOrigin = 'anonymous';
        
        // Set up event handlers for the new image
        this.frameImage.onload = () => {
            console.log('Frame loaded:', this.frameImage.naturalWidth, 'x', this.frameImage.naturalHeight);
            
            // Validate canvas context exists
            if (!this.videoContext) {
                console.error('Video context not available!');
                return;
            }
            
            // Validate frame dimensions
            if (this.frameImage.naturalWidth === 0 || this.frameImage.naturalHeight === 0) {
                console.warn('Invalid frame dimensions:', this.frameImage.naturalWidth, 'x', this.frameImage.naturalHeight);
                this.frameErrorCount++;
                if (this.frameErrorCount >= 10) {
                    this.stopDetection();
                    this.handleStreamError('Invalid frame dimensions');
                    return;
                }
                setTimeout(() => this.fetchNextFrame(), 100);
                return;
            }
            
            // Reset error count on successful load
            this.frameErrorCount = 0;
            
            // Update canvas size if needed
            const canvasWidth = this.videoCanvas.width;
            const canvasHeight = this.videoCanvas.height;
            
            if (canvasWidth !== this.frameImage.naturalWidth || canvasHeight !== this.frameImage.naturalHeight) {
                console.log('Updating canvas size to:', this.frameImage.naturalWidth, 'x', this.frameImage.naturalHeight);
                this.setupCanvases(this.frameImage.naturalWidth, this.frameImage.naturalHeight);
            }
            
            // Clear canvas before drawing
            this.videoContext.clearRect(0, 0, this.videoCanvas.width, this.videoCanvas.height);
            
            // Draw frame to canvas
            try {
                this.videoContext.drawImage(this.frameImage, 0, 0, this.videoCanvas.width, this.videoCanvas.height);
                console.log('Frame drawn to canvas successfully');
            } catch (error) {
                console.error('Error drawing frame to canvas:', error);
                return;
            }
            
            // Update frame counter
            this.frameCount++;
            
            // Trigger AI detection if enabled
            if (this.isActive && window.detectionManager && window.detectionManager.isModelLoaded) {
                console.log('Triggering AI detection...');
                this.performDetection();
            } else {
                console.log('AI detection not ready:', {
                    isActive: this.isActive,
                    detectionManager: !!window.detectionManager,
                    modelLoaded: window.detectionManager ? window.detectionManager.isModelLoaded : false
                });
            }
            
            // Schedule next frame fetch immediately to maintain continuous flow
            if (this.isActive) {
                setTimeout(() => this.fetchNextFrame(), 50); // ~20 FPS for smoother video
            }
        };
        
        this.frameImage.onerror = (error) => {
            console.warn('Failed to load frame:', error);
            console.warn('Frame URL was:', this.frameImage.src);
            this.frameErrorCount++;
            
            if (this.frameErrorCount >= 10) {
                this.stopDetection();
                this.handleStreamError('Failed to load video frames');
                return;
            }
            
            // Continue fetching even after error to maintain video flow
            if (this.isActive) {
                setTimeout(() => this.fetchNextFrame(), 500);
            }
        };
        
        const timestamp = Date.now() + Math.random() * 1000; // Add randomness to prevent caching
        const frameUrl = `./video_feed?t=${timestamp}&r=${Math.random()}`;
        console.log('Fetching frame from:', frameUrl);
        this.frameImage.src = frameUrl;
    }

    /**
     * Validate if current frame is ready for detection
     */
    isFrameValid() {
        if (!this.frameImage) {
            return false;
        }
        
        // Check if image is loaded and has valid dimensions
        if (this.frameImage.complete && 
            this.frameImage.naturalWidth > 0 && 
            this.frameImage.naturalHeight > 0) {
            return true;
        }
        
        return false;
    }
    
    /**
     * Update canvas size based on actual frame dimensions
     */
    updateCanvasSize(frameWidth, frameHeight) {
        if (!frameWidth || !frameHeight) return;
        
        const containerWidth = this.videoCanvas.parentElement.offsetWidth;
        const aspectRatio = frameWidth / frameHeight;
        
        const canvasWidth = Math.min(containerWidth, 800);
        const canvasHeight = canvasWidth / aspectRatio;

        if (this.videoCanvas.width !== canvasWidth || this.videoCanvas.height !== canvasHeight) {
            this.setupCanvases(canvasWidth, canvasHeight);
        }
    }

    /**
     * Start object detection processing
     */
    startDetection() {
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
        }

        this.detectionErrorCount = 0;
        
        this.detectionInterval = setInterval(async () => {
            if (!this.isActive || !this.frameImage || !window.detectionManager) return;

            // Validate frame before detection
            if (!this.isFrameValid()) {
                return; // Skip this detection cycle
            }

            try {
                // Perform object detection on the current frame
                const predictions = await window.detectionManager.detectObjects(this.frameImage);
                
                // Draw detection results
                this.drawDetections(predictions);
                
                // Reset error count on successful detection
                this.detectionErrorCount = 0;
                
            } catch (error) {
                this.detectionErrorCount++;
                console.error(`Detection error (${this.detectionErrorCount}):`, error.message);
                
                // Stop detection if too many errors
                if (this.detectionErrorCount >= 5) {
                    console.error('Too many detection errors, stopping detection');
                    this.stopDetection();
                    this.showAlert('AI detection stopped due to repeated errors', 'warning');
                }
            }
        }, 200); // 5 FPS detection rate (less frequent than frame updates)
    }

    /**
     * Perform detection on current frame
     */
    async performDetection() {
        if (!this.frameImage || !window.detectionManager || !window.detectionManager.isModelLoaded) {
            return;
        }
        
        try {
            const predictions = await window.detectionManager.detectObjects(this.frameImage);
            this.drawDetections(predictions);
        } catch (error) {
            console.error('Detection error:', error);
        }
    }

    /**
     * Draw detection bounding boxes and labels
     */
    drawDetections(predictions) {
        if (!this.detectionContext || !predictions) return;

        console.log('Drawing detections:', predictions.length);
        
        // Clear previous detections
        this.detectionContext.clearRect(0, 0, this.detectionCanvas.width, this.detectionCanvas.height);

        predictions.forEach(prediction => {
            const [x, y, width, height] = prediction.bbox;
            const confidence = (prediction.score * 100).toFixed(1);
            
            // Scale coordinates to canvas size (assuming detection was done on the displayed image)
            const scaleX = this.detectionCanvas.width / (this.frameImage?.naturalWidth || this.detectionCanvas.width);
            const scaleY = this.detectionCanvas.height / (this.frameImage?.naturalHeight || this.detectionCanvas.height);
            
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
        
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }
        
        // Reset error counters
        this.detectionErrorCount = 0;
        this.frameErrorCount = 0;
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

        // Clean up frame image
        if (this.frameImage) {
            this.frameImage.src = '';
            this.frameImage = null;
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
