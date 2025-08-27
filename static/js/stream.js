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
        
        // Tracking & detection throttle config
        // 追踪与检测节流配置
        this.tracker = new window.Tracker({
            enableReID: true,
            focusClasses: ['person'],
            autoCreate: false
        });
        this.detectEveryMs = 180; // Run detection ~5-6 FPS 检测节流（毫秒）
        this.lastDetectionTime = 0;
        this.lastScaledPredictions = [];
        this.debug = false; // 控制调试日志开关
        
        // Hover tooltip state
        // 悬停提示状态
        this.hoverTooltip = {
            active: false,
            x: 0,
            y: 0,
            objectInfo: null,
            element: null
        };
        
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

        // Click-to-lock on detection overlay
        // 在检测画布上捕获点击以锁定目标
        if (this.detectionCanvas) {
            this.detectionCanvas.style.pointerEvents = 'auto';
            this.detectionCanvas.addEventListener('click', (e) => this.onCanvasClick(e));
            // Right-click to clear all tracks 右键清空所有轨迹
            this.detectionCanvas.addEventListener('contextmenu', (e) => {
                e.preventDefault();
                if (this.tracker) {
                    this.tracker.clear();
                    this.lastScaledPredictions = [];
                    this.clearOverlay();
                    this.showAlert('Cleared all tracks', 'warning');
                    this.updateTrackingList();
                }
            });
            
            // Hover to show detection information
            // 悬停显示检测目标信息
            this.detectionCanvas.addEventListener('mousemove', (e) => this.onCanvasMouseMove(e));
            
            // Clear tooltip when mouse leaves canvas
            // 鼠标离开画布时清除提示
            this.detectionCanvas.addEventListener('mouseleave', () => this.hideTooltip());
        }

        // Keyboard shortcuts 键盘快捷键
        document.addEventListener('keydown', (e) => {
            if (!this.isActive || !this.tracker) return;
            // ignore when typing in input/textarea/select or contenteditable
            const tag = (e.target && e.target.tagName) ? e.target.tagName.toUpperCase() : '';
            if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag) || (e.target && e.target.isContentEditable)) return;
            // ignore auto-repeated keydown when key is held
            if (e.repeat) return;
            const k = e.key;
            if (k === 'l' || k === 'L') {
                // Unlock all 解锁全部
                for (const t of this.tracker.getTracks()) this.tracker.unlock(t.id);
                this.clearOverlay();
                this.drawTracks();
                this.showAlert('All tracks unlocked', 'info');
                this.updateTrackingList();
            } else if (k === 'c' || k === 'C') {
                // Clear all 清空全部
                this.tracker.clear();
                this.lastScaledPredictions = [];
                this.clearOverlay();
                this.showAlert('Cleared all tracks', 'warning');
                this.updateTrackingList();
            } else if (k === 'a' || k === 'A') {
                // Toggle auto-create 切换自动创建
                this.tracker.autoCreate = !this.tracker.autoCreate;
                this.showAlert(`Auto-create ${this.tracker.autoCreate ? 'ON' : 'OFF'}`, 'info');
            }
        });

        // Window resize handling 窗口缩放自适应
        window.addEventListener('resize', () => this.onWindowResize());
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
            if (this.tracker) this.tracker.clear(); // reset tracker on new stream
            this.updateTrackingList();
            
            // Update button states
            this.updateButtonStates(true);
            
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
            if (this.tracker) this.tracker.clear();
            this.lastScaledPredictions = [];
            this.clearOverlay();
            this.updateTrackingList();
            
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
            if (this.debug) console.log('Frame loaded:', this.frameImage.naturalWidth, 'x', this.frameImage.naturalHeight);
            
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
                if (this.debug) console.log('Updating canvas size to:', this.frameImage.naturalWidth, 'x', this.frameImage.naturalHeight);
                this.setupCanvases(this.frameImage.naturalWidth, this.frameImage.naturalHeight);
            }
            
            // Clear canvas before drawing
            this.videoContext.clearRect(0, 0, this.videoCanvas.width, this.videoCanvas.height);
            
            // Draw frame to canvas
            try {
                this.videoContext.drawImage(this.frameImage, 0, 0, this.videoCanvas.width, this.videoCanvas.height);
                if (this.debug) console.log('Frame drawn to canvas successfully');
            } catch (error) {
                console.error('Error drawing frame to canvas:', error);
                return;
            }
            
            // Update frame counter
            this.frameCount++;
            
            // Trigger AI detection if enabled
            if (this.isActive && window.detectionManager && window.detectionManager.isModelLoaded) {
                if (this.debug) console.log('Triggering AI detection...');
                this.performDetection();
            } else {
                if (this.debug) console.log('AI detection not ready:', {
                    isActive: this.isActive,
                    detectionManager: !!window.detectionManager,
                    modelLoaded: window.detectionManager ? window.detectionManager.isModelLoaded : false
                });
            }
            
            // Schedule next frame fetch immediately to maintain continuous flow
            if (this.isActive) {
                setTimeout(() => this.fetchNextFrame(), 16); // ~60 FPS for ultra smooth video
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
        if (this.debug) console.log('Fetching frame from:', frameUrl);
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
        
        // Throttle detection for performance; predict-only in between
        // 为性能进行检测节流；间隔帧仅做预测
        const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
        const shouldDetect = !this.lastDetectionTime || (now - this.lastDetectionTime >= this.detectEveryMs);

        if (!shouldDetect) {
            try {
                if (this.tracker) this.tracker.predictOnly();
                this.clearOverlay();
                this.drawTracks();
            } catch (e) {
                console.warn('Predict-only step failed:', e);
            }
            return;
        }

        try {
            // 运行检测
            const predictions = await window.detectionManager.detectObjects(this.frameImage);

            // 将检测框缩放到画布坐标，用于追踪与点击匹配
            const scaleX = this.detectionCanvas.width / (this.frameImage?.naturalWidth || this.detectionCanvas.width);
            const scaleY = this.detectionCanvas.height / (this.frameImage?.naturalHeight || this.detectionCanvas.height);
            const scaled = predictions.map(p => {
                const [x, y, w, h] = p.bbox;
                return {
                    bbox: [x * scaleX, y * scaleY, w * scaleX, h * scaleY],
                    score: p.score,
                    class: p.class
                };
            });
            this.lastScaledPredictions = scaled;

            // 更新追踪器（传入视频画布上下文以提取外观特征）
            if (this.tracker && this.videoContext) {
                this.tracker.update(scaled, this.videoContext);
            }

            // 仅绘制追踪框（不显示检测框）
            this.clearOverlay();
            this.drawTracks();

            this.lastDetectionTime = now;
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
     * Draw tracks (IDs, boxes, and trajectories)
     * 绘制追踪框与轨迹（ID/锁定状态）
     */
    drawTracks() {
        if (!this.detectionContext || !this.tracker) return;
        const ctx = this.detectionContext;

        // Only draw locked tracks so boxes appear only after user clicks
        const tracks = this.tracker.getTracks().filter(t => t.locked);
        for (const t of tracks) {
            const b = t.bbox;
            ctx.save();
            ctx.strokeStyle = t.color;
            ctx.lineWidth = t.locked ? 3 : 2;
            ctx.setLineDash(t.locked ? [6, 4] : []);
            ctx.strokeRect(b.x, b.y, b.w, b.h);

            // Draw ID label
            const label = `ID ${t.id}${t.locked ? ' • LOCK' : ''}`;
            ctx.font = '14px Arial';
            const tw = ctx.measureText(label).width;
            ctx.fillStyle = t.color;
            ctx.globalAlpha = 0.9;
            ctx.fillRect(b.x, Math.max(0, b.y - 22), tw + 10, 20);
            ctx.fillStyle = '#fff';
            ctx.globalAlpha = 1.0;
            ctx.fillText(label, b.x + 5, Math.max(14, b.y - 6));

            ctx.restore();
        }
    }

    /**
     * Clear only the overlay canvas
     * 清空叠加层画布
     */
    clearOverlay() {
        if (this.detectionContext) {
            this.detectionContext.clearRect(0, 0, this.detectionCanvas.width, this.detectionCanvas.height);
        }
    }

    /**
     * Handle click on overlay: map to canvas coords and lock a track
     * 处理画布点击：坐标映射并锁定目标
     */
    async onCanvasClick(e) {
        if (!this.detectionCanvas || !this.tracker) return;
        const rect = this.detectionCanvas.getBoundingClientRect();
        const scaleX = this.detectionCanvas.width / rect.width;
        const scaleY = this.detectionCanvas.height / rect.height;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        
        // Toggle lock: if clicking hits a track, toggle lock/unlock directly; otherwise try detection-based locking
        const tracks = this.tracker.getTracks();
        const hit = tracks.find(t => {
            const b = t.bbox; return x >= b.x && y >= b.y && x <= b.x + b.w && y <= b.y + b.h;
        });
        if (hit) {
            if (hit.locked) {
                this.tracker.unlock(hit.id);
                this.showAlert(`Unlocked target #${hit.id}`, 'info');
            } else {
                hit.locked = true;
                hit.lostFrames = 0;
                this.showAlert(`Locked target #${hit.id}`, 'success');
            }
            this.clearOverlay();
            this.drawTracks();
            this.updateTrackingList();
            return;
        }

        let id = this.tracker.lockFromPoint(x, y, this.lastScaledPredictions || [], this.videoContext);
        if (!id && this.frameImage && window.detectionManager && window.detectionManager.isModelLoaded) {
            // One-shot detection fallback to ensure immediate lock on click
            try {
                const predictions = await window.detectionManager.detectObjects(this.frameImage);
                const scaleX2 = this.detectionCanvas.width / (this.frameImage?.naturalWidth || this.detectionCanvas.width);
                const scaleY2 = this.detectionCanvas.height / (this.frameImage?.naturalHeight || this.detectionCanvas.height);
                const scaled = predictions.map(p => {
                    const [bx, by, bw, bh] = p.bbox;
                    return { bbox: [bx * scaleX2, by * scaleY2, bw * scaleX2, bh * scaleY2], score: p.score, class: p.class };
                });
                this.lastScaledPredictions = scaled;
                if (this.tracker && this.videoContext) this.tracker.update(scaled, this.videoContext);
                id = this.tracker.lockFromPoint(x, y, this.lastScaledPredictions || [], this.videoContext);
            } catch (err) {
                console.warn('One-shot detection on click failed:', err);
            }
        }

        if (id) {
            this.showAlert(`Locked target #${id}`, 'success');
            // 立即绘制最新轨迹
            this.clearOverlay();
            this.drawTracks();
            this.updateTrackingList();
        }
    }

    /**
     * Handle window resize to keep canvases sized to container
     * 处理窗口缩放，保持画布尺寸与容器一致
     */
    onWindowResize() {
        if (!this.videoCanvas || !this.detectionCanvas) return;
        const w = this.videoCanvas.width || 800;
        const h = this.videoCanvas.height || 450;
        this.setupCanvases(w, h);
        // 重绘轨迹避免缩放后残影
        this.clearOverlay();
        this.drawTracks();
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
        this.updateTrackingList();
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
    
    /**
     * Handle mouse movement on canvas to show object hover tooltip
     * 处理画布上的鼠标移动以显示目标悬停提示
     */
    onCanvasMouseMove(e) {
        if (!this.isActive || !this.lastScaledPredictions || this.lastScaledPredictions.length === 0) {
            this.hideTooltip();
            return;
        }
        
        // Convert mouse position to canvas coordinates
        // 将鼠标位置转换为画布坐标
        const rect = this.detectionCanvas.getBoundingClientRect();
        const scaleX = this.detectionCanvas.width / rect.width;
        const scaleY = this.detectionCanvas.height / rect.height;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        
        // Check if mouse is over any detection
        // 检查鼠标是否在任何检测目标上方
        let hoveredObject = null;
        
        // First check if over any tracked objects that are not locked (we want hover to prioritize untracked objects)
        // 首先检查是否在任何未锁定的跟踪对象上方（我们希望悬停优先考虑未追踪的对象）
        const tracks = this.tracker.getTracks().filter(t => !t.locked);
        for (const track of tracks) {
            const b = track.bbox;
            if (x >= b.x && y >= b.y && x <= b.x + b.w && y <= b.y + b.h) {
                this.hideTooltip(); // Don't show tooltip for already tracked objects
                return;
            }
        }
        
        // Then check raw detections
        // 然后检查原始检测结果
        for (const pred of this.lastScaledPredictions) {
            const [bx, by, bw, bh] = pred.bbox;
            if (x >= bx && y >= by && x <= bx + bw && y <= by + bh) {
                hoveredObject = pred;
                break;
            }
        }
        
        if (hoveredObject) {
            // We found an object under the mouse, show tooltip
            // 找到鼠标下方的对象，显示提示
            this.showTooltip(e.clientX, e.clientY, hoveredObject);
        } else {
            // No object under mouse
            // 鼠标下方没有对象
            this.hideTooltip();
        }
    }
    
    /**
     * Create and show tooltip with object information
     * 创建并显示包含对象信息的提示框
     */
    showTooltip(clientX, clientY, objectInfo) {
        // Create tooltip if it doesn't exist
        // 如果提示框不存在则创建
        if (!this.hoverTooltip.element) {
            const tooltip = document.createElement('div');
            tooltip.className = 'detection-tooltip';
            document.body.appendChild(tooltip);
            this.hoverTooltip.element = tooltip;
        }
        
        // Format confidence score
        // 格式化置信度分数
        const confidence = (objectInfo.score * 100).toFixed(1);
        
        // Update tooltip content
        // 更新提示框内容
        this.hoverTooltip.element.innerHTML = `
            <div class="tooltip-content">
                <div class="tooltip-class">${this.formatClassName(objectInfo.class)}</div>
                <div class="tooltip-confidence">${confidence}% 置信度</div>
                <div class="tooltip-instruction">点击即可追踪 / Click to track</div>
            </div>
        `;
        
        // Position tooltip near mouse but not under it
        // 将提示框定位在鼠标附近但不在其下方
        const tooltipRect = this.hoverTooltip.element.getBoundingClientRect();
        let left = clientX + 15; // Offset from cursor
        let top = clientY - tooltipRect.height - 10; // Position above cursor
        
        // Keep tooltip within window bounds
        // 保持提示框在窗口范围内
        if (left + tooltipRect.width > window.innerWidth) {
            left = clientX - tooltipRect.width - 10; // Position to left if too far right
        }
        if (top < 0) {
            top = clientY + 20; // Position below cursor if too high
        }
        
        // Update tooltip position
        // 更新提示框位置
        this.hoverTooltip.element.style.left = `${left}px`;
        this.hoverTooltip.element.style.top = `${top}px`;
        this.hoverTooltip.element.style.display = 'block';
        
        // Update hover state
        // 更新悬停状态
        this.hoverTooltip.active = true;
        this.hoverTooltip.objectInfo = objectInfo;
    }
    
    /**
     * Hide tooltip
     * 隐藏提示框
     */
    hideTooltip() {
        if (this.hoverTooltip.element) {
            this.hoverTooltip.element.style.display = 'none';
        }
        this.hoverTooltip.active = false;
        this.hoverTooltip.objectInfo = null;
    }
    
    /**
     * Format class name for display (borrowed from detection.js)
     * 格式化类名以便显示（从detection.js借用）
     */
    formatClassName(className) {
        return className.split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
}

/**
 * Update tracking list UI (locked tracks) with individual control buttons
 * 更新追踪列表界面（已锁定目标）并添加单独控制按钮
 */
StreamManager.prototype.updateTrackingList = function () {
    const container = document.getElementById('trackingList');
    if (!container || !this.tracker) return;
    
    const locked = this.tracker.getTracks().filter(t => t.locked);
    
    // Show empty state when no locked targets / 无锁定目标时显示空状态
    if (locked.length === 0) {
        container.innerHTML = `
            <div class="no-detections text-center py-4">
                <i class="fas fa-bullseye fa-2x text-muted mb-2"></i>
                <p class="text-muted mb-0">未选择目标</p>
                <small class="text-muted">点击画面中的目标以开始跟踪</small>
            </div>
        `;
        return;
    }
    
    // Generate HTML for each locked target with control buttons
    // 为每个锁定目标生成带控制按钮的HTML
    const html = locked.map(t => `
        <div class="object-item fade-in d-flex justify-content-between align-items-center">
            <div class="flex-grow-1">
                <div class="object-name">ID ${t.id}</div>
                <div class="confidence-score">Locked</div>
            </div>
            <div class="btn-group btn-group-sm" role="group">
                <button type="button" 
                        class="btn btn-outline-warning btn-sm" 
                        onclick="window.streamManager.unlockTarget(${t.id})"
                        title="Unlock target / 解锁目标">
                    <i class="fas fa-unlock"></i>
                </button>
                <button type="button" 
                        class="btn btn-outline-danger btn-sm" 
                        onclick="window.streamManager.removeTarget(${t.id})"
                        title="Remove target / 移除目标">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </div>
    `).join('');
    
    container.innerHTML = html;
};

/**
 * Unlock a specific target by ID (keep track but unlock)
 * 解锁指定ID的目标（保留追踪但解锁）
 */
StreamManager.prototype.unlockTarget = function(targetId) {
    if (!this.tracker) return;
    
    // Find and unlock the target / 查找并解锁目标
    const track = this.tracker.getTracks().find(t => t.id === targetId);
    if (track && track.locked) {
        this.tracker.unlock(targetId);
        this.showAlert(`Unlocked target #${targetId}`, 'info');
        
        // Update UI immediately / 立即更新界面
        this.clearOverlay();
        this.drawTracks();
        this.updateTrackingList();
    }
};

/**
 * Remove a specific target by ID (completely remove from tracker)
 * 移除指定ID的目标（从追踪器中完全移除）
 */
StreamManager.prototype.removeTarget = function(targetId) {
    if (!this.tracker) return;
    
    // Find the target and remove it / 查找目标并移除
    const tracks = this.tracker.getTracks();
    const targetIndex = tracks.findIndex(t => t.id === targetId);
    
    if (targetIndex !== -1) {
        // Remove from tracker's internal array / 从追踪器内部数组中移除
        tracks.splice(targetIndex, 1);
        this.showAlert(`Removed target #${targetId}`, 'warning');
        
        // Update UI immediately / 立即更新界面
        this.clearOverlay();
        this.drawTracks();
        this.updateTrackingList();
    }
};


// Initialize managers when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initialize stream manager
    window.streamManager = new StreamManager();
    
    console.log('Stream application initialized');
});
