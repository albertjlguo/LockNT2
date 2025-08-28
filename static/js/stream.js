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
        
        // Frame state (used for detection source)
        // 帧状态（检测输入源）
        this.frameImage = null;
        this.frameCount = 0;
        
        // MJPEG streaming state for smoother playback
        // 使用 MJPEG 推流提升流畅度
        this.mjpegImg = null; // offscreen <img> for multipart stream
        this.rafId = null;    // requestAnimationFrame id
        this._mjpegErrorNotified = false; // 避免重复告警
        this._stopping = false; // 正在主动停止标志，避免误报错误
        
        // Tracking & detection throttle config
        // 追踪与检测节流配置
        this.tracker = new window.Tracker({
            enableReID: true,
            focusClasses: ['person'],
            autoCreate: false
        });
        this.detectEveryMs = 150; // Increased detection frequency for smoother tracking 提高检测频率以实现更流畅的追踪
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
                    this.showAlert('已清空所有目标', 'warning');
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
                this.showAlert('已解锁所有目标', 'info');
                this.updateTrackingList();
            } else if (k === 'c' || k === 'C') {
                // Clear all 清空全部
                this.tracker.clear();
                this.lastScaledPredictions = [];
                this.clearOverlay();
                this.showAlert('已清空所有目标', 'warning');
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
            // Mark as stopping to suppress MJPEG error handling
            // 标记主动停止，抑制 MJPEG 错误提示/重连
            this._stopping = true;
            // Stop local processing
            this.stopDetection();
            this.stopStatusMonitoring();
            // Stop MJPEG stream if running  停止 MJPEG 连接
            this.stopMjpegStream();
            
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
            this._stopping = false;
            
        } catch (error) {
            console.error('Error stopping stream:', error);
            this.showAlert(error.message, 'danger');
            this._stopping = false;
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
        console.log('Starting MJPEG streaming...');
        this.startMjpegStream();
    }

    /**
     * Start MJPEG streaming via <img> and draw to canvas with rAF
     * 通过 <img> 拉取 MJPEG，并使用 rAF 绘制到画布
     */
    startMjpegStream() {
        try {
            // Reset error notify flag when starting a new MJPEG session
            this._mjpegErrorNotified = false;
            if (!this.mjpegImg) {
                this.mjpegImg = new Image();
                this.mjpegImg.crossOrigin = 'anonymous';
            }
            // Set current frame source for detection to mjpeg image
            this.frameImage = this.mjpegImg;
            
            // Handlers
            this.mjpegImg.onload = () => {
                // First frame arrived; start draw loop on demand
                if (!this.rafId && this.isActive) {
                    this.drawLoop();
                }
            };
            this.mjpegImg.onerror = (e) => {
                console.warn('MJPEG stream error', e);
                // Enhanced error handling with exponential backoff
                // 增强的错误处理，使用指数退避
                if (this._stopping || !this.isActive) {
                    return;
                }
                if (!this._mjpegErrorNotified) {
                    this._mjpegErrorNotified = true;
                    this.showAlert('MJPEG 流出现错误，尝试重连…', 'warning');
                }
                // Implement exponential backoff for reconnection
                // 实现指数退避重连机制
                const retryDelay = Math.min(5000, (this.retryCount || 0) * 500 + 500);
                this.retryCount = (this.retryCount || 0) + 1;
                setTimeout(() => {
                    if (this._stopping || !this.isActive) return;
                    const q2 = `t=${Date.now()}&r=${Math.random()}`;
                    this.mjpegImg.src = `/video_feed_mjpeg?${q2}`;
                }, retryDelay);
            };
            
            // Reset retry counter on successful start
            // 成功启动时重置重试计数器
            this.retryCount = 0;
            
            // Cache-busting query with connection optimization hints
            // 带连接优化提示的缓存清除查询
            const q = `t=${Date.now()}&r=${Math.random()}&buffer=low`;
            this.mjpegImg.src = `/video_feed_mjpeg?${q}`;
        } catch (err) {
            console.error('Failed to start MJPEG stream:', err);
            this.showAlert('无法开始 MJPEG 流', 'danger');
        }
    }

    /**
     * Optimized draw loop for MJPEG with frame skipping and adaptive rendering
     * 优化的MJPEG绘制循环，支持跳帧和自适应渲染
     */
    drawLoop() {
        if (!this.isActive) return;
        if (!this.videoContext || !this.videoCanvas || !this.mjpegImg) return;

        try {
            const img = this.mjpegImg;
            if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                // Resize canvases to match stream on first frames
                if (this.videoCanvas.width !== img.naturalWidth || this.videoCanvas.height !== img.naturalHeight) {
                    this.setupCanvases(img.naturalWidth, img.naturalHeight);
                }

                // Use efficient canvas drawing with image smoothing control
                // 使用高效的画布绘制并控制图像平滑
                this.videoContext.imageSmoothingEnabled = true;
                this.videoContext.imageSmoothingQuality = 'low'; // Faster rendering
                this.videoContext.clearRect(0, 0, this.videoCanvas.width, this.videoCanvas.height);
                this.videoContext.drawImage(img, 0, 0, this.videoCanvas.width, this.videoCanvas.height);
                this.frameCount++;

                // Trigger detection with optimized throttling
                // 使用优化节流触发检测
                if (this.isActive && window.detectionManager && window.detectionManager.isModelLoaded) {
                    this.performDetection();
                }
            }
        } catch (err) {
            console.warn('MJPEG draw error:', err);
        }

        // Use optimized frame scheduling for smoother playback
        // 使用优化的帧调度以实现更流畅的播放
        this.rafId = requestAnimationFrame(() => this.drawLoop());
    }

    /**
     * Stop MJPEG streaming and rAF
     * 停止 MJPEG 推流与绘制循环
     */
    stopMjpegStream() {
        if (this.rafId) {
            cancelAnimationFrame(this.rafId);
            this.rafId = null;
        }
        if (this.mjpegImg) {
            // Reset src to terminate connection
            this.mjpegImg.src = '';
        }
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
     * Start object detection processing (now handled in performDetection during frame fetch)
     * 开始目标检测处理（现在在帧获取时通过performDetection处理）
     */
    startDetection() {
        // Reset error counter for detection
        // 重置检测错误计数器
        this.detectionErrorCount = 0;
        console.log('Detection enabled - will run during frame processing');
    }

    /**
     * Perform detection on current frame
     */
    async performDetection() {
        if (!this.frameImage || !window.detectionManager || !window.detectionManager.isModelLoaded) {
            return;
        }
        
        // Enhanced detection throttling with frame skipping for better performance
        // 增强的检测节流机制，支持跳帧以提升性能
        const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
        const shouldDetect = !this.lastDetectionTime || (now - this.lastDetectionTime >= this.detectEveryMs);

        if (!shouldDetect) {
            // Use lightweight prediction-only updates for non-detection frames
            // 对非检测帧使用轻量级的仅预测更新
            try {
                if (this.tracker && this.tracker.getTracks().length > 0) {
                    this.tracker.predictOnly();
                    // Only redraw if there are locked tracks to avoid unnecessary rendering
                    // 仅在有锁定目标时重绘，避免不必要的渲染
                    if (this.tracker.getTracks().some(t => t.locked)) {
                        this.clearOverlay();
                        this.drawTracks();
                    }
                }
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

            // 仅绘制追踪框（不显示检测框），只在有锁定目标时重绘
            // Only redraw tracking overlay when there are locked targets
            if (this.tracker.getTracks().some(t => t.locked)) {
                this.clearOverlay();
                this.drawTracks();
            } else {
                // Clear overlay if no locked targets
                this.clearOverlay();
            }

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
     * 绘制追踪框与轨迹（显示类别+ID）
     */
    drawTracks() {
        if (!this.detectionContext || !this.tracker) return;
        const ctx = this.detectionContext;

        // Only draw locked tracks so boxes appear only after user clicks
        // 仅绘制已锁定的追踪目标，点击后才显示框
        const tracks = this.tracker.getTracks().filter(t => t.locked);
        for (const t of tracks) {
            const b = t.bbox;
            ctx.save();
            ctx.strokeStyle = t.color;
            ctx.lineWidth = t.locked ? 3 : 2;
            ctx.setLineDash(t.locked ? [6, 4] : []);
            ctx.strokeRect(b.x, b.y, b.w, b.h);

            // Draw class + ID label (no "Lock" text)
            // 绘制类别+ID标签（不显示"Lock"文字）
            const className = this.formatClassName(t.class || 'object');
            const label = `${className} ID ${t.id}`;
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
            const className = this.formatClassName(hit.class || 'object');
            if (hit.locked) {
                this.tracker.unlock(hit.id);
                this.showAlert(`已解锁 ${className} ID ${hit.id}`, 'info');
            } else {
                hit.locked = true;
                hit.lostFrames = 0;
                this.showAlert(`已锁定 ${className} ID ${hit.id}`, 'success');
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
            // Get the track to show its class in the alert
            // 获取追踪目标以在提示中显示其类别
            const track = this.tracker.getTracks().find(t => t.id === id);
            const className = this.formatClassName(track?.class || 'object');
            this.showAlert(`已锁定 ${className} ID ${id}`, 'success');
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
     * Format class name for display with Chinese translation support
     * 格式化类别名称用于显示，支持中文翻译
     */
    formatClassName(className) {
        // Common object class translations
        // 常见目标类别翻译
        const translations = {
            'person': '人物',
            'car': '汽车',
            'truck': '卡车',
            'bus': '公交车',
            'motorcycle': '摩托车',
            'bicycle': '自行车',
            'dog': '狗',
            'cat': '猫',
            'bird': '鸟',
            'bottle': '瓶子',
            'cup': '杯子',
            'chair': '椅子',
            'laptop': '笔记本电脑',
            'cell phone': '手机',
            'book': '书籍'
        };
        
        const formatted = className.split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
        
        // Return Chinese translation if available, otherwise return formatted English
        // 如果有中文翻译则返回，否则返回格式化的英文
        return translations[className.toLowerCase()] || formatted;
    }
}

/**
 * Update tracking list UI (locked tracks) with individual control buttons
 * 更新追踪列表界面（已锁定目标）并添加单独控制按钮
 */
StreamManager.prototype.updateTrackingList = function () {
    const container = document.getElementById('trackingList');
    if (!container || !this.tracker) return;
    
    // Update tracking list with current locked tracks
    // 更新追踪列表显示当前锁定目标
    const trackingList = document.getElementById('trackingList');
    if (!trackingList) return;
    
    const tracks = this.tracker ? this.tracker.getTracks().filter(t => t.locked) : [];
    
    if (tracks.length === 0) {
        trackingList.innerHTML = `
            <div class="no-detections text-center py-4">
                <i class="fas fa-bullseye fa-2x text-muted mb-2"></i>
                <p class="text-muted mb-0">未选择目标</p>
                <small class="text-muted">点击画面中的目标以开始追踪</small>
            </div>
        `;
        return;
    }
    
    // Generate HTML for each locked track with enhanced status display
    // 为每个锁定目标生成HTML，包含增强的状态显示
    const tracksHTML = tracks.map(track => {
        const className = this.formatClassName(track.class || 'object');
        const status = track.lostFrames > 0 ? '丢失' : '追踪中';
        const statusClass = track.lostFrames > 0 ? 'text-warning' : 'text-success';
        const confidenceLevel = track.hits > 10 ? '高' : track.hits > 5 ? '中' : '低';
        
        return `
            <div class="tracking-item" style="border-left: 4px solid ${track.color};">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <div class="fw-bold">${className} ID ${track.id}</div>
                        <small class="${statusClass}">${status} | 精度: ${confidenceLevel}</small>
                    </div>
                    <div class="tracking-actions">
                        <button class="btn btn-sm btn-outline-secondary" 
                                onclick="window.streamManager.tracker.unlock(${track.id}); window.streamManager.updateTrackingList(); window.streamManager.clearOverlay(); window.streamManager.drawTracks();" 
                                title="解除追踪目标">
                            <i class="fas fa-unlock"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    trackingList.innerHTML = tracksHTML;
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
        const className = this.formatClassName(track.class || 'object');
        this.tracker.unlock(targetId);
        this.showAlert(`已解锁 ${className} ID ${targetId}`, 'info');
        
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
