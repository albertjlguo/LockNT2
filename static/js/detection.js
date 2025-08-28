/**
 * Enhanced Object Detection Manager with PP-YOLOE
 * 使用PP-YOLOE的增强目标检测管理器
 * Handles PaddlePaddle.js PP-YOLOE model loading and object detection
 * Better performance for small objects and dense crowds
 * 处理PaddlePaddle.js PP-YOLOE模型加载和目标检测，对小目标和密集人群有更好的性能
 */
class ObjectDetectionManager {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.detectionCallbacks = [];
        this.detectionStats = {
            totalDetections: 0,
            objectCounts: {},
            recentDetections: []
        };
        
        // Enhanced confidence thresholds for PP-YOLOE
        // PP-YOLOE的增强置信度阈值
        this.confidenceThresholds = {
            person: 0.3,  // Lower threshold for better small object detection
            car: 0.4      // Optimized for PP-YOLOE performance
        };
        
        // PP-YOLOE specific configuration
        // PP-YOLOE特定配置
        this.modelConfig = {
            modelPath: 'https://paddlejs.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco/',
            inputSize: [640, 640], // Standard input size for PP-YOLOE
            maxDetections: 100,    // Maximum detections per frame
            nmsThreshold: 0.45,    // Non-maximum suppression threshold
            useGPU: true          // Enable GPU acceleration if available
        };
        
        this.initializeConfidenceControls();
    }

    /**
     * Initialize confidence control sliders
     * 初始化置信度控制滑块
     */
    initializeConfidenceControls() {
        // Person confidence slider
        // 人物置信度滑块
        const personSlider = document.getElementById('personConfidence');
        const personValue = document.getElementById('personConfidenceValue');
        
        if (personSlider && personValue) {
            personSlider.addEventListener('input', (e) => {
                const value = parseInt(e.target.value);
                this.confidenceThresholds.person = value / 100;
                personValue.textContent = value + '%';
                console.log(`Person confidence threshold updated to: ${value}%`);
            });
        }
        
        // Car confidence slider
        // 汽车置信度滑块
        const carSlider = document.getElementById('carConfidence');
        const carValue = document.getElementById('carConfidenceValue');
        
        if (carSlider && carValue) {
            carSlider.addEventListener('input', (e) => {
                const value = parseInt(e.target.value);
                this.confidenceThresholds.car = value / 100;
                carValue.textContent = value + '%';
                console.log(`Car confidence threshold updated to: ${value}%`);
            });
        }
    }

    /**
     * Initialize and load the PP-YOLOE model
     * 初始化并加载PP-YOLOE模型
     */
    async loadModel() {
        try {
            this.updateModelStatus('loading', 'Loading PP-YOLOE Model...');
            console.log('Starting PP-YOLOE model initialization...');
            
            // Initialize PaddlePaddle.js environment
            // 初始化PaddlePaddle.js环境
            if (typeof paddle === 'undefined') {
                throw new Error('PaddlePaddle.js not loaded. Please check script imports.');
            }
            
            // Configure PP-YOLOE model options
            // 配置PP-YOLOE模型选项
            const modelOptions = {
                modelUrl: this.modelConfig.modelPath,
                inputSize: this.modelConfig.inputSize,
                useGPU: this.modelConfig.useGPU,
                backend: 'webgl', // Use WebGL backend for better performance
                precision: 'fp16'  // Use half precision for faster inference
            };
            
            // Load PP-YOLOE model
            // 加载PP-YOLOE模型
            this.model = await paddle.vision.detection.ppyoloe(modelOptions);
            
            if (!this.model) {
                throw new Error('Failed to initialize PP-YOLOE model');
            }
            
            this.isModelLoaded = true;
            this.updateModelStatus('ready', 'PP-YOLOE Model Ready');
            console.log('PP-YOLOE model loaded successfully with enhanced small object detection');
            
            return true;
        } catch (error) {
            console.error('Error loading PP-YOLOE model:', error);
            console.warn('Falling back to alternative detection method...');
            
            // Fallback to a simpler detection method if PP-YOLOE fails
            // 如果PP-YOLOE失败，回退到更简单的检测方法
            try {
                await this.loadFallbackModel();
                return true;
            } catch (fallbackError) {
                console.error('Fallback model also failed:', fallbackError);
                this.updateModelStatus('error', 'Failed to load Detection Model');
                return false;
            }
        }
    }
    
    /**
     * Fallback model loading if PP-YOLOE fails
     * PP-YOLOE失败时的回退模型加载
     */
    async loadFallbackModel() {
        console.log('Loading fallback detection model...');
        this.updateModelStatus('loading', 'Loading Fallback Model...');
        
        // Use a simple TensorFlow.js based detection as fallback
        // 使用简单的TensorFlow.js检测作为回退
        if (typeof tf !== 'undefined' && typeof cocoSsd !== 'undefined') {
            this.model = await cocoSsd.load();
            this.isModelLoaded = true;
            this.updateModelStatus('ready', 'Fallback Model Ready');
            console.log('Fallback COCO-SSD model loaded');
        } else {
            throw new Error('No fallback model available');
        }
    }

    /**
     * Update model status in the UI
     */
    updateModelStatus(status, message) {
        const statusIndicator = document.getElementById('modelStatus');
        const statusText = document.getElementById('modelStatusText');
        const progressBar = document.querySelector('#modelProgress .progress-bar');

        if (statusIndicator && statusText) {
            statusText.textContent = message;
            
            // Update status indicator
            statusIndicator.innerHTML = '';
            switch (status) {
                case 'loading':
                    statusIndicator.innerHTML = '<i class="fas fa-circle text-warning pulse"></i>';
                    if (progressBar) {
                        progressBar.style.width = '50%';
                    }
                    break;
                case 'ready':
                    statusIndicator.innerHTML = '<i class="fas fa-circle text-success"></i>';
                    if (progressBar) {
                        progressBar.style.width = '100%';
                        setTimeout(() => {
                            document.getElementById('modelProgress').style.display = 'none';
                        }, 1000);
                    }
                    break;
                case 'error':
                    statusIndicator.innerHTML = '<i class="fas fa-circle text-danger"></i>';
                    if (progressBar) {
                        progressBar.style.width = '0%';
                    }
                    break;
            }
        }
    }

    /**
     * Enhanced object detection with PP-YOLOE for better small object and crowd detection
     * 使用PP-YOLOE增强目标检测，更好地检测小目标和人群
     */
    async detectObjects(videoElement) {
        if (!this.isModelLoaded || !this.model) {
            console.warn('PP-YOLOE model not loaded yet');
            return [];
        }

        try {
            let rawPredictions;
            
            // Check if using PP-YOLOE or fallback model
            // 检查是否使用PP-YOLOE或回退模型
            if (this.model.predict && typeof this.model.predict === 'function') {
                // PP-YOLOE detection
                // PP-YOLOE检测
                rawPredictions = await this.detectWithPPYOLOE(videoElement);
            } else if (this.model.detect && typeof this.model.detect === 'function') {
                // Fallback COCO-SSD detection
                // 回退COCO-SSD检测
                rawPredictions = await this.model.detect(videoElement);
                console.log('Using fallback COCO-SSD detection');
            } else {
                throw new Error('Invalid model interface');
            }
            
            // Log detection info for debugging
            // 记录检测信息用于调试
            if (rawPredictions && rawPredictions.length > 0) {
                const imageSize = `${videoElement.naturalWidth || videoElement.width}x${videoElement.naturalHeight || videoElement.height}`;
                console.log(`PP-YOLOE detected ${rawPredictions.length} objects on ${imageSize} image`);
            }
            
            // Apply enhanced bounding box refinement for PP-YOLOE
            // 为PP-YOLOE应用增强的边界框优化
            const refinedPredictions = this.refineBoundingBoxes(rawPredictions || [], videoElement);
            
            this.processDetections(refinedPredictions);
            return refinedPredictions;
        } catch (error) {
            console.error('Error during PP-YOLOE object detection:', error);
            return [];
        }
    }
    
    /**
     * PP-YOLOE specific detection method
     * PP-YOLOE特定检测方法
     */
    async detectWithPPYOLOE(videoElement) {
        try {
            // Prepare input tensor for PP-YOLOE
            // 为PP-YOLOE准备输入张量
            const inputTensor = this.preprocessImageForPPYOLOE(videoElement);
            
            // Run PP-YOLOE inference
            // 运行PP-YOLOE推理
            const predictions = await this.model.predict(inputTensor, {
                maxDetections: this.modelConfig.maxDetections,
                scoreThreshold: Math.min(...Object.values(this.confidenceThresholds)),
                nmsThreshold: this.modelConfig.nmsThreshold
            });
            
            // Convert PP-YOLOE output to standard format
            // 将PP-YOLOE输出转换为标准格式
            return this.convertPPYOLOEOutput(predictions, videoElement);
        } catch (error) {
            console.error('PP-YOLOE detection error:', error);
            throw error;
        }
    }
    
    /**
     * Preprocess image for PP-YOLOE input requirements
     * 为PP-YOLOE输入要求预处理图像
     */
    preprocessImageForPPYOLOE(videoElement) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Set canvas size to model input size
        // 设置画布大小为模型输入大小
        canvas.width = this.modelConfig.inputSize[0];
        canvas.height = this.modelConfig.inputSize[1];
        
        // Draw and resize image
        // 绘制并调整图像大小
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        // Get image data and normalize
        // 获取图像数据并归一化
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        // Convert to tensor format expected by PP-YOLOE
        // 转换为PP-YOLOE期望的张量格式
        const tensor = new Float32Array(canvas.width * canvas.height * 3);
        for (let i = 0; i < data.length; i += 4) {
            const pixelIndex = i / 4;
            tensor[pixelIndex] = data[i] / 255.0;     // R
            tensor[pixelIndex + canvas.width * canvas.height] = data[i + 1] / 255.0; // G
            tensor[pixelIndex + 2 * canvas.width * canvas.height] = data[i + 2] / 255.0; // B
        }
        
        return tensor;
    }
    
    /**
     * Convert PP-YOLOE output to standard detection format
     * 将PP-YOLOE输出转换为标准检测格式
     */
    convertPPYOLOEOutput(predictions, videoElement) {
        if (!predictions || !predictions.boxes) {
            return [];
        }
        
        const detections = [];
        const scaleX = (videoElement.naturalWidth || videoElement.width) / this.modelConfig.inputSize[0];
        const scaleY = (videoElement.naturalHeight || videoElement.height) / this.modelConfig.inputSize[1];
        
        for (let i = 0; i < predictions.boxes.length; i++) {
            const box = predictions.boxes[i];
            const score = predictions.scores[i];
            const classId = predictions.classes[i];
            
            // Map class ID to class name (COCO dataset classes)
            // 将类别ID映射到类别名称（COCO数据集类别）
            const className = this.getClassNameFromId(classId);
            
            if (className && (className === 'person' || className === 'car')) {
                detections.push({
                    class: className,
                    score: score,
                    bbox: [
                        box[0] * scaleX, // x
                        box[1] * scaleY, // y
                        (box[2] - box[0]) * scaleX, // width
                        (box[3] - box[1]) * scaleY  // height
                    ]
                });
            }
        }
        
        return detections;
    }
    
    /**
     * Get class name from COCO class ID
     * 从COCO类别ID获取类别名称
     */
    getClassNameFromId(classId) {
        const cocoClasses = {
            0: 'person',
            2: 'car',
            // Add more classes as needed
            // 根据需要添加更多类别
        };
        return cocoClasses[classId] || null;
    }
    
    /**
     * Refine bounding boxes with improved accuracy and coordinate validation
     * 优化边界框，提高准确性和坐标验证
     */
    refineBoundingBoxes(predictions, videoElement) {
        if (!predictions || predictions.length === 0) return [];
        
        // Get image dimensions for coordinate validation
        // 获取图像尺寸用于坐标验证
        const imageWidth = videoElement.naturalWidth || videoElement.width || 640;
        const imageHeight = videoElement.naturalHeight || videoElement.height || 480;
        
        return predictions.map(pred => {
            // Filter to only detect persons and cars (PP-YOLOE supports more classes)
            // 只检测人和汽车（PP-YOLOE支持更多类别）
            const allowedClasses = ['person', 'car'];
            if (!allowedClasses.includes(pred.class)) {
                return null;
            }
            
            // Apply class-specific confidence filtering
            // 应用针对类别的置信度过滤
            const minConfidence = this.confidenceThresholds[pred.class] || 0.3;
            if (pred.score < minConfidence) return null;
            
            // Validate original bbox coordinates
            // 验证原始边界框坐标
            const [x, y, w, h] = pred.bbox;
            if (x < 0 || y < 0 || w <= 0 || h <= 0 || 
                x + w > imageWidth || y + h > imageHeight) {
                console.warn('Invalid bbox coordinates:', pred.bbox, 'Image size:', imageWidth, 'x', imageHeight);
                // Clamp to valid range
                const clampedBbox = [
                    Math.max(0, Math.min(x, imageWidth - 1)),
                    Math.max(0, Math.min(y, imageHeight - 1)),
                    Math.max(1, Math.min(w, imageWidth - Math.max(0, x))),
                    Math.max(1, Math.min(h, imageHeight - Math.max(0, y)))
                ];
                pred.bbox = clampedBbox;
            }
            
            // Apply conservative bounding box optimization
            // 应用保守的边界框优化
            const refinedBbox = this.optimizeBoundingBox(pred.bbox, pred.class, pred.score);
            
            // Final coordinate validation after optimization
            // 优化后的最终坐标验证
            const [fx, fy, fw, fh] = refinedBbox;
            const finalBbox = [
                Math.max(0, Math.min(fx, imageWidth - 1)),
                Math.max(0, Math.min(fy, imageHeight - 1)),
                Math.max(1, Math.min(fw, imageWidth - Math.max(0, fx))),
                Math.max(1, Math.min(fh, imageHeight - Math.max(0, fy)))
            ];
            
            return {
                ...pred,
                bbox: finalBbox,
                // Store original image dimensions for scaling reference
                // 存储原始图像尺寸作为缩放参考
                originalImageSize: { width: imageWidth, height: imageHeight },
                // Add stability score for tracking
                // 添加用于追踪的稳定性分数
                stability: this.calculateStabilityScore(pred.score, pred.class)
            };
        }).filter(pred => pred !== null);
    }
    
    /**
     * Enhanced bounding box optimization for PP-YOLOE with small object focus
     * PP-YOLOE增强边界框优化，专注于小目标
     */
    optimizeBoundingBox(bbox, objectClass, confidence) {
        const [x, y, width, height] = bbox;
        
        // Validate input bbox
        // 验证输入边界框
        if (width <= 0 || height <= 0) {
            console.warn('Invalid bbox dimensions:', bbox);
            return bbox; // Return original if invalid
        }
        
        // PP-YOLOE optimized adjustments for better small object detection
        // PP-YOLOE优化调整，更好地检测小目标
        let adjustmentFactor = 1.0;
        let paddingRatio = 0.01; // Minimal padding for PP-YOLOE precision
        
        // Determine if this is a small object (area < 32x32 pixels)
        // 判断是否为小目标（面积 < 32x32像素）
        const isSmallObject = (width * height) < (32 * 32);
        
        switch (objectClass) {
            case 'person':
                // Enhanced handling for person detection, especially small persons
                // 增强人物检测处理，特别是小人物
                if (isSmallObject) {
                    adjustmentFactor = 1.02; // Slight expansion for small persons
                    paddingRatio = 0.02;
                } else {
                    adjustmentFactor = confidence > 0.7 ? 0.99 : 1.0;
                    paddingRatio = 0.008;
                }
                break;
            case 'car':
                // Optimized car detection with PP-YOLOE
                // PP-YOLOE优化的汽车检测
                if (isSmallObject) {
                    adjustmentFactor = 1.01; // Minimal expansion for small cars
                    paddingRatio = 0.015;
                } else {
                    adjustmentFactor = confidence > 0.8 ? 0.98 : 1.0;
                    paddingRatio = 0.01;
                }
                break;
            default:
                // Conservative approach for other classes
                // 其他类别的保守方法
                adjustmentFactor = 1.0;
                paddingRatio = 0.005;
        }
        
        // Apply PP-YOLOE specific adjustments
        // 应用PP-YOLOE特定调整
        const centerX = x + width / 2;
        const centerY = y + height / 2;
        const newWidth = width * adjustmentFactor;
        const newHeight = height * adjustmentFactor;
        
        // Add adaptive padding based on confidence and object size
        // 基于置信度和目标大小添加自适应填充
        const confidenceBonus = Math.max(0, (confidence - 0.5) * 0.01);
        const finalPaddingRatio = paddingRatio + confidenceBonus;
        
        const paddedWidth = newWidth * (1 + finalPaddingRatio);
        const paddedHeight = newHeight * (1 + finalPaddingRatio);
        
        // Ensure the adjusted bbox doesn't go negative
        // 确保调整后的边界框不为负数
        const finalX = Math.max(0, centerX - paddedWidth / 2);
        const finalY = Math.max(0, centerY - paddedHeight / 2);
        
        return [
            finalX,
            finalY,
            paddedWidth,
            paddedHeight
        ];
    }
    
    /**
     * Enhanced stability score calculation for PP-YOLOE tracking reliability
     * PP-YOLOE追踪可靠性的增强稳定性分数计算
     */
    calculateStabilityScore(confidence, objectClass) {
        // Base stability from confidence with PP-YOLOE adjustments
        // 基于置信度的基础稳定性，包含PP-YOLOE调整
        let stability = confidence;
        
        // PP-YOLOE specific class multipliers (better performance than COCO-SSD)
        // PP-YOLOE特定类别乘数（比COCO-SSD性能更好）
        const classMultipliers = {
            'person': 0.92,     // PP-YOLOE has better person detection
            'car': 0.96         // Excellent car detection with PP-YOLOE
        };
        
        const multiplier = classMultipliers[objectClass] || 0.88;
        
        // Add bonus for high confidence detections from PP-YOLOE
        // 为PP-YOLOE的高置信度检测添加奖励
        const confidenceBonus = confidence > 0.8 ? 0.05 : 0;
        
        return Math.min(1.0, (stability * multiplier) + confidenceBonus);
    }

    /**
     * Process detection results and update statistics
     */
    processDetections(predictions) {
        // Update total detection count
        this.detectionStats.totalDetections += predictions.length;
        
        // Reset current frame object counts
        const currentFrameCounts = {};
        
        // Process each prediction
        predictions.forEach(prediction => {
            const className = prediction.class;
            const confidence = (prediction.score * 100).toFixed(1);
            
            // Update object counts
            currentFrameCounts[className] = (currentFrameCounts[className] || 0) + 1;
            
            // Add to recent detections
            this.detectionStats.recentDetections.unshift({
                class: className,
                confidence: confidence,
                timestamp: new Date()
            });
            
            // Keep only last 50 detections
            if (this.detectionStats.recentDetections.length > 50) {
                this.detectionStats.recentDetections.pop();
            }
        });
        
        // Update object counts (using current frame counts)
        this.detectionStats.objectCounts = currentFrameCounts;
        
        // Update UI
        this.updateDetectionUI();
        
        // Notify callbacks
        this.detectionCallbacks.forEach(callback => {
            callback(predictions, this.detectionStats);
        });
    }

    /**
     * Update the detection results UI
     */
    updateDetectionUI() {
        // Update object counts
        this.updateObjectCounts();
        
        // Update detection history
        this.updateDetectionHistory();
        
        // Update total count
        const detectionCountElement = document.getElementById('detectionCount');
        if (detectionCountElement) {
            const totalCurrentObjects = Object.values(this.detectionStats.objectCounts)
                .reduce((sum, count) => sum + count, 0);
            detectionCountElement.textContent = totalCurrentObjects;
        }
    }

    /**
     * Update object counts display
     */
    updateObjectCounts() {
        const objectCountsContainer = document.getElementById('objectCounts');
        if (!objectCountsContainer) return;

        const counts = this.detectionStats.objectCounts;
        
        if (Object.keys(counts).length === 0) {
            objectCountsContainer.innerHTML = `
                <div class="no-detections text-center py-4">
                    <i class="fas fa-search fa-2x text-muted mb-2"></i>
                    <p class="text-muted mb-0">No objects detected</p>
                    <small class="text-muted">Objects will appear here when detected</small>
                </div>
            `;
            return;
        }

        // Sort objects by count (descending)
        const sortedObjects = Object.entries(counts)
            .sort(([,a], [,b]) => b - a);

        const objectsHTML = sortedObjects.map(([className, count]) => {
            // Get average confidence for this object class
            const recentForClass = this.detectionStats.recentDetections
                .filter(d => d.class === className)
                .slice(0, 5); // Last 5 detections
            
            const avgConfidence = recentForClass.length > 0 
                ? (recentForClass.reduce((sum, d) => sum + parseFloat(d.confidence), 0) / recentForClass.length).toFixed(1)
                : '0.0';

            return `
                <div class="object-item fade-in">
                    <div>
                        <div class="object-name">${this.formatClassName(className)}</div>
                        <div class="confidence-score">${avgConfidence}% confidence</div>
                    </div>
                    <div class="object-count">${count}</div>
                </div>
            `;
        }).join('');

        objectCountsContainer.innerHTML = objectsHTML;
    }

    /**
     * Update detection history display
     */
    updateDetectionHistory() {
        const historyContainer = document.getElementById('detectionHistory');
        if (!historyContainer) return;

        const recentDetections = this.detectionStats.recentDetections.slice(0, 10);
        
        if (recentDetections.length === 0) {
            historyContainer.innerHTML = `
                <div class="no-history text-center py-3">
                    <i class="fas fa-history fa-lg text-muted mb-2"></i>
                    <p class="text-muted mb-0 small">Detection history will appear here</p>
                </div>
            `;
            return;
        }

        const historyHTML = recentDetections.map(detection => {
            const timeStr = detection.timestamp.toLocaleTimeString();
            return `
                <div class="detection-entry fade-in">
                    <div class="d-flex justify-content-between align-items-center">
                        <span>${this.formatClassName(detection.class)}</span>
                        <span class="confidence-score">${detection.confidence}%</span>
                    </div>
                    <div class="detection-time">${timeStr}</div>
                </div>
            `;
        }).join('');

        historyContainer.innerHTML = historyHTML;
    }

    /**
     * Format class name for display
     */
    formatClassName(className) {
        return className.split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    /**
     * Add a callback for detection events
     */
    addDetectionCallback(callback) {
        this.detectionCallbacks.push(callback);
    }
    
    /**
     * Get detection quality metrics for tracking optimization
     * 获取用于追踪优化的检测质量指标
     */
    getDetectionQuality() {
        const recentDetections = this.detectionStats.recentDetections.slice(0, 10);
        if (recentDetections.length === 0) return { quality: 0, stability: 0 };
        
        const avgConfidence = recentDetections.reduce((sum, det) => 
            sum + parseFloat(det.confidence), 0) / recentDetections.length;
        
        const stability = recentDetections.length >= 5 ? 0.8 : 0.5;
        
        return {
            quality: avgConfidence / 100,
            stability: stability,
            frameCount: recentDetections.length
        };
    }

    /**
     * Clear detection statistics
     */
    clearStats() {
        this.detectionStats = {
            totalDetections: 0,
            objectCounts: {},
            recentDetections: []
        };
        this.updateDetectionUI();
    }

    /**
     * Get current detection statistics
     */
    getStats() {
        return { ...this.detectionStats };
    }
    
    /**
     * Apply temporal smoothing to reduce detection jitter
     * 应用时间平滑以减少检测抖动
     */
    applyTemporalSmoothing(currentDetections, previousDetections) {
        if (!previousDetections || previousDetections.length === 0) {
            return currentDetections;
        }
        
        return currentDetections.map(current => {
            // Find closest previous detection
            // 找到最接近的先前检测
            const closest = this.findClosestDetection(current, previousDetections);
            
            if (closest && this.calculateIoU(current.bbox, closest.bbox) > 0.3) {
                // Apply smoothing to reduce jitter
                // 应用平滑以减少抖动
                const smoothingFactor = 0.3;
                const [cx, cy, cw, ch] = current.bbox;
                const [px, py, pw, ph] = closest.bbox;
                
                return {
                    ...current,
                    bbox: [
                        cx * (1 - smoothingFactor) + px * smoothingFactor,
                        cy * (1 - smoothingFactor) + py * smoothingFactor,
                        cw * (1 - smoothingFactor) + pw * smoothingFactor,
                        ch * (1 - smoothingFactor) + ph * smoothingFactor
                    ]
                };
            }
            
            return current;
        });
    }
    
    /**
     * Find closest detection by center distance
     * 通过中心距离找到最接近的检测
     */
    findClosestDetection(target, detections) {
        const [tx, ty, tw, th] = target.bbox;
        const tcx = tx + tw / 2;
        const tcy = ty + th / 2;
        
        let closest = null;
        let minDistance = Infinity;
        
        for (const det of detections) {
            if (det.class !== target.class) continue;
            
            const [dx, dy, dw, dh] = det.bbox;
            const dcx = dx + dw / 2;
            const dcy = dy + dh / 2;
            
            const distance = Math.sqrt((tcx - dcx) ** 2 + (tcy - dcy) ** 2);
            
            if (distance < minDistance) {
                minDistance = distance;
                closest = det;
            }
        }
        
        return minDistance < 100 ? closest : null; // Max 100px distance
    }
    
    /**
     * Calculate IoU between two bounding boxes
     * 计算两个边界框之间的IoU
     */
    calculateIoU(bbox1, bbox2) {
        const [x1, y1, w1, h1] = bbox1;
        const [x2, y2, w2, h2] = bbox2;
        
        const x1_max = x1 + w1;
        const y1_max = y1 + h1;
        const x2_max = x2 + w2;
        const y2_max = y2 + h2;
        
        const intersect_x1 = Math.max(x1, x2);
        const intersect_y1 = Math.max(y1, y2);
        const intersect_x2 = Math.min(x1_max, x2_max);
        const intersect_y2 = Math.min(y1_max, y2_max);
        
        if (intersect_x2 <= intersect_x1 || intersect_y2 <= intersect_y1) {
            return 0;
        }
        
        const intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1);
        const bbox1_area = w1 * h1;
        const bbox2_area = w2 * h2;
        const union_area = bbox1_area + bbox2_area - intersect_area;
        
        return intersect_area / union_area;
    }
}

// Global PP-YOLOE detection manager instance with enhanced capabilities
// 全局PP-YOLOE检测管理器实例，具有增强功能
window.detectionManager = new ObjectDetectionManager();
window.detectionManager.previousDetections = []; // Store for temporal smoothing
window.detectionManager.modelType = 'PP-YOLOE'; // Track which model is being used

// Initialize PP-YOLOE detection manager when page loads
// 页面加载时初始化PP-YOLOE检测管理器
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Loading PP-YOLOE model for enhanced small object and crowd detection...');
    console.log('正在加载PP-YOLOE模型以增强小目标和人群检测...');
    
    const loadSuccess = await window.detectionManager.loadModel();
    
    if (loadSuccess) {
        console.log('PP-YOLOE model ready for enhanced detection capabilities');
        console.log('PP-YOLOE模型已就绪，具备增强检测能力');
    } else {
        console.error('Failed to load detection model. Please refresh the page.');
        console.error('检测模型加载失败，请刷新页面');
    }
});
