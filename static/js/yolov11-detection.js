/**
 * YOLOv11 Object Detection Manager
 * YOLOv11 目标检测管理器
 * Handles YOLOv11 model loading and inference for superior small object and dense crowd detection
 * 处理YOLOv11模型加载和推理，提供卓越的小目标和密集人群检测
 */

class YOLOv11DetectionManager {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.detectionCallbacks = [];
        
        // Detection statistics
        // 检测统计
        this.detectionStats = {
            totalDetections: 0,
            objectCounts: {},
            recentDetections: []
        };
        
        // YOLOv11 optimized confidence thresholds for small targets
        // YOLOv11 针对小目标优化的置信度阈值
        this.confidenceThresholds = {
            person: 0.3,  // Lower threshold for better small target detection / 更低阈值以更好检测小目标
            car: 0.35,    // Slightly higher for vehicles / 车辆稍高
            bicycle: 0.3,
            motorcycle: 0.3,
            bus: 0.4,
            truck: 0.4
        };
        
        // Server-side YOLOv11 configuration
        // 服务器端YOLOv11配置
        this.modelConfig = {
            serverEndpoint: '/yolo_detect',
            statusEndpoint: '/yolo_status',
            maxDetections: 100,
            scoreThreshold: 0.25
        };
        
        // Class names mapping (COCO 80 classes)
        // 类别名称映射（COCO 80个类别）
        this.classNames = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ];
        
        // Classes to detect (focused on persons and vehicles)
        // 要检测的类别（专注于人和车辆）
        this.targetClasses = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck'];
        
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
            personSlider.value = this.confidenceThresholds.person * 100;
            personValue.textContent = Math.round(this.confidenceThresholds.person * 100) + '%';
            
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
            carSlider.value = this.confidenceThresholds.car * 100;
            carValue.textContent = Math.round(this.confidenceThresholds.car * 100) + '%';
            
            carSlider.addEventListener('input', (e) => {
                const value = parseInt(e.target.value);
                this.confidenceThresholds.car = value / 100;
                carValue.textContent = value + '%';
                console.log(`Car confidence threshold updated to: ${value}%`);
            });
        }
    }

    /**
     * Load YOLOv11 model (server-side approach)
     * 加载YOLOv11模型（服务器端方式）
     */
    async loadModel() {
        try {
            this.updateModelStatus('loading', 'Initializing YOLOv11 Server...');
            
            // Check server-side YOLOv11 status
            // 检查服务器端YOLOv11状态
            const response = await fetch('/yolo_status');
            const status = await response.json();
            
            if (status.model_loaded) {
                this.isModelLoaded = true;
                this.updateModelStatus('ready', 'YOLOv11 Server Ready');
                console.log('Server-side YOLOv11 model ready');
                return true;
            } else {
                throw new Error('Server-side YOLOv11 not available');
            }
            
        } catch (error) {
            console.error('Error connecting to YOLOv11 server:', error);
            this.updateModelStatus('error', 'YOLOv11 Server Unavailable');
            
            // Fallback to COCO-SSD if server-side YOLOv11 fails
            // 如果服务器端YOLOv11失败，回退到COCO-SSD
            console.log('Attempting fallback to COCO-SSD...');
            return this.loadFallbackModel();
        }
    }

    // Warmup not needed for server-side detection
    // 服务器端检测不需要预热

    /**
     * Fallback to COCO-SSD model if YOLOv11 fails
     * 如果YOLOv11失败，回退到COCO-SSD模型
     */
    async loadFallbackModel() {
        try {
            console.log('Loading fallback COCO-SSD model...');
            this.model = await cocoSsd.load();
            this.isModelLoaded = true;
            this.isFallbackMode = true;
            this.updateModelStatus('ready', 'COCO-SSD Model Ready (Fallback)');
            console.log('COCO-SSD fallback model loaded successfully');
            return true;
        } catch (error) {
            console.error('Failed to load fallback model:', error);
            this.updateModelStatus('error', 'Failed to load any model');
            return false;
        }
    }

    /**
     * Update model status in the UI
     * 更新UI中的模型状态
     */
    updateModelStatus(status, message) {
        const statusIndicator = document.getElementById('modelStatus');
        const statusText = document.getElementById('modelStatusText');
        const progressBar = document.querySelector('#modelProgress .progress-bar');

        if (statusIndicator && statusText) {
            statusText.textContent = message;
            
            // Update status indicator
            // 更新状态指示器
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
                            const progressContainer = document.getElementById('modelProgress');
                            if (progressContainer) {
                                progressContainer.style.display = 'none';
                            }
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
     * Detect objects using server-side YOLOv11 API
     * 使用服务器端YOLOv11 API检测目标
     */
    async detectObjects(videoElement) {
        try {
            // Use fallback detection if in fallback mode
            // 如果在回退模式下使用回退检测
            if (this.isFallbackMode) {
                return this.detectObjectsFallback(videoElement);
            }

            // Convert video element to base64 for API
            // 将视频元素转换为base64供API使用
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = videoElement.width || videoElement.videoWidth || 640;
            canvas.height = videoElement.height || videoElement.videoHeight || 480;
            
            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Get current confidence thresholds
            // 获取当前置信度阈值
            const personConfidence = this.confidenceThresholds.person;
            
            // Send request to server-side YOLOv11 API
            // 发送请求到服务器端YOLOv11 API
            const response = await fetch('/yolo_detect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: imageData,
                    confidence: personConfidence
                })
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (!result.success) {
                throw new Error(result.error || 'Detection failed');
            }
            
            // Process and return detections
            // 处理并返回检测结果
            const detections = result.detections || [];
            this.processDetections(detections);
            return detections;
            
        } catch (error) {
            console.error('Error during server-side YOLOv11 detection:', error);
            // Fallback to COCO-SSD on error
            // 出错时回退到COCO-SSD
            return this.detectObjectsFallback(videoElement);
        }
    }

    /**
     * Fallback detection using COCO-SSD
     * 使用COCO-SSD的回退检测
     */
    async detectObjectsFallback(videoElement) {
        try {
            const rawPredictions = await this.model.detect(videoElement);
            const refinedPredictions = this.refineBoundingBoxes(rawPredictions, videoElement);
            this.processDetections(refinedPredictions);
            return refinedPredictions;
        } catch (error) {
            console.error('Error during fallback detection:', error);
            return [];
        }
    }

    /**
     * Preprocess image for model input with proper data type handling
     * 为模型输入预处理图像，正确处理数据类型
     */
    async preprocessImage(videoElement) {
        return tf.tidy(() => {
            // Convert video frame to tensor
            // 将视频帧转换为张量
            const imageTensor = tf.browser.fromPixels(videoElement);
            
            // Resize to model input size
            // 调整到模型输入尺寸
            const resized = tf.image.resizeBilinear(
                imageTensor, 
                [this.modelConfig.inputSize, this.modelConfig.inputSize]
            );
            
            // Handle different model input requirements
            // 处理不同模型的输入要求
            let processedTensor;
            
            // Check if model expects int32 (like some TensorFlow Hub models)
            // 检查模型是否期望int32（如某些TensorFlow Hub模型）
            if (this.modelExpectsInt32) {
                // Keep pixel values as integers [0, 255] and cast to int32
                // 保持像素值为整数[0, 255]并转换为int32
                processedTensor = resized.cast('int32');
            } else {
                // Normalize pixel values to [0, 1] for float32 models
                // 为float32模型将像素值归一化到[0, 1]
                processedTensor = resized.div(255.0);
            }
            
            // Add batch dimension
            // 添加批次维度
            return processedTensor.expandDims(0);
        });
    }

    /**
     * Post-process YOLOv11 predictions
     * 后处理YOLOv11预测
     */
    async postprocessPredictions(predictions, videoElement) {
        return tf.tidy(() => {
            // YOLOv11 output format: [batch, num_boxes, 4 + num_classes]
            // YOLOv11输出格式：[批次, 框数量, 4 + 类别数]
            let boxes, scores, classes;
            
            if (Array.isArray(predictions)) {
                // Handle multiple output format
                // 处理多输出格式
                [boxes, scores, classes] = predictions;
            } else {
                // Handle single output format
                // 处理单输出格式
                const output = predictions;
                const [numBoxes, numFeatures] = output.shape.slice(1);
                
                // Extract boxes, scores, and classes
                // 提取框、分数和类别
                boxes = output.slice([0, 0, 0], [1, numBoxes, 4]);
                const classProbs = output.slice([0, 0, 4], [1, numBoxes, numFeatures - 4]);
                scores = classProbs.max(2);
                classes = classProbs.argMax(2);
            }
            
            // Apply NMS (Non-Maximum Suppression)
            // 应用非极大值抑制
            const nmsResults = this.applyNMS(
                boxes.squeeze(0),
                scores.squeeze(0),
                classes.squeeze(0)
            );
            
            // Convert to detection format
            // 转换为检测格式
            return this.formatDetections(
                nmsResults,
                videoElement.width,
                videoElement.height
            );
        });
    }

    /**
     * Apply Non-Maximum Suppression
     * 应用非极大值抑制
     */
    applyNMS(boxes, scores, classes) {
        const detections = [];
        
        // Convert tensors to arrays
        // 将张量转换为数组
        const boxesArray = boxes.arraySync();
        const scoresArray = scores.arraySync();
        const classesArray = classes.arraySync();
        
        // Group detections by class
        // 按类别分组检测
        const detectionsByClass = {};
        
        for (let i = 0; i < scoresArray.length; i++) {
            const score = scoresArray[i];
            const classId = classesArray[i];
            const className = this.classNames[classId];
            
            // Skip if not target class or below threshold
            // 如果不是目标类别或低于阈值则跳过
            if (!this.targetClasses.includes(className)) continue;
            
            const threshold = this.confidenceThresholds[className] || this.modelConfig.scoreThreshold;
            if (score < threshold) continue;
            
            if (!detectionsByClass[classId]) {
                detectionsByClass[classId] = [];
            }
            
            detectionsByClass[classId].push({
                box: boxesArray[i],
                score: score,
                class: className
            });
        }
        
        // Apply NMS per class
        // 对每个类别应用NMS
        for (const classId in detectionsByClass) {
            const classDetections = detectionsByClass[classId];
            const nmsDetections = this.nmsPerClass(classDetections);
            detections.push(...nmsDetections);
        }
        
        return detections;
    }

    /**
     * NMS for a single class
     * 单个类别的NMS
     */
    nmsPerClass(detections) {
        // Sort by score descending
        // 按分数降序排序
        detections.sort((a, b) => b.score - a.score);
        
        const keep = [];
        const used = new Set();
        
        for (let i = 0; i < detections.length; i++) {
            if (used.has(i)) continue;
            
            keep.push(detections[i]);
            used.add(i);
            
            for (let j = i + 1; j < detections.length; j++) {
                if (used.has(j)) continue;
                
                const iou = this.calculateIoU(
                    detections[i].box,
                    detections[j].box
                );
                
                if (iou > this.modelConfig.iouThreshold) {
                    used.add(j);
                }
            }
        }
        
        return keep;
    }

    /**
     * Format detections to standard format
     * 将检测格式化为标准格式
     */
    formatDetections(nmsResults, imageWidth, imageHeight) {
        return nmsResults.map(detection => {
            const [x, y, w, h] = detection.box;
            
            // Convert from normalized to pixel coordinates
            // 从归一化坐标转换为像素坐标
            const bbox = [
                x * imageWidth,
                y * imageHeight,
                w * imageWidth,
                h * imageHeight
            ];
            
            return {
                bbox: bbox,
                class: detection.class,
                score: detection.score,
                originalImageSize: { width: imageWidth, height: imageHeight }
            };
        });
    }

    /**
     * Refine bounding boxes (compatibility method)
     * 优化边界框（兼容方法）
     */
    refineBoundingBoxes(predictions, videoElement) {
        if (!predictions || predictions.length === 0) return [];
        
        const imageWidth = videoElement.width || 640;
        const imageHeight = videoElement.height || 480;
        
        return predictions.map(pred => {
            // Filter to target classes
            // 过滤到目标类别
            if (!this.targetClasses.includes(pred.class)) {
                return null;
            }
            
            // Apply confidence threshold
            // 应用置信度阈值
            const minConfidence = this.confidenceThresholds[pred.class] || 0.25;
            if (pred.score < minConfidence) return null;
            
            // Validate and clamp bbox
            // 验证并限制边界框
            const [x, y, w, h] = pred.bbox;
            const clampedBbox = [
                Math.max(0, Math.min(x, imageWidth - 1)),
                Math.max(0, Math.min(y, imageHeight - 1)),
                Math.max(1, Math.min(w, imageWidth - Math.max(0, x))),
                Math.max(1, Math.min(h, imageHeight - Math.max(0, y)))
            ];
            
            return {
                ...pred,
                bbox: clampedBbox,
                originalImageSize: { width: imageWidth, height: imageHeight }
            };
        }).filter(pred => pred !== null);
    }

    /**
     * Process detection results and update statistics
     * 处理检测结果并更新统计
     */
    processDetections(predictions) {
        // Update total detection count
        // 更新总检测计数
        this.detectionStats.totalDetections += predictions.length;
        
        // Reset current frame object counts
        // 重置当前帧目标计数
        const currentFrameCounts = {};
        
        // Process each prediction
        // 处理每个预测
        predictions.forEach(prediction => {
            const className = prediction.class;
            const confidence = (prediction.score * 100).toFixed(1);
            
            // Update object counts
            // 更新目标计数
            currentFrameCounts[className] = (currentFrameCounts[className] || 0) + 1;
            
            // Add to recent detections
            // 添加到最近检测
            this.detectionStats.recentDetections.unshift({
                class: className,
                confidence: confidence,
                timestamp: new Date()
            });
            
            // Keep only last 50 detections
            // 只保留最后50个检测
            if (this.detectionStats.recentDetections.length > 50) {
                this.detectionStats.recentDetections.pop();
            }
        });
        
        // Update object counts
        // 更新目标计数
        this.detectionStats.objectCounts = currentFrameCounts;
        
        // Update UI
        // 更新UI
        this.updateDetectionUI();
        
        // Notify callbacks
        // 通知回调
        this.detectionCallbacks.forEach(callback => {
            callback(predictions, this.detectionStats);
        });
    }

    /**
     * Update the detection results UI
     * 更新检测结果UI
     */
    updateDetectionUI() {
        // Update object counts
        // 更新目标计数
        this.updateObjectCounts();
        
        // Update detection history
        // 更新检测历史
        this.updateDetectionHistory();
        
        // Update total count
        // 更新总计数
        const detectionCountElement = document.getElementById('detectionCount');
        if (detectionCountElement) {
            const totalCurrentObjects = Object.values(this.detectionStats.objectCounts)
                .reduce((sum, count) => sum + count, 0);
            detectionCountElement.textContent = totalCurrentObjects;
        }
    }

    /**
     * Update object counts display
     * 更新目标计数显示
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
        // 按计数排序（降序）
        const sortedObjects = Object.entries(counts)
            .sort(([,a], [,b]) => b - a);

        const objectsHTML = sortedObjects.map(([className, count]) => {
            // Get average confidence for this object class
            // 获取此目标类别的平均置信度
            const recentForClass = this.detectionStats.recentDetections
                .filter(d => d.class === className)
                .slice(0, 5);
            
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
     * 更新检测历史显示
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
     * 格式化类别名称以供显示
     */
    formatClassName(className) {
        return className.split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
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

    /**
     * Add a callback for detection events
     * 添加检测事件的回调
     */
    addDetectionCallback(callback) {
        this.detectionCallbacks.push(callback);
    }

    /**
     * Get detection quality metrics
     * 获取检测质量指标
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
     * 清除检测统计
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
     * 获取当前检测统计
     */
    getStats() {
        return { ...this.detectionStats };
    }
}

// Create global YOLOv11 detection manager instance
// 创建全局YOLOv11检测管理器实例
window.yolov11DetectionManager = new YOLOv11DetectionManager();

// Initialize detection manager when page loads
// 页面加载时初始化检测管理器
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Initializing YOLOv11 object detection...');
    await window.yolov11DetectionManager.loadModel();
});
