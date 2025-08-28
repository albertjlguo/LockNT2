/**
 * Object Detection Manager
 * Handles TensorFlow.js model loading and object detection
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
        
        // Configurable confidence thresholds (aligned with backend tracker values)
        // 可配置的置信度阈值（与后端追踪器数值对齐）
        this.confidenceThresholds = {
            person: 0.5,   // 50% - Aligned with tracker detectHigh
            car: 0.5       // 50% - Aligned with tracker detectHigh
        };
        
        // Previous detections for temporal smoothing
        // 用于时间平滑的先前检测
        this.previousDetections = [];
        
        // Multi-frame detection buffer for fusion
        // 多帧检测缓冲区用于融合
        this.detectionBuffer = [];
        this.maxBufferSize = 3; // Keep last 3 frames for fusion
        
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
     * Initialize and load the COCO-SSD model
     */
    async loadModel() {
        try {
            this.updateModelStatus('loading', 'Loading AI Model...');
            
            // Load the COCO-SSD model
            this.model = await cocoSsd.load();
            this.isModelLoaded = true;
            
            this.updateModelStatus('ready', 'AI Model Ready');
            console.log('COCO-SSD model loaded successfully');
            
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            this.updateModelStatus('error', 'Failed to load AI Model');
            return false;
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
     * Enhanced object detection with improved coordinate handling and temporal smoothing
     * 增强的目标检测，改进坐标处理和时间平滑
     */
    async detectObjects(videoElement) {
        if (!this.isModelLoaded || !this.model) {
            console.warn('Model not loaded yet');
            return [];
        }

        try {
            // Get raw predictions from COCO-SSD model
            // 从 COCO-SSD 模型获取原始预测结果
            const rawPredictions = await this.model.detect(videoElement);
            
            // Apply bounding box refinement and filtering
            // 应用边界框优化和过滤
            const refinedPredictions = this.refineBoundingBoxes(rawPredictions, videoElement);
            
            // Apply Non-Maximum Suppression to remove overlapping detections
            // 应用非极大值抑制移除重叠检测
            const nmsFiltered = this.applyNonMaxSuppression(refinedPredictions);
            
            // Apply multi-frame detection fusion for better accuracy
            // 应用多帧检测融合以提高准确性
            const fusedDetections = this.applyMultiFrameFusion(nmsFiltered);
            
            // Apply temporal smoothing to reduce jitter
            // 应用时间平滑以减少抖动
            const smoothedPredictions = this.applyTemporalSmoothing(fusedDetections, this.previousDetections);
            
            // Store for next frame smoothing
            // 存储用于下一帧平滑
            this.previousDetections = smoothedPredictions;
            
            this.processDetections(smoothedPredictions);
            return smoothedPredictions;
        } catch (error) {
            console.error('Error during object detection:', error);
            return [];
        }
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
            // Filter to only detect persons and cars
            // 只检测人和汽车
            const allowedClasses = ['person', 'car'];
            if (!allowedClasses.includes(pred.class)) {
                return null;
            }
            
            // Apply adaptive confidence filtering based on object size and context
            // 应用基于目标大小和上下文的自适应置信度过滤
            const baseConfidence = this.confidenceThresholds[pred.class] || 0.3;
            const objectSize = pred.bbox[2] * pred.bbox[3];
            const imageArea = imageWidth * imageHeight;
            const sizeRatio = objectSize / imageArea;
            
            // Adaptive threshold based on object size relative to image
            // 基于目标相对于图像大小的自适应阈值
            let sizeAdjustment = 0;
            if (sizeRatio < 0.01) { // Very small objects
                sizeAdjustment = -0.1;
            } else if (sizeRatio < 0.05) { // Small objects
                sizeAdjustment = -0.05;
            } else if (sizeRatio > 0.3) { // Large objects
                sizeAdjustment = 0.05;
            }
            
            const minConfidence = Math.max(0.15, baseConfidence + sizeAdjustment);
            
            if (pred.score < minConfidence) return null;
            
            // Enhanced coordinate validation and intelligent clamping
            // 增强的坐标验证和智能钳制
            const [x, y, w, h] = pred.bbox;
            
            // Check for invalid coordinates
            // 检查无效坐标
            let needsClamping = false;
            if (x < 0 || y < 0 || w <= 0 || h <= 0 || 
                x + w > imageWidth || y + h > imageHeight) {
                needsClamping = true;
            }
            
            if (needsClamping) {
                // Intelligent clamping that preserves aspect ratio when possible
                // 智能钳制，尽可能保持宽高比
                const originalAspectRatio = w / h;
                
                // Calculate maximum possible dimensions within image bounds
                // 计算图像边界内的最大可能尺寸
                const maxX = Math.max(0, Math.min(x, imageWidth - 1));
                const maxY = Math.max(0, Math.min(y, imageHeight - 1));
                const maxW = imageWidth - maxX;
                const maxH = imageHeight - maxY;
                
                let clampedW = Math.max(1, Math.min(w, maxW));
                let clampedH = Math.max(1, Math.min(h, maxH));
                
                // Try to preserve aspect ratio
                // 尝试保持宽高比
                if (clampedW / clampedH > originalAspectRatio * 1.2) {
                    // Width is too large relative to height
                    // 宽度相对于高度过大
                    clampedW = clampedH * originalAspectRatio;
                } else if (clampedH / clampedW > (1/originalAspectRatio) * 1.2) {
                    // Height is too large relative to width
                    // 高度相对于宽度过大
                    clampedH = clampedW / originalAspectRatio;
                }
                
                // Final bounds check
                // 最终边界检查
                clampedW = Math.max(1, Math.min(clampedW, maxW));
                clampedH = Math.max(1, Math.min(clampedH, maxH));
                
                pred.bbox = [maxX, maxY, clampedW, clampedH];
                
                // Add quality penalty for heavily clamped detections
                // 为严重钳制的检测添加质量惩罚
                const clampingPenalty = Math.max(
                    Math.abs(x - maxX) / w,
                    Math.abs(y - maxY) / h,
                    Math.abs(w - clampedW) / w,
                    Math.abs(h - clampedH) / h
                );
                
                if (clampingPenalty > 0.3) {
                    pred.score *= (1 - clampingPenalty * 0.3); // Reduce confidence for heavily modified boxes
                }
            }
            
            // Apply minimal bounding box optimization to preserve accuracy
            // 应用最小边界框优化以保持准确性
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
     * Optimize bounding box based on object class and confidence with improved accuracy
     * 根据目标类别和置信度优化边界框，提高准确性
     */
    optimizeBoundingBox(bbox, objectClass, confidence) {
        const [x, y, width, height] = bbox;
        
        // Validate input bbox
        // 验证输入边界框
        if (width <= 0 || height <= 0) {
            console.warn('Invalid bbox dimensions:', bbox);
            return bbox; // Return original if invalid
        }
        
        // Enhanced optimization based on confidence and object characteristics
        // 基于置信度和目标特征的增强优化
        let adjustmentFactor = 1.0;
        let paddingRatio = 0.0;
        
        // Confidence-based adjustments
        // 基于置信度的调整
        if (confidence < 0.6) {
            // Lower confidence detections get slight tightening
            // 低置信度检测稍微收紧
            adjustmentFactor = 0.95;
        } else if (confidence > 0.8) {
            // High confidence detections get slight expansion for completeness
            // 高置信度检测稍微扩展以确保完整性
            adjustmentFactor = 1.02;
            paddingRatio = 0.01;
        }
        
        // Class-specific optimizations
        // 类别特定优化
        switch (objectClass) {
            case 'person':
                // Persons often need slight vertical expansion for full body
                // 人物通常需要轻微的垂直扩展以包含全身
                if (height / width > 1.5) { // Likely full body
                    adjustmentFactor *= 1.01;
                    paddingRatio += 0.005;
                }
                break;
            case 'car':
                // Cars benefit from slight horizontal expansion
                // 汽车受益于轻微的水平扩展
                if (width / height > 1.8) { // Likely side view
                    adjustmentFactor *= 1.01;
                    paddingRatio += 0.005;
                }
                break;
        }
        
        // Apply size-based adjustments
        // 应用基于大小的调整
        const objectArea = width * height;
        if (objectArea < 2000) { // Small objects
            adjustmentFactor *= 1.05; // Slight expansion for small objects
            paddingRatio += 0.02;
        } else if (objectArea > 50000) { // Large objects
            adjustmentFactor *= 0.98; // Slight tightening for large objects
        }
        
        // Calculate optimized dimensions
        // 计算优化后的尺寸
        const centerX = x + width / 2;
        const centerY = y + height / 2;
        const newWidth = width * adjustmentFactor;
        const newHeight = height * adjustmentFactor;
        
        // Add padding for stability
        // 添加填充以提高稳定性
        const paddedWidth = newWidth * (1 + paddingRatio);
        const paddedHeight = newHeight * (1 + paddingRatio);
        
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
     * Calculate stability score for tracking reliability
     * 计算用于追踪可靠性的稳定性分数
     */
    calculateStabilityScore(confidence, objectClass) {
        // Base stability from confidence
        // 基于置信度的基础稳定性
        let stability = confidence;
        
        // Class-specific stability adjustments (only for person and car)
        // 针对人和汽车的稳定性调整
        const classMultipliers = {
            'person': 0.9,      // Persons are generally well detected
            'car': 0.95         // Cars are very stable
        };
        
        const multiplier = classMultipliers[objectClass] || 0.85;
        return Math.min(1.0, stability * multiplier);
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
    
    /**
     * Apply Non-Maximum Suppression to remove overlapping detections
     * 应用非极大值抑制移除重叠检测
     */
    applyNonMaxSuppression(detections, iouThreshold = 0.5) {
        if (!detections || detections.length === 0) return [];
        
        // Group detections by class
        // 按类别分组检测
        const detectionsByClass = {};
        detections.forEach(det => {
            if (!detectionsByClass[det.class]) {
                detectionsByClass[det.class] = [];
            }
            detectionsByClass[det.class].push(det);
        });
        
        const finalDetections = [];
        
        // Apply NMS for each class separately
        // 对每个类别分别应用NMS
        Object.keys(detectionsByClass).forEach(className => {
            const classDetections = detectionsByClass[className];
            
            // Sort by confidence score (descending)
            // 按置信度分数排序（降序）
            classDetections.sort((a, b) => b.score - a.score);
            
            const keep = [];
            const suppressed = new Set();
            
            for (let i = 0; i < classDetections.length; i++) {
                if (suppressed.has(i)) continue;
                
                const currentDet = classDetections[i];
                keep.push(currentDet);
                
                // Suppress overlapping detections
                // 抑制重叠检测
                for (let j = i + 1; j < classDetections.length; j++) {
                    if (suppressed.has(j)) continue;
                    
                    const otherDet = classDetections[j];
                    const iou = this.calculateIoU(currentDet.bbox, otherDet.bbox);
                    
                    if (iou > iouThreshold) {
                        suppressed.add(j);
                    }
                }
            }
            
            finalDetections.push(...keep);
        });
        
        return finalDetections;
    }
    
    /**
     * Apply multi-frame detection fusion for better accuracy
     * 应用多帧检测融合以提高准确性
     */
    applyMultiFrameFusion(currentDetections) {
        // Add current detections to buffer
        // 将当前检测添加到缓冲区
        this.detectionBuffer.push({
            detections: currentDetections,
            timestamp: Date.now()
        });
        
        // Keep only recent frames
        // 只保留最近的帧
        if (this.detectionBuffer.length > this.maxBufferSize) {
            this.detectionBuffer.shift();
        }
        
        // If we don't have enough frames yet, return current detections
        // 如果还没有足够的帧，返回当前检测
        if (this.detectionBuffer.length < 2) {
            return currentDetections;
        }
        
        const fusedDetections = [];
        
        // For each current detection, try to find matches in previous frames
        // 对于每个当前检测，尝试在先前帧中找到匹配
        currentDetections.forEach(currentDet => {
            const matches = [];
            matches.push(currentDet);
            
            // Look for matches in previous frames
            // 在先前帧中寻找匹配
            for (let i = this.detectionBuffer.length - 2; i >= 0; i--) {
                const frameData = this.detectionBuffer[i];
                const bestMatch = this.findBestMatch(currentDet, frameData.detections);
                
                if (bestMatch) {
                    matches.push(bestMatch);
                }
            }
            
            // Fuse the matches to create a more accurate detection
            // 融合匹配以创建更准确的检测
            if (matches.length >= 2) {
                const fusedDetection = this.fuseDetections(matches);
                fusedDetections.push(fusedDetection);
            } else {
                // No matches found, use current detection
                // 未找到匹配，使用当前检测
                fusedDetections.push(currentDet);
            }
        });
        
        return fusedDetections;
    }
    
    /**
     * Find best matching detection in a frame
     * 在帧中找到最佳匹配检测
     */
    findBestMatch(targetDet, frameDetections) {
        let bestMatch = null;
        let bestScore = 0;
        
        frameDetections.forEach(det => {
            if (det.class !== targetDet.class) return;
            
            const iou = this.calculateIoU(targetDet.bbox, det.bbox);
            const centerDist = this.calculateCenterDistance(targetDet.bbox, det.bbox);
            const maxDim = Math.max(targetDet.bbox[2], targetDet.bbox[3]);
            const normalizedCenterDist = centerDist / maxDim;
            
            // Combined score: IoU + center distance + confidence
            // 组合分数：IoU + 中心距离 + 置信度
            const score = iou * 0.6 + (1 - Math.min(1, normalizedCenterDist)) * 0.3 + det.score * 0.1;
            
            if (score > bestScore && iou > 0.3) {
                bestScore = score;
                bestMatch = det;
            }
        });
        
        return bestMatch;
    }
    
    /**
     * Calculate center distance between two bounding boxes
     * 计算两个边界框之间的中心距离
     */
    calculateCenterDistance(bbox1, bbox2) {
        const [x1, y1, w1, h1] = bbox1;
        const [x2, y2, w2, h2] = bbox2;
        
        const cx1 = x1 + w1 / 2;
        const cy1 = y1 + h1 / 2;
        const cx2 = x2 + w2 / 2;
        const cy2 = y2 + h2 / 2;
        
        return Math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2);
    }
    
    /**
     * Fuse multiple detections into a single, more accurate detection
     * 将多个检测融合为单个更准确的检测
     */
    fuseDetections(detections) {
        if (detections.length === 0) return null;
        if (detections.length === 1) return detections[0];
        
        // Weight more recent detections higher
        // 给更近期的检测更高权重
        const weights = detections.map((_, idx) => Math.pow(0.8, detections.length - 1 - idx));
        const totalWeight = weights.reduce((sum, w) => sum + w, 0);
        
        // Weighted average of bounding boxes
        // 边界框的加权平均
        let fusedX = 0, fusedY = 0, fusedW = 0, fusedH = 0;
        let fusedScore = 0;
        
        detections.forEach((det, idx) => {
            const weight = weights[idx] / totalWeight;
            const [x, y, w, h] = det.bbox;
            
            fusedX += x * weight;
            fusedY += y * weight;
            fusedW += w * weight;
            fusedH += h * weight;
            fusedScore += det.score * weight;
        });
        
        // Use the most recent detection as base and update bbox
        // 使用最近的检测作为基础并更新边界框
        const baseDet = detections[0];
        
        return {
            ...baseDet,
            bbox: [fusedX, fusedY, fusedW, fusedH],
            score: fusedScore,
            // Add fusion metadata
            // 添加融合元数据
            fusionInfo: {
                frameCount: detections.length,
                confidence: fusedScore,
                stability: detections.length / this.maxBufferSize
            }
        };
    }
}

// Global detection manager instance with previous frame storage
window.detectionManager = new ObjectDetectionManager();
window.detectionManager.previousDetections = []; // Store for temporal smoothing

// Initialize detection manager when page loads
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Loading enhanced object detection model...');
    await window.detectionManager.loadModel();
});
