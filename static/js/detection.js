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
        
        // 优化的置信度阈值 - 降低以捕获更多有效检测
        // Optimized confidence thresholds - lowered to capture more valid detections
        this.confidenceThresholds = {
            person: 0.35,  // 降低到35%以捕获更多人物检测
            car: 0.40      // 降低到40%以捕获更多汽车检测
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
     * Initialize and load the COCO-SSD model
     */
    async loadModel() {
        try {
            this.updateModelStatus('loading', 'Loading AI Model...');
            
            // 加载优化的COCO-SSD模型配置
            // Load optimized COCO-SSD model configuration
            this.model = await cocoSsd.load({
              base: 'mobilenet_v2', // 使用更准确的mobilenet_v2而非lite版本
              modelUrl: undefined   // 使用默认URL以获得最佳性能
            });
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
     * Enhanced object detection with improved coordinate handling
     * 增强的目标检测，改进坐标处理
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
            
            // Log detection info for debugging
            // 记录检测信息用于调试
            if (rawPredictions.length > 0) {
                console.log(`Detected ${rawPredictions.length} objects on ${videoElement.naturalWidth || videoElement.width}x${videoElement.naturalHeight || videoElement.height} image`);
            }
            
            // Apply bounding box refinement and filtering
            // 应用边界框优化和过滤
            const refinedPredictions = this.refineBoundingBoxes(rawPredictions, videoElement);
            
            this.processDetections(refinedPredictions);
            return refinedPredictions;
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
            
            // 动态置信度调整 - 根据目标大小和位置调整阈值
            // Dynamic confidence adjustment - adjust threshold based on target size and position
            let minConfidence = this.confidenceThresholds[pred.class] || 0.3;
            
            // 根据目标大小调整置信度阈值
            // Adjust confidence threshold based on target size
            const targetArea = pred.bbox[2] * pred.bbox[3];
            const imageArea = imageWidth * imageHeight;
            const sizeRatio = targetArea / imageArea;
            
            if (sizeRatio < 0.01) {
                // 小目标：降低阈值
                // Small targets: lower threshold
                minConfidence = Math.max(0.25, minConfidence - 0.1);
            } else if (sizeRatio > 0.25) {
                // 大目标：提高阈值
                // Large targets: raise threshold
                minConfidence = Math.min(0.6, minConfidence + 0.05);
            }
            
            // 根据目标位置调整（边缘目标通常置信度较低）
            // Adjust based on target position (edge targets usually have lower confidence)
            const centerX = pred.bbox[0] + pred.bbox[2] / 2;
            const centerY = pred.bbox[1] + pred.bbox[3] / 2;
            const distFromCenter = Math.sqrt(
                Math.pow((centerX - imageWidth/2) / (imageWidth/2), 2) + 
                Math.pow((centerY - imageHeight/2) / (imageHeight/2), 2)
            );
            
            if (distFromCenter > 0.7) {
                // 边缘目标：降低阈值
                // Edge targets: lower threshold
                minConfidence = Math.max(0.2, minConfidence - 0.08);
            }
            
            if (pred.score < minConfidence) return null;
            
            // 改进的坐标验证 - 支持部分可见目标
            // Improved coordinate validation - support partially visible targets
            const [x, y, w, h] = pred.bbox;
            
            // 检查边界框是否完全无效
            // Check if bbox is completely invalid
            if (w <= 0 || h <= 0) {
                console.warn('Invalid bbox dimensions:', pred.bbox);
                return null; // 完全无效的框直接丢弃
            }
            
            // 计算目标在图像内的可见比例
            // Calculate visible ratio of target within image
            const visibleX = Math.max(0, Math.min(x + w, imageWidth) - Math.max(0, x));
            const visibleY = Math.max(0, Math.min(y + h, imageHeight) - Math.max(0, y));
            const visibleArea = visibleX * visibleY;
            const totalArea = w * h;
            const visibleRatio = visibleArea / totalArea;
            
            // 如果可见比例太小，丢弃检测
            // Discard detection if visible ratio is too small
            if (visibleRatio < 0.3) {
                console.log(`Discarding detection with low visibility: ${(visibleRatio * 100).toFixed(1)}%`);
                return null;
            }
            
            // 对于部分可见的目标，保持原始坐标但添加标记
            // For partially visible targets, keep original coordinates but add flag
            if (x < 0 || y < 0 || x + w > imageWidth || y + h > imageHeight) {
                pred.partiallyVisible = true;
                pred.visibleRatio = visibleRatio;
                console.log(`Partially visible target detected: ${(visibleRatio * 100).toFixed(1)}% visible`);
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
        
        // 大幅减少边界框调整，保持原始检测结果的准确性
        // Significantly reduce bbox adjustments to maintain original detection accuracy
        let adjustmentFactor = 1.0; // 默认不调整
        let paddingRatio = 0.0; // 移除不必要的填充
        
        // 只对高置信度检测进行微调
        // Only fine-tune high-confidence detections
        if (confidence > 0.85) {
            switch (objectClass) {
                case 'person':
                    // 对人物检测几乎不调整
                    // Minimal adjustment for person detection
                    adjustmentFactor = 1.0; // 保持原始大小
                    paddingRatio = 0.0;
                    break;
                case 'car':
                    // 对汽车检测几乎不调整
                    // Minimal adjustment for car detection
                    adjustmentFactor = 1.0; // 保持原始大小
                    paddingRatio = 0.0;
                    break;
                default:
                    // 其他类别完全不调整
                    // No adjustment for other classes
                    adjustmentFactor = 1.0;
                    paddingRatio = 0.0;
            }
        }
        
        // 如果不需要调整，直接返回原始边界框
        // Return original bbox if no adjustment needed
        if (adjustmentFactor === 1.0 && paddingRatio === 0.0) {
            return bbox;
        }
        
        // Apply minimal adjustments only when necessary
        // 仅在必要时应用最小调整
        const centerX = x + width / 2;
        const centerY = y + height / 2;
        const newWidth = width * adjustmentFactor;
        const newHeight = height * adjustmentFactor;
        
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
}

// Global detection manager instance with previous frame storage
window.detectionManager = new ObjectDetectionManager();
window.detectionManager.previousDetections = []; // Store for temporal smoothing

// Initialize detection manager when page loads
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Loading enhanced object detection model...');
    await window.detectionManager.loadModel();
});
