/**
 * Object Detection Manager
 * Handles TensorFlow.js model loading and object detection
 */
class ObjectDetectionManager {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.detectionCallbacks = [];
        this.modelPath = './static/models/yolov8n.onnx'; // Path to the ONNX model
        this.yoloClasses = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ];
        this.detectionStats = {
            totalDetections: 0,
            objectCounts: {},
            recentDetections: []
        };
        
        // Enhanced confidence thresholds with dynamic adjustment
        this.confidenceThresholds = {
            person: 0.45,   // Lowered for better person detection
            car: 0.55,      // Slightly higher for more accurate car detection
            getThreshold: (className, objectSize, imageArea) => {
                const baseThreshold = this.confidenceThresholds[className] || 0.5;
                const sizeRatio = objectSize / imageArea;
                if (sizeRatio < 0.005) return Math.max(0.3, baseThreshold - 0.15);
                if (sizeRatio < 0.02) return Math.max(0.35, baseThreshold - 0.1);
                if (sizeRatio > 0.25) return Math.min(0.8, baseThreshold + 0.1);
                return baseThreshold;
            }
        };
        
        // Previous detections for temporal smoothing
        this.previousDetections = [];
        
        // Multi-frame detection buffer for fusion
        this.detectionBuffer = [];
        this.maxBufferSize = 3; // Keep last 3 frames for fusion
        
        // Occlusion and partial visibility handling
        this.occlusionHandler = {
            partialDetections: [],
            boundaryThreshold: 0.15,
            minVisibleRatio: 0.3,
            edgeExpansionFactor: 1.5,
            confidenceBoost: 0.1
        };
        
        this.isDetecting = false; // Add a flag to prevent concurrent detections
        this.initializeConfidenceControls();
    }

    /**
     * Initialize confidence control sliders
     * 初始化置信度控制滑块
     */
    initializeConfidenceControls() {
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
     * Initialize and load the ONNX YOLO model using ONNX Runtime Web
     */
    async loadModel() {
        let provider;
        try {
            this.updateModelStatus('loading', 'Loading AI Model (WebGPU)...');
            // First, try to create a session with the WebGPU provider.
            this.model = await ort.InferenceSession.create(this.modelPath, {
                executionProviders: ['webgpu'],
                graphOptimizationLevel: 'all',
            });
            provider = 'webgpu';
            console.log('ONNX model loaded successfully with WebGPU provider.');
        } catch (error) {
            console.warn('WebGPU is not available or failed to initialize. Falling back to WASM.', error);
            try {
                this.updateModelStatus('loading', 'Loading AI Model (WASM)...');
                // If WebGPU fails, fall back to the WASM provider.
                this.model = await ort.InferenceSession.create(this.modelPath, {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all',
                });
                provider = 'wasm';
                console.log('ONNX model loaded successfully with WASM provider.');
            } catch (wasmError) {
                console.error('Failed to load ONNX model with both WebGPU and WASM:', wasmError);
                this.updateModelStatus('error', 'Failed to load AI Model');
                return false;
            }
        }

        this.isModelLoaded = true;
        this.updateModelStatus('ready', `AI Model Ready (${provider})`);
        return true;
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
     * Preprocess the video frame to create a tensor for the YOLO model.
     * @param {HTMLVideoElement} videoElement The video element to process.
     * @returns {ort.Tensor} The preprocessed tensor.
     */
    preprocess(videoElement) {
        const modelWidth = 640;
        const modelHeight = 640;

        // Use an offscreen canvas for preprocessing
        const canvas = document.createElement('canvas');
        canvas.width = modelWidth;
        canvas.height = modelHeight;
        const ctx = canvas.getContext('2d');

        // Draw the video frame onto the canvas, resizing it
        ctx.drawImage(videoElement, 0, 0, modelWidth, modelHeight);
        const imageData = ctx.getImageData(0, 0, modelWidth, modelHeight);
        const { data } = imageData;

        // Convert image data to a Float32Array in NCHW format
        const red = [], green = [], blue = [];
        for (let i = 0; i < data.length; i += 4) {
            red.push(data[i] / 255.0);
            green.push(data[i + 1] / 255.0);
            blue.push(data[i + 2] / 255.0);
        }
        const transposedData = red.concat(green, blue);

        // Create the ONNX tensor
        const tensor = new ort.Tensor('float32', new Float32Array(transposedData), [1, 3, modelHeight, modelWidth]);
        return tensor;
    }

    /**
     * Postprocess the model's output tensor to get detections.
     * @param {ort.Tensor} outputTensor The output tensor from the model.
     * @param {number} imageWidth The original width of the video frame.
     * @param {number} imageHeight The original height of the video frame.
     * @returns {Array<object>} A list of detection objects.
     */
    postprocess(outputTensor, imageWidth, imageHeight) {
        const modelWidth = 640;
        const modelHeight = 640;
        const data = outputTensor.data;
        const predictions = [];

        // The output shape is [1, 84, 8400], where 84 = 4 (box) + 80 (classes)
        // We need to transpose it to [1, 8400, 84]
        const transposedData = [];
        for (let i = 0; i < 8400; i++) {
            for (let j = 0; j < 84; j++) {
                transposedData.push(data[j * 8400 + i]);
            }
        }

        for (let i = 0; i < 8400; i++) {
            const offset = i * 84;
            const box = transposedData.slice(offset, offset + 4);
            const classScores = transposedData.slice(offset + 4, offset + 84);

            let maxScore = 0;
            let maxIndex = -1;
            for (let j = 0; j < classScores.length; j++) {
                if (classScores[j] > maxScore) {
                    maxScore = classScores[j];
                    maxIndex = j;
                }
            }

            if (maxIndex !== -1) {
                const className = this.yoloClasses[maxIndex];
                const minConfidence = this.confidenceThresholds[className] || 0.5;

                if (maxScore > minConfidence) {
                const [cx, cy, w, h] = box;

                // Scale box coordinates from model space (640x640) to image space
                const scaleX = imageWidth / modelWidth;
                const scaleY = imageHeight / modelHeight;

                const x1 = (cx - w / 2) * scaleX;
                const y1 = (cy - h / 2) * scaleY;
                const width = w * scaleX;
                const height = h * scaleY;

                predictions.push({
                    bbox: [x1, y1, width, height],
                    class: className,
                    score: maxScore,
                });
                }
            }
        }
        return predictions;
    }

    /**
     * Object detection using the ONNX YOLO model.
     */
    async detectObjects(videoElement) {
        if (!this.isModelLoaded || !this.model || this.isDetecting) {
            return [];
        }

        this.isDetecting = true;

        try {
            // 1. Preprocess the frame
            const tensor = this.preprocess(videoElement);

            // 2. Run inference
            const feeds = { [this.model.inputNames[0]]: tensor };
            const results = await this.model.run(feeds);
            const outputTensor = results[this.model.outputNames[0]];

            // 3. Postprocess the results
            const rawPredictions = this.postprocess(outputTensor, videoElement.naturalWidth, videoElement.naturalHeight);

            // 4. Apply Non-Maximum Suppression
            const nmsFiltered = this.applyNonMaxSuppression(rawPredictions);

            // 5. Apply other refinements (smoothing, fusion, etc.)
            const fusedDetections = this.applyMultiFrameFusion(nmsFiltered);
            const occlusionEnhanced = this.handlePartialVisibility(fusedDetections, videoElement);
            const smoothedPredictions = this.applyTemporalSmoothing(occlusionEnhanced, this.previousDetections);

            this.previousDetections = smoothedPredictions;
            this.processDetections(smoothedPredictions);

            return smoothedPredictions;
        } catch (error) {
            console.error('Error during ONNX object detection:', error);
            return [];
        } finally {
            this.isDetecting = false; // Release the lock
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
            
            // Enhanced adaptive confidence filtering with improved accuracy
            // 增强的自适应置信度过滤，提高准确性
            const objectSize = pred.bbox[2] * pred.bbox[3];
            const imageArea = imageWidth * imageHeight;
            
            // Use dynamic threshold calculation
            // 使用动态阈值计算
            const minConfidence = this.confidenceThresholds.getThreshold(pred.class, objectSize, imageArea);
            
            // Additional context-based adjustments
            // 额外的基于上下文的调整
            let contextAdjustment = 0;
            const aspectRatio = pred.bbox[2] / pred.bbox[3];
            
            // Class-specific aspect ratio validation
            // 类别特定的宽高比验证
            if (pred.class === 'person') {
                // Persons should typically be taller than wide
                // 人物通常应该比宽度更高
                if (aspectRatio > 1.2) contextAdjustment -= 0.1; // Penalize wide person boxes
                if (aspectRatio < 0.3) contextAdjustment -= 0.05; // Penalize very narrow boxes
            } else if (pred.class === 'car') {
                // Cars should typically be wider than tall
                // 汽车通常应该比高度更宽
                if (aspectRatio < 0.8) contextAdjustment -= 0.1; // Penalize tall car boxes
                if (aspectRatio > 4.0) contextAdjustment -= 0.05; // Penalize very wide boxes
            }
            
            const finalMinConfidence = Math.max(0.2, minConfidence + contextAdjustment);
            
            if (pred.score < finalMinConfidence) return null;
            
            // Enhanced coordinate validation with scaling awareness
            // 增强的坐标验证，考虑缩放因子
            const [x, y, w, h] = pred.bbox;
            
            // Get display dimensions for proper scaling
            // 获取显示尺寸以进行正确缩放
            const displayWidth = videoElement.width || videoElement.clientWidth || 640;
            const displayHeight = videoElement.height || videoElement.clientHeight || 480;
            
            // Calculate scaling factors between natural and display size
            // 计算自然尺寸和显示尺寸之间的缩放因子
            const scaleX = imageWidth / displayWidth;
            const scaleY = imageHeight / displayHeight;
            
            // Check for invalid coordinates with scaling consideration
            // 检查无效坐标，考虑缩放
            let needsClamping = false;
            if (x < 0 || y < 0 || w <= 0 || h <= 0 || 
                x + w > imageWidth || y + h > imageHeight) {
                needsClamping = true;
            }
            
            if (needsClamping) {
                // Enhanced intelligent clamping with aspect ratio preservation
                // 增强的智能钳制，保持宽高比
                const originalAspectRatio = w / h;
                
                // Calculate maximum possible dimensions within image bounds
                // 计算图像边界内的最大可能尺寸
                const maxX = Math.max(0, Math.min(x, imageWidth - 1));
                const maxY = Math.max(0, Math.min(y, imageHeight - 1));
                const maxW = imageWidth - maxX;
                const maxH = imageHeight - maxY;
                
                let clampedW = Math.max(1, Math.min(w, maxW));
                let clampedH = Math.max(1, Math.min(h, maxH));
                
                // Enhanced aspect ratio preservation with class-specific constraints
                // 增强的宽高比保持，使用类别特定约束
                const expectedRatios = {
                    'person': { min: 0.3, max: 0.8, typical: 0.5 },
                    'car': { min: 1.5, max: 3.5, typical: 2.2 }
                };
                
                const classRatio = expectedRatios[pred.class];
                if (classRatio) {
                    const currentRatio = clampedW / clampedH;
                    
                    // Adjust to fit within expected ratio range
                    // 调整以适应预期的比例范围
                    if (currentRatio < classRatio.min) {
                        // Too narrow - expand width or reduce height
                        // 太窄 - 扩展宽度或减少高度
                        const targetW = clampedH * classRatio.min;
                        if (targetW <= maxW) {
                            clampedW = targetW;
                        } else {
                            clampedH = clampedW / classRatio.min;
                        }
                    } else if (currentRatio > classRatio.max) {
                        // Too wide - reduce width or expand height
                        // 太宽 - 减少宽度或扩展高度
                        const targetH = clampedW / classRatio.max;
                        if (targetH <= maxH) {
                            clampedH = targetH;
                        } else {
                            clampedW = clampedH * classRatio.max;
                        }
                    }
                }
                
                // Final bounds check with stricter validation
                // 最终边界检查，更严格的验证
                clampedW = Math.max(1, Math.min(clampedW, maxW));
                clampedH = Math.max(1, Math.min(clampedH, maxH));
                
                pred.bbox = [maxX, maxY, clampedW, clampedH];
                
                // Enhanced quality penalty calculation
                // 增强的质量惩罚计算
                const clampingPenalty = Math.max(
                    Math.abs(x - maxX) / Math.max(w, 1),
                    Math.abs(y - maxY) / Math.max(h, 1),
                    Math.abs(w - clampedW) / Math.max(w, 1),
                    Math.abs(h - clampedH) / Math.max(h, 1)
                );
                
                // More aggressive penalty for heavily modified boxes
                // 对严重修改的框施加更严厉的惩罚
                if (clampingPenalty > 0.2) {
                    pred.score *= (1 - clampingPenalty * 0.4);
                }
                
                // Mark as clamped for tracking system awareness
                // 标记为已钳制，供追踪系统感知
                pred.wasClamped = true;
                pred.clampingPenalty = clampingPenalty;
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
                // Store comprehensive metadata for tracking and scaling
                // 存储用于追踪和缩放的综合元数据
                originalImageSize: { width: imageWidth, height: imageHeight },
                displaySize: { width: displayWidth, height: displayHeight },
                scalingFactors: { x: scaleX, y: scaleY },
                // Enhanced stability score with accuracy factors
                // 增强的稳定性分数，包含准确性因子
                stability: this.calculateStabilityScore(pred.score, pred.class),
                // Add accuracy metrics
                // 添加准确性指标
                accuracyMetrics: {
                    aspectRatioValid: this.validateAspectRatio(finalBbox, pred.class),
                    sizeReasonable: this.validateObjectSize(finalBbox, imageWidth, imageHeight),
                    positionValid: this.validatePosition(finalBbox, imageWidth, imageHeight)
                }
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
        
        // Enhanced optimization with stricter accuracy controls
        // 增强优化，更严格的准确性控制
        let adjustmentFactor = 1.0;
        let paddingRatio = 0.0;
        
        // More conservative confidence-based adjustments
        // 更保守的基于置信度的调整
        if (confidence < 0.5) {
            // Lower confidence detections get moderate tightening
            // 低置信度检测适度收紧
            adjustmentFactor = 0.92;
        } else if (confidence > 0.85) {
            // Only very high confidence detections get expansion
            // 只有非常高置信度的检测才扩展
            adjustmentFactor = 1.01;
            paddingRatio = 0.005;
        }
        
        // More precise class-specific optimizations
        // 更精确的类别特定优化
        const aspectRatio = width / height;
        switch (objectClass) {
            case 'person':
                // Validate person aspect ratio and adjust accordingly
                // 验证人物宽高比并相应调整
                if (aspectRatio > 0.3 && aspectRatio < 0.8 && height / width > 1.2) {
                    // Valid person proportions - minimal adjustment
                    // 有效的人物比例 - 最小调整
                    adjustmentFactor *= 1.005;
                } else if (aspectRatio > 1.0) {
                    // Unusually wide person box - tighten
                    // 异常宽的人物框 - 收紧
                    adjustmentFactor *= 0.95;
                }
                break;
            case 'car':
                // Validate car aspect ratio
                // 验证汽车宽高比
                if (aspectRatio > 1.5 && aspectRatio < 3.5) {
                    // Valid car proportions - minimal adjustment
                    // 有效的汽车比例 - 最小调整
                    adjustmentFactor *= 1.005;
                } else if (aspectRatio < 1.0) {
                    // Unusually tall car box - tighten
                    // 异常高的汽车框 - 收紧
                    adjustmentFactor *= 0.95;
                }
                break;
        }
        
        // More conservative size-based adjustments
        // 更保守的基于大小的调整
        const objectArea = width * height;
        if (objectArea < 1500) { // Very small objects
            adjustmentFactor *= 1.03; // Moderate expansion for very small objects
            paddingRatio += 0.01;
        } else if (objectArea > 80000) { // Very large objects
            adjustmentFactor *= 0.99; // Minimal tightening for very large objects
        }
        
        // Calculate optimized dimensions with precision preservation
        // 计算优化后的尺寸，保持精度
        const centerX = x + width / 2;
        const centerY = y + height / 2;
        
        // Apply adjustments more conservatively
        // 更保守地应用调整
        const newWidth = width * adjustmentFactor;
        const newHeight = height * adjustmentFactor;
        
        // Minimal padding to avoid over-expansion
        // 最小填充以避免过度扩展
        const paddedWidth = newWidth * (1 + paddingRatio);
        const paddedHeight = newHeight * (1 + paddingRatio);
        
        // Ensure the adjusted bbox stays within reasonable bounds
        // 确保调整后的边界框保持在合理范围内
        const finalX = Math.max(0, centerX - paddedWidth / 2);
        const finalY = Math.max(0, centerY - paddedHeight / 2);
        
        // Additional validation to prevent extreme adjustments
        // 额外验证以防止极端调整
        const maxAllowedChange = 0.15; // Maximum 15% change
        const widthChange = Math.abs(paddedWidth - width) / width;
        const heightChange = Math.abs(paddedHeight - height) / height;
        
        if (widthChange > maxAllowedChange || heightChange > maxAllowedChange) {
            // If adjustment is too large, use original bbox
            // 如果调整过大，使用原始边界框
            return bbox;
        }
        
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
     * Validate aspect ratio for object class
     * 验证目标类别的宽高比
     */
    validateAspectRatio(bbox, objectClass) {
        const [x, y, w, h] = bbox;
        const aspectRatio = w / h;
        
        const validRanges = {
            'person': { min: 0.25, max: 0.9 },
            'car': { min: 1.2, max: 4.0 }
        };
        
        const range = validRanges[objectClass];
        if (!range) return true; // Unknown class, assume valid
        
        return aspectRatio >= range.min && aspectRatio <= range.max;
    }
    
    /**
     * Validate object size reasonableness
     * 验证目标大小的合理性
     */
    validateObjectSize(bbox, imageWidth, imageHeight) {
        const [x, y, w, h] = bbox;
        const objectArea = w * h;
        const imageArea = imageWidth * imageHeight;
        const sizeRatio = objectArea / imageArea;
        
        // Object should be between 0.1% and 80% of image area
        // 目标应该在图像面积的0.1%到80%之间
        return sizeRatio >= 0.001 && sizeRatio <= 0.8;
    }
    
    /**
     * Validate object position
     * 验证目标位置
     */
    validatePosition(bbox, imageWidth, imageHeight) {
        const [x, y, w, h] = bbox;
        
        // Check if bbox is completely within image bounds
        // 检查边界框是否完全在图像边界内
        return x >= 0 && y >= 0 && 
               x + w <= imageWidth && y + h <= imageHeight &&
               w > 0 && h > 0;
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
     * Enhanced coordinate scaling for different display contexts
     * 针对不同显示上下文的增强坐标缩放
     */
    scaleCoordinatesForDisplay(bbox, fromSize, toSize) {
        const [x, y, w, h] = bbox;
        const scaleX = toSize.width / fromSize.width;
        const scaleY = toSize.height / fromSize.height;
        
        return [
            x * scaleX,
            y * scaleY,
            w * scaleX,
            h * scaleY
        ];
    }
    
    /**
     * Validate and correct detection coordinates for accuracy
     * 验证并修正检测坐标以提高准确性
     */
    validateAndCorrectCoordinates(detection, imageWidth, imageHeight) {
        const [x, y, w, h] = detection.bbox;
        
        // Check for common coordinate errors
        // 检查常见的坐标错误
        let correctedBbox = [x, y, w, h];
        let needsCorrection = false;
        
        // Fix negative coordinates
        // 修复负坐标
        if (x < 0) {
            correctedBbox[2] = Math.max(1, w + x); // Adjust width
            correctedBbox[0] = 0;
            needsCorrection = true;
        }
        if (y < 0) {
            correctedBbox[3] = Math.max(1, h + y); // Adjust height
            correctedBbox[1] = 0;
            needsCorrection = true;
        }
        
        // Fix out-of-bounds coordinates
        // 修复超出边界的坐标
        if (correctedBbox[0] + correctedBbox[2] > imageWidth) {
            correctedBbox[2] = Math.max(1, imageWidth - correctedBbox[0]);
            needsCorrection = true;
        }
        if (correctedBbox[1] + correctedBbox[3] > imageHeight) {
            correctedBbox[3] = Math.max(1, imageHeight - correctedBbox[1]);
            needsCorrection = true;
        }
        
        if (needsCorrection) {
            detection.bbox = correctedBbox;
            detection.wasCorrected = true;
            // Reduce confidence for corrected detections
            // 降低修正检测的置信度
            detection.score *= 0.9;
        }
        
        return detection;
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
     * Apply enhanced temporal smoothing to reduce detection jitter with improved accuracy
     * 应用增强的时间平滑以减少检测抖动，提高准确性
     */
    applyTemporalSmoothing(currentDetections, previousDetections) {
        if (!previousDetections || previousDetections.length === 0) {
            return currentDetections;
        }
        
        return currentDetections.map(current => {
            // Find closest previous detection with stricter matching
            // 找到最接近的先前检测，使用更严格的匹配
            const closest = this.findClosestDetection(current, previousDetections);
            
            if (closest && this.calculateIoU(current.bbox, closest.bbox) > 0.4) {
                // Apply adaptive smoothing based on confidence and movement
                // 基于置信度和移动应用自适应平滑
                const confidenceRatio = current.score / closest.score;
                const movementDistance = this.calculateCenterDistance(current.bbox, closest.bbox);
                const maxDimension = Math.max(current.bbox[2], current.bbox[3]);
                const normalizedMovement = movementDistance / maxDimension;
                
                // Adjust smoothing factor based on confidence and movement
                // 根据置信度和移动调整平滑因子
                let smoothingFactor = 0.2; // More conservative base smoothing
                
                if (confidenceRatio > 1.2) {
                    // Current detection is much more confident
                    // 当前检测置信度更高
                    smoothingFactor = 0.1;
                } else if (confidenceRatio < 0.8) {
                    // Previous detection was more confident
                    // 先前检测置信度更高
                    smoothingFactor = 0.3;
                }
                
                if (normalizedMovement > 0.3) {
                    // Large movement - reduce smoothing
                    // 大幅移动 - 减少平滑
                    smoothingFactor *= 0.5;
                }
                
                const [cx, cy, cw, ch] = current.bbox;
                const [px, py, pw, ph] = closest.bbox;
                
                return {
                    ...current,
                    bbox: [
                        cx * (1 - smoothingFactor) + px * smoothingFactor,
                        cy * (1 - smoothingFactor) + py * smoothingFactor,
                        cw * (1 - smoothingFactor) + pw * smoothingFactor,
                        ch * (1 - smoothingFactor) + ph * smoothingFactor
                    ],
                    // Add smoothing metadata
                    // 添加平滑元数据
                    smoothingInfo: {
                        factor: smoothingFactor,
                        confidenceRatio: confidenceRatio,
                        movement: normalizedMovement
                    }
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
        
        // Adaptive distance threshold based on object size
        // 基于目标大小的自适应距离阈值
        const maxDimension = Math.max(target.bbox[2], target.bbox[3]);
        const adaptiveThreshold = Math.min(100, maxDimension * 0.5);
        
        return minDistance < adaptiveThreshold ? closest : null;
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
     * Apply Non-Maximum Suppression to remove overlapping detections with improved accuracy
     * 应用非极大值抑制移除重叠检测，提高准确性
     */
    applyNonMaxSuppression(detections, iouThreshold = 0.4) {
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
                    
                    // Enhanced suppression logic with size and confidence consideration
                    // 增强的抑制逻辑，考虑大小和置信度
                    if (iou > iouThreshold) {
                        // Additional check: suppress smaller box if confidence difference is small
                        // 额外检查：如果置信度差异小，抑制较小的框
                        const confidenceDiff = currentDet.score - otherDet.score;
                        const currentArea = currentDet.bbox[2] * currentDet.bbox[3];
                        const otherArea = otherDet.bbox[2] * otherDet.bbox[3];
                        
                        if (confidenceDiff < 0.1 && otherArea < currentArea * 0.8) {
                            suppressed.add(j); // Suppress smaller, similar-confidence detection
                        } else if (confidenceDiff >= 0.1) {
                            suppressed.add(j); // Suppress lower confidence detection
                        }
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
    
    /**
     * Handle partial visibility and occlusion cases for better detection continuity
     * 处理部分可见性和遮挡情况以提高检测连续性
     */
    handlePartialVisibility(detections, videoElement) {
        const imageWidth = videoElement.naturalWidth || videoElement.width || 640;
        const imageHeight = videoElement.naturalHeight || videoElement.height || 480;
        
        const enhancedDetections = [...detections];
        
        // Analyze each detection for partial visibility
        // 分析每个检测的部分可见性
        detections.forEach(detection => {
            const [x, y, w, h] = detection.bbox;
            const edgeInfo = this.analyzeEdgeProximity(x, y, w, h, imageWidth, imageHeight);
            
            if (edgeInfo.isNearEdge) {
                // Enhance detection for edge cases
                // 增强边缘情况的检测
                this.enhanceEdgeDetection(detection, edgeInfo, imageWidth, imageHeight);
            }
        });
        
        // Try to recover partially occluded objects from previous frames
        // 尝试从先前帧恢复部分遮挡的目标
        const recoveredDetections = this.recoverPartiallyOccludedObjects(enhancedDetections, imageWidth, imageHeight);
        
        // Merge recovered detections with current ones
        // 将恢复的检测与当前检测合并
        return this.mergeDetections(enhancedDetections, recoveredDetections);
    }
    
    /**
     * Analyze proximity to image edges for partial visibility detection
     * 分析与图像边缘的接近程度以检测部分可见性
     */
    analyzeEdgeProximity(x, y, w, h, imageWidth, imageHeight) {
        const threshold = this.occlusionHandler.boundaryThreshold;
        const edgeThreshold = Math.min(imageWidth, imageHeight) * threshold;
        
        const edgeInfo = {
            isNearEdge: false,
            edges: [],
            visibilityRatio: 1.0,
            truncationInfo: {}
        };
        
        // Check each edge
        // 检查每个边缘
        if (x < edgeThreshold) {
            edgeInfo.isNearEdge = true;
            edgeInfo.edges.push('left');
            edgeInfo.truncationInfo.left = Math.max(0, -x);
        }
        if (y < edgeThreshold) {
            edgeInfo.isNearEdge = true;
            edgeInfo.edges.push('top');
            edgeInfo.truncationInfo.top = Math.max(0, -y);
        }
        if (x + w > imageWidth - edgeThreshold) {
            edgeInfo.isNearEdge = true;
            edgeInfo.edges.push('right');
            edgeInfo.truncationInfo.right = Math.max(0, (x + w) - imageWidth);
        }
        if (y + h > imageHeight - edgeThreshold) {
            edgeInfo.isNearEdge = true;
            edgeInfo.edges.push('bottom');
            edgeInfo.truncationInfo.bottom = Math.max(0, (y + h) - imageHeight);
        }
        
        // Calculate visible area ratio
        // 计算可见区域比例
        const visibleX = Math.max(0, x);
        const visibleY = Math.max(0, y);
        const visibleW = Math.min(w, imageWidth - visibleX);
        const visibleH = Math.min(h, imageHeight - visibleY);
        const visibleArea = Math.max(0, visibleW * visibleH);
        const totalArea = w * h;
        
        edgeInfo.visibilityRatio = totalArea > 0 ? visibleArea / totalArea : 0;
        
        return edgeInfo;
    }
    
    /**
     * Enhance detection for objects near edges with extrapolation
     * 通过外推增强边缘附近目标的检测
     */
    enhanceEdgeDetection(detection, edgeInfo, imageWidth, imageHeight) {
        const [x, y, w, h] = detection.bbox;
        
        // Boost confidence for edge detections if visibility is sufficient
        // 如果可见性足够，提升边缘检测的置信度
        if (edgeInfo.visibilityRatio >= this.occlusionHandler.minVisibleRatio) {
            detection.score = Math.min(1.0, detection.score + this.occlusionHandler.confidenceBoost);
            
            // Add metadata for tracking system
            // 为追踪系统添加元数据
            detection.edgeInfo = {
                isPartiallyVisible: true,
                visibilityRatio: edgeInfo.visibilityRatio,
                nearEdges: edgeInfo.edges,
                estimatedFullBbox: this.estimateFullBoundingBox(detection, edgeInfo, imageWidth, imageHeight)
            };
        }
    }
    
    /**
     * Estimate full bounding box for partially visible objects
     * 估计部分可见目标的完整边界框
     */
    estimateFullBoundingBox(detection, edgeInfo, imageWidth, imageHeight) {
        const [x, y, w, h] = detection.bbox;
        let estimatedX = x, estimatedY = y, estimatedW = w, estimatedH = h;
        
        // Use class-specific aspect ratios for extrapolation
        // 使用类别特定的宽高比进行外推
        const expectedAspectRatios = {
            'person': { min: 0.3, max: 0.8, typical: 0.5 }, // Height > Width for persons
            'car': { min: 1.5, max: 3.0, typical: 2.2 }     // Width > Height for cars
        };
        
        const aspectRatio = expectedAspectRatios[detection.class];
        
        if (aspectRatio) {
            // Extrapolate based on visible portion and expected aspect ratio
            // 基于可见部分和预期宽高比进行外推
            if (edgeInfo.edges.includes('left') || edgeInfo.edges.includes('right')) {
                // Horizontal truncation - estimate full width
                // 水平截断 - 估计完整宽度
                if (detection.class === 'person') {
                    estimatedH = h; // Keep height
                    estimatedW = estimatedH * aspectRatio.typical;
                    estimatedX = edgeInfo.edges.includes('left') ? 
                        Math.max(0, x - (estimatedW - w)) : x;
                } else if (detection.class === 'car') {
                    estimatedH = h; // Keep height
                    estimatedW = estimatedH * aspectRatio.typical;
                    estimatedX = edgeInfo.edges.includes('left') ? 
                        Math.max(0, x - (estimatedW - w)) : x;
                }
            }
            
            if (edgeInfo.edges.includes('top') || edgeInfo.edges.includes('bottom')) {
                // Vertical truncation - estimate full height
                // 垂直截断 - 估计完整高度
                if (detection.class === 'person') {
                    estimatedW = w; // Keep width
                    estimatedH = estimatedW / aspectRatio.typical;
                    estimatedY = edgeInfo.edges.includes('top') ? 
                        Math.max(0, y - (estimatedH - h)) : y;
                } else if (detection.class === 'car') {
                    estimatedW = w; // Keep width
                    estimatedH = estimatedW / aspectRatio.typical;
                    estimatedY = edgeInfo.edges.includes('top') ? 
                        Math.max(0, y - (estimatedH - h)) : y;
                }
            }
        }
        
        return [estimatedX, estimatedY, estimatedW, estimatedH];
    }
    
    /**
     * Recover partially occluded objects using motion prediction and history
     * 使用运动预测和历史记录恢复部分遮挡的目标
     */
    recoverPartiallyOccludedObjects(currentDetections, imageWidth, imageHeight) {
        const recoveredDetections = [];
        
        // Look for missing objects that might be partially occluded
        // 寻找可能部分遮挡的缺失目标
        if (this.previousDetections && this.previousDetections.length > 0) {
            this.previousDetections.forEach(prevDet => {
                // Check if this object is missing in current frame
                // 检查此目标是否在当前帧中缺失
                const hasMatch = currentDetections.some(currDet => {
                    const iou = this.calculateIoU(prevDet.bbox, currDet.bbox);
                    return iou > 0.3 && currDet.class === prevDet.class;
                });
                
                if (!hasMatch && prevDet.edgeInfo?.isPartiallyVisible) {
                    // Try to predict where this object might be now
                    // 尝试预测此目标现在可能的位置
                    const predictedDetection = this.predictOccludedObjectPosition(prevDet, imageWidth, imageHeight);
                    
                    if (predictedDetection) {
                        recoveredDetections.push(predictedDetection);
                    }
                }
            });
        }
        
        return recoveredDetections;
    }
    
    /**
     * Predict position of occluded object based on motion history
     * 基于运动历史预测遮挡目标的位置
     */
    predictOccludedObjectPosition(previousDetection, imageWidth, imageHeight) {
        // Simple motion prediction - can be enhanced with Kalman filter
        // 简单运动预测 - 可以用卡尔曼滤波器增强
        const [prevX, prevY, prevW, prevH] = previousDetection.bbox;
        
        // Estimate velocity from trajectory if available
        // 如果可用，从轨迹估计速度
        let vx = 0, vy = 0;
        if (previousDetection.motionHistory && previousDetection.motionHistory.length >= 2) {
            const recent = previousDetection.motionHistory.slice(-2);
            vx = recent[1].x - recent[0].x;
            vy = recent[1].y - recent[0].y;
        }
        
        // Predict new position
        // 预测新位置
        const predictedX = prevX + vx;
        const predictedY = prevY + vy;
        
        // Check if predicted position is still partially visible
        // 检查预测位置是否仍然部分可见
        const edgeInfo = this.analyzeEdgeProximity(predictedX, predictedY, prevW, prevH, imageWidth, imageHeight);
        
        if (edgeInfo.visibilityRatio >= this.occlusionHandler.minVisibleRatio) {
            return {
                ...previousDetection,
                bbox: [predictedX, predictedY, prevW, prevH],
                score: previousDetection.score * 0.7, // Reduce confidence for predicted
                isPredicted: true,
                edgeInfo: edgeInfo
            };
        }
        
        return null;
    }
    
    /**
     * Merge current and recovered detections, avoiding duplicates
     * 合并当前和恢复的检测，避免重复
     */
    mergeDetections(currentDetections, recoveredDetections) {
        const merged = [...currentDetections];
        
        recoveredDetections.forEach(recovered => {
            // Check if this recovered detection conflicts with current ones
            // 检查此恢复的检测是否与当前检测冲突
            const hasConflict = currentDetections.some(current => {
                const iou = this.calculateIoU(recovered.bbox, current.bbox);
                return iou > 0.3 && current.class === recovered.class;
            });
            
            if (!hasConflict) {
                merged.push(recovered);
            }
        });
        
        return merged;
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
