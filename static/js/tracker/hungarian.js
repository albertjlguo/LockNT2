/**
 * Hungarian Algorithm for Optimal Assignment
 * 匈牙利算法用于最优分配
 * 
 * Solves the assignment problem for track-detection association
 * 解决轨迹-检测关联的分配问题
 */
class HungarianAssignment {
    constructor() {
        // Maximum cost threshold - assignments above this are rejected
        // 最大成本阈值 - 超过此值的分配将被拒绝
        this.maxCost = 1000;
        this.infinity = 999999;
        
        // Temporary matrices for calculations
        // 计算用临时矩阵
        this.tempMatrix = null;
        this.rowCover = null;
        this.colCover = null;
        this.path = null;
    }
    
    /**
     * Solve assignment problem using Hungarian algorithm
     * 使用匈牙利算法解决分配问题
     * 
     * @param {Array<Array<number>>} costMatrix - Cost matrix where costMatrix[i][j] is cost of assigning track i to detection j
     * @param {number} maxCost - Maximum allowed cost for valid assignment
     * @returns {Array<{trackIdx: number, detectionIdx: number, cost: number}>} - Optimal assignments
     */
    solve(costMatrix, maxCost = this.maxCost) {
        if (!costMatrix || costMatrix.length === 0 || costMatrix[0].length === 0) {
            return [];
        }
        
        const numTracks = costMatrix.length;
        const numDetections = costMatrix[0].length;
        const size = Math.max(numTracks, numDetections);
        
        // Create square matrix by padding with high costs
        // 通过填充高成本创建方阵
        const squareMatrix = this.createSquareMatrix(costMatrix, size);
        
        // Initialize working arrays
        // 初始化工作数组
        this.initializeArrays(size);
        
        // Execute Hungarian algorithm steps
        // 执行匈牙利算法步骤
        const assignments = this.hungarianAlgorithm(squareMatrix, size);
        
        // Filter valid assignments within cost threshold
        // 过滤成本阈值内的有效分配
        const validAssignments = [];
        for (const assignment of assignments) {
            const { trackIdx, detectionIdx } = assignment;
            
            // Check if assignment is within original matrix bounds and cost threshold
            // 检查分配是否在原始矩阵边界和成本阈值内
            if (trackIdx < numTracks && detectionIdx < numDetections) {
                const originalCost = costMatrix[trackIdx][detectionIdx];
                if (originalCost <= maxCost) {
                    validAssignments.push({
                        trackIdx,
                        detectionIdx,
                        cost: originalCost
                    });
                }
            }
        }
        
        return validAssignments;
    }
    
    /**
     * Create square matrix by padding with infinity costs
     * 通过填充无穷大成本创建方阵
     */
    createSquareMatrix(costMatrix, size) {
        const squareMatrix = [];
        
        for (let i = 0; i < size; i++) {
            squareMatrix[i] = [];
            for (let j = 0; j < size; j++) {
                if (i < costMatrix.length && j < costMatrix[0].length) {
                    squareMatrix[i][j] = costMatrix[i][j];
                } else {
                    squareMatrix[i][j] = this.infinity;
                }
            }
        }
        
        return squareMatrix;
    }
    
    /**
     * Initialize working arrays for Hungarian algorithm
     * 为匈牙利算法初始化工作数组
     */
    initializeArrays(size) {
        this.tempMatrix = Array(size).fill().map(() => Array(size).fill(0));
        this.rowCover = Array(size).fill(false);
        this.colCover = Array(size).fill(false);
        this.path = Array(size * 2).fill().map(() => Array(2).fill(0));
    }
    
    /**
     * Main Hungarian algorithm implementation
     * 主要的匈牙利算法实现
     */
    hungarianAlgorithm(costMatrix, size) {
        // Copy cost matrix to working matrix
        // 将成本矩阵复制到工作矩阵
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                this.tempMatrix[i][j] = costMatrix[i][j];
            }
        }
        
        // Step 1: Subtract row minima
        // 步骤1：减去行最小值
        this.subtractRowMinima(size);
        
        // Step 2: Subtract column minima
        // 步骤2：减去列最小值
        this.subtractColumnMinima(size);
        
        // Step 3: Cover zeros and find minimum number of lines
        // 步骤3：覆盖零并找到最小线数
        let step = 3;
        let done = false;
        
        while (!done) {
            switch (step) {
                case 3:
                    step = this.step3(size);
                    break;
                case 4:
                    step = this.step4(size);
                    break;
                case 5:
                    step = this.step5(size);
                    break;
                case 6:
                    step = this.step6(size);
                    break;
                default:
                    done = true;
                    break;
            }
        }
        
        // Extract assignments
        // 提取分配结果
        return this.extractAssignments(size);
    }
    
    /**
     * Step 1: Subtract minimum value from each row
     * 步骤1：从每行减去最小值
     */
    subtractRowMinima(size) {
        for (let i = 0; i < size; i++) {
            let minVal = this.infinity;
            
            // Find minimum in row
            // 找到行中的最小值
            for (let j = 0; j < size; j++) {
                if (this.tempMatrix[i][j] < minVal) {
                    minVal = this.tempMatrix[i][j];
                }
            }
            
            // Subtract minimum from all elements in row
            // 从行中所有元素减去最小值
            for (let j = 0; j < size; j++) {
                this.tempMatrix[i][j] -= minVal;
            }
        }
    }
    
    /**
     * Step 2: Subtract minimum value from each column
     * 步骤2：从每列减去最小值
     */
    subtractColumnMinima(size) {
        for (let j = 0; j < size; j++) {
            let minVal = this.infinity;
            
            // Find minimum in column
            // 找到列中的最小值
            for (let i = 0; i < size; i++) {
                if (this.tempMatrix[i][j] < minVal) {
                    minVal = this.tempMatrix[i][j];
                }
            }
            
            // Subtract minimum from all elements in column
            // 从列中所有元素减去最小值
            for (let i = 0; i < size; i++) {
                this.tempMatrix[i][j] -= minVal;
            }
        }
    }
    
    /**
     * Step 3: Cover all zeros with minimum number of lines
     * 步骤3：用最少的线覆盖所有零
     */
    step3(size) {
        // Clear covers
        // 清除覆盖
        this.rowCover.fill(false);
        this.colCover.fill(false);
        
        // Mark zeros and cover columns
        // 标记零并覆盖列
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                if (this.tempMatrix[i][j] === 0 && !this.rowCover[i] && !this.colCover[j]) {
                    this.tempMatrix[i][j] = -1; // Mark as starred zero
                    this.rowCover[i] = true;
                    this.colCover[j] = true;
                }
            }
        }
        
        // Clear row covers for next step
        // 为下一步清除行覆盖
        this.rowCover.fill(false);
        
        // Cover columns with starred zeros
        // 覆盖有星标零的列
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                if (this.tempMatrix[i][j] === -1) {
                    this.colCover[j] = true;
                }
            }
        }
        
        // Count covered columns
        // 计算覆盖的列数
        let coveredCols = 0;
        for (let j = 0; j < size; j++) {
            if (this.colCover[j]) coveredCols++;
        }
        
        return coveredCols >= size ? 7 : 4; // Done if all columns covered
    }
    
    /**
     * Step 4: Find uncovered zero and create alternating path
     * 步骤4：找到未覆盖的零并创建交替路径
     */
    step4(size) {
        let row = -1, col = -1;
        let done = false;
        
        while (!done) {
            // Find uncovered zero
            // 找到未覆盖的零
            const zero = this.findUncoveredZero(size);
            
            if (zero.row === -1) {
                done = true;
                return 6; // No uncovered zeros, go to step 6
            }
            
            row = zero.row;
            col = zero.col;
            
            // Prime the zero
            // 标记零为质数
            this.tempMatrix[row][col] = -2;
            
            // Check if there's a starred zero in this row
            // 检查此行是否有星标零
            const starCol = this.findStarInRow(row, size);
            
            if (starCol !== -1) {
                // Cover this row and uncover the star column
                // 覆盖此行并取消覆盖星标列
                this.rowCover[row] = true;
                this.colCover[starCol] = false;
            } else {
                // No starred zero in row, construct alternating path
                // 行中没有星标零，构造交替路径
                done = true;
                return this.step5(size, row, col);
            }
        }
        
        return 4;
    }
    
    /**
     * Step 5: Construct alternating path and update stars
     * 步骤5：构造交替路径并更新星标
     */
    step5(size, primeRow = -1, primeCol = -1) {
        if (primeRow === -1 || primeCol === -1) {
            // Find a primed zero
            // 找到一个质数零
            const prime = this.findPrimedZero(size);
            if (prime.row === -1) return 3;
            primeRow = prime.row;
            primeCol = prime.col;
        }
        
        let pathCount = 1;
        this.path[0][0] = primeRow;
        this.path[0][1] = primeCol;
        
        let done = false;
        while (!done) {
            // Find star in column of last primed zero
            // 在最后质数零的列中找星标
            const starRow = this.findStarInCol(this.path[pathCount - 1][1], size);
            
            if (starRow !== -1) {
                // Add star to path
                // 将星标添加到路径
                this.path[pathCount][0] = starRow;
                this.path[pathCount][1] = this.path[pathCount - 1][1];
                pathCount++;
                
                // Find prime in row of star
                // 在星标行中找质数
                const primeCol = this.findPrimeInRow(starRow, size);
                this.path[pathCount][0] = starRow;
                this.path[pathCount][1] = primeCol;
                pathCount++;
            } else {
                done = true;
            }
        }
        
        // Update stars and primes along path
        // 沿路径更新星标和质数
        for (let i = 0; i < pathCount; i++) {
            const row = this.path[i][0];
            const col = this.path[i][1];
            
            if (this.tempMatrix[row][col] === -1) {
                this.tempMatrix[row][col] = 0; // Unstar
            } else if (this.tempMatrix[row][col] === -2) {
                this.tempMatrix[row][col] = -1; // Star
            }
        }
        
        // Clear covers and primes
        // 清除覆盖和质数
        this.clearCovers(size);
        this.clearPrimes(size);
        
        return 3;
    }
    
    /**
     * Step 6: Add value to all covered elements and subtract from uncovered
     * 步骤6：向所有覆盖元素添加值，从未覆盖元素减去值
     */
    step6(size) {
        // Find minimum uncovered value
        // 找到最小未覆盖值
        let minVal = this.infinity;
        
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                if (!this.rowCover[i] && !this.colCover[j]) {
                    if (this.tempMatrix[i][j] < minVal) {
                        minVal = this.tempMatrix[i][j];
                    }
                }
            }
        }
        
        // Add to covered rows, subtract from uncovered columns
        // 向覆盖行添加，从未覆盖列减去
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                if (this.rowCover[i]) {
                    this.tempMatrix[i][j] += minVal;
                }
                if (!this.colCover[j]) {
                    this.tempMatrix[i][j] -= minVal;
                }
            }
        }
        
        return 4;
    }
    
    /**
     * Extract final assignments from starred zeros
     * 从星标零提取最终分配
     */
    extractAssignments(size) {
        const assignments = [];
        
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                if (this.tempMatrix[i][j] === -1) {
                    assignments.push({
                        trackIdx: i,
                        detectionIdx: j
                    });
                }
            }
        }
        
        return assignments;
    }
    
    // ======================== Helper Functions ========================
    
    findUncoveredZero(size) {
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                if (this.tempMatrix[i][j] === 0 && !this.rowCover[i] && !this.colCover[j]) {
                    return { row: i, col: j };
                }
            }
        }
        return { row: -1, col: -1 };
    }
    
    findStarInRow(row, size) {
        for (let j = 0; j < size; j++) {
            if (this.tempMatrix[row][j] === -1) {
                return j;
            }
        }
        return -1;
    }
    
    findStarInCol(col, size) {
        for (let i = 0; i < size; i++) {
            if (this.tempMatrix[i][col] === -1) {
                return i;
            }
        }
        return -1;
    }
    
    findPrimeInRow(row, size) {
        for (let j = 0; j < size; j++) {
            if (this.tempMatrix[row][j] === -2) {
                return j;
            }
        }
        return -1;
    }
    
    findPrimedZero(size) {
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                if (this.tempMatrix[i][j] === -2) {
                    return { row: i, col: j };
                }
            }
        }
        return { row: -1, col: -1 };
    }
    
    clearCovers(size) {
        this.rowCover.fill(false);
        this.colCover.fill(false);
    }
    
    clearPrimes(size) {
        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                if (this.tempMatrix[i][j] === -2) {
                    this.tempMatrix[i][j] = 0;
                }
            }
        }
    }
}

/**
 * Data Association Manager
 * 数据关联管理器
 * 
 * Handles track-detection association using multiple cost functions
 * 使用多种成本函数处理轨迹-检测关联
 */
class DataAssociationManager {
    constructor(options = {}) {
        this.hungarian = new HungarianAssignment();
        
        // Cost function weights
        // 成本函数权重
        this.weights = {
            iou: options.iouWeight || 0.3,
            distance: options.distanceWeight || 0.25,
            appearance: options.appearanceWeight || 0.25,
            motion: options.motionWeight || 0.2
        };
        
        // Gating thresholds
        // 门控阈值
        this.gating = {
            maxDistance: options.maxDistance || 150,
            minIoU: options.minIoU || 0.1,
            maxAppearanceDist: options.maxAppearanceDist || 0.8
        };
        
        // Cost thresholds for different scenarios
        // 不同场景的成本阈值
        this.thresholds = {
            normal: options.normalThreshold || 0.7,
            crowded: options.crowdedThreshold || 0.6,
            locked: options.lockedThreshold || 0.8
        };
    }
    
    /**
     * Associate tracks with detections using optimal assignment
     * 使用最优分配关联轨迹与检测
     */
    associate(tracks, detections, appearanceExtractor = null, ctx = null) {
        if (tracks.length === 0 || detections.length === 0) {
            return {
                assignments: [],
                unmatchedTracks: tracks.map((_, i) => i),
                unmatchedDetections: detections.map((_, i) => i)
            };
        }
        
        // Build cost matrix
        // 构建成本矩阵
        const costMatrix = this.buildCostMatrix(tracks, detections, appearanceExtractor, ctx);
        
        // Determine appropriate threshold based on scene complexity
        // 根据场景复杂度确定适当阈值
        const threshold = this.getAdaptiveThreshold(tracks, detections);
        
        // Solve assignment problem
        // 解决分配问题
        const assignments = this.hungarian.solve(costMatrix, threshold);
        
        // Identify unmatched tracks and detections
        // 识别未匹配的轨迹和检测
        const matchedTrackIds = new Set(assignments.map(a => a.trackIdx));
        const matchedDetectionIds = new Set(assignments.map(a => a.detectionIdx));
        
        const unmatchedTracks = tracks
            .map((_, i) => i)
            .filter(i => !matchedTrackIds.has(i));
        
        const unmatchedDetections = detections
            .map((_, i) => i)
            .filter(i => !matchedDetectionIds.has(i));
        
        return {
            assignments,
            unmatchedTracks,
            unmatchedDetections,
            costMatrix // For debugging
        };
    }
    
    /**
     * Build cost matrix for track-detection pairs
     * 为轨迹-检测对构建成本矩阵
     */
    buildCostMatrix(tracks, detections, appearanceExtractor, ctx) {
        const costMatrix = [];
        
        for (let i = 0; i < tracks.length; i++) {
            costMatrix[i] = [];
            const track = tracks[i];
            
            for (let j = 0; j < detections.length; j++) {
                const detection = detections[j];
                
                // Apply gating to reject impossible associations
                // 应用门控拒绝不可能的关联
                if (!this.passesGating(track, detection)) {
                    costMatrix[i][j] = this.hungarian.infinity;
                    continue;
                }
                
                // Calculate multi-modal cost
                // 计算多模态成本
                const cost = this.calculateAssociationCost(
                    track, detection, appearanceExtractor, ctx
                );
                
                costMatrix[i][j] = cost;
            }
        }
        
        return costMatrix;
    }
    
    /**
     * Calculate comprehensive association cost
     * 计算综合关联成本
     */
    calculateAssociationCost(track, detection, appearanceExtractor, ctx) {
        // IoU cost (1 - IoU)
        // IoU成本 (1 - IoU)
        const iouCost = 1 - this.calculateIoU(track.bbox, detection);
        
        // Distance cost (normalized)
        // 距离成本（归一化）
        const distanceCost = this.calculateDistanceCost(track, detection);
        
        // Motion consistency cost
        // 运动一致性成本
        const motionCost = this.calculateMotionCost(track, detection);
        
        // Appearance cost
        // 外观成本
        let appearanceCost = 0.5; // Default neutral cost
        if (appearanceExtractor && ctx && track.appearanceFeatures) {
            appearanceCost = this.calculateAppearanceCost(
                track, detection, appearanceExtractor, ctx
            );
        }
        
        // Weighted combination
        // 加权组合
        const totalCost = 
            this.weights.iou * iouCost +
            this.weights.distance * distanceCost +
            this.weights.motion * motionCost +
            this.weights.appearance * appearanceCost;
        
        return totalCost;
    }
    
    /**
     * Check if track-detection pair passes gating criteria
     * 检查轨迹-检测对是否通过门控标准
     */
    passesGating(track, detection) {
        // Distance gating
        // 距离门控
        const distance = this.calculateCenterDistance(track.bbox, detection);
        if (distance > this.gating.maxDistance) {
            return false;
        }
        
        // IoU gating
        // IoU门控
        const iou = this.calculateIoU(track.bbox, detection);
        if (iou < this.gating.minIoU) {
            return false;
        }
        
        // Class consistency (if available)
        // 类别一致性（如果可用）
        if (track.class && detection.class && track.class !== detection.class) {
            return false;
        }
        
        return true;
    }
    
    /**
     * Get adaptive threshold based on scene complexity
     * 根据场景复杂度获取自适应阈值
     */
    getAdaptiveThreshold(tracks, detections) {
        const activeTracks = tracks.filter(t => t.lostFrames < 5).length;
        const isCrowded = activeTracks > 6 || detections.length > 8;
        const hasLockedTracks = tracks.some(t => t.locked);
        
        if (hasLockedTracks) {
            return this.thresholds.locked;
        } else if (isCrowded) {
            return this.thresholds.crowded;
        } else {
            return this.thresholds.normal;
        }
    }
    
    // ======================== Cost Calculation Functions ========================
    
    calculateIoU(bbox1, bbox2) {
        const b1 = this.normalizeBbox(bbox1);
        const b2 = this.normalizeBbox(bbox2);
        
        const x1 = Math.max(b1.x, b2.x);
        const y1 = Math.max(b1.y, b2.y);
        const x2 = Math.min(b1.x + b1.w, b2.x + b2.w);
        const y2 = Math.min(b1.y + b1.h, b2.y + b2.h);
        
        if (x2 <= x1 || y2 <= y1) return 0;
        
        const intersection = (x2 - x1) * (y2 - y1);
        const union = b1.w * b1.h + b2.w * b2.h - intersection;
        
        return union > 0 ? intersection / union : 0;
    }
    
    calculateCenterDistance(bbox1, bbox2) {
        const b1 = this.normalizeBbox(bbox1);
        const b2 = this.normalizeBbox(bbox2);
        
        const dx = (b1.x + b1.w / 2) - (b2.x + b2.w / 2);
        const dy = (b1.y + b1.h / 2) - (b2.y + b2.h / 2);
        
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    calculateDistanceCost(track, detection) {
        const distance = this.calculateCenterDistance(track.bbox, detection);
        return Math.min(1.0, distance / this.gating.maxDistance);
    }
    
    calculateMotionCost(track, detection) {
        if (!track.vx && !track.vy) return 0.5; // No motion info
        
        const predictedX = track.bbox.x + track.bbox.w / 2 + (track.vx || 0) / 30;
        const predictedY = track.bbox.y + track.bbox.h / 2 + (track.vy || 0) / 30;
        const detectionX = detection.x + detection.w / 2;
        const detectionY = detection.y + detection.h / 2;
        
        const motionError = Math.sqrt(
            (predictedX - detectionX) ** 2 + (predictedY - detectionY) ** 2
        );
        
        return Math.min(1.0, motionError / 100); // Normalize by 100 pixels
    }
    
    calculateAppearanceCost(track, detection, appearanceExtractor, ctx) {
        try {
            const detectionFeatures = appearanceExtractor.extractFeatures(
                ctx, detection, detection.class
            );
            
            if (!detectionFeatures || !track.appearanceFeatures) {
                return 0.5; // Neutral cost if features unavailable
            }
            
            const similarity = appearanceExtractor.calculateWeightedSimilarity(
                track.appearanceFeatures, detectionFeatures, detection.class
            );
            
            return 1 - similarity; // Convert similarity to cost
        } catch (error) {
            console.warn('Appearance cost calculation failed:', error);
            return 0.5;
        }
    }
    
    normalizeBbox(bbox) {
        if (bbox.x !== undefined) {
            return bbox; // Already normalized
        }
        
        // Handle array format [x, y, w, h]
        return {
            x: bbox[0],
            y: bbox[1],
            w: bbox[2],
            h: bbox[3]
        };
    }
}

// Export for use in other modules
// 导出供其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { HungarianAssignment, DataAssociationManager };
} else if (typeof window !== 'undefined') {
    window.HungarianAssignment = HungarianAssignment;
    window.DataAssociationManager = DataAssociationManager;
}
