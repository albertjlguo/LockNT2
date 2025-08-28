/**
 * Target Re-identification Manager
 * 目标重识别管理器
 * 
 * Handles long-term target re-identification with expanded search and appearance matching
 * 处理具有扩展搜索和外观匹配的长期目标重识别
 */
class ReidentificationManager {
    constructor(options = {}) {
        // Re-identification parameters
        // 重识别参数
        this.params = {
            expandedSearchRadius: options.expandedSearchRadius || 200,
            appearanceWeight: options.appearanceWeight || 0.6,
            motionWeight: options.motionWeight || 0.25,
            sizeWeight: options.sizeWeight || 0.15,
            minSimilarityThreshold: options.minSimilarityThreshold || 0.65,
            maxFramesForReidentification: options.maxFramesForReidentification || 90
        };
        
        // Feature bank for storing historical appearance features
        // 用于存储历史外观特征的特征库
        this.featureBank = new Map();
        
        // Re-identification history
        // 重识别历史
        this.reidentificationHistory = [];
        
        // Performance metrics
        // 性能指标
        this.metrics = {
            successfulReidentifications: 0,
            failedAttempts: 0,
            avgReidentificationTime: 0
        };
    }
    
    /**
     * Store appearance features for a track in the feature bank
     * 在特征库中存储轨迹的外观特征
     */
    storeTrackFeatures(track, features, frameCount) {
        if (!track.id || !features) return;
        
        if (!this.featureBank.has(track.id)) {
            this.featureBank.set(track.id, {
                features: [],
                lastUpdated: frameCount,
                trackInfo: {
                    class: track.class,
                    avgSize: { w: track.w, h: track.h },
                    lastPosition: { x: track.cx, y: track.cy }
                }
            });
        }
        
        const bankEntry = this.featureBank.get(track.id);
        
        // Store multiple feature snapshots for robust matching
        // 存储多个特征快照以实现稳健匹配
        bankEntry.features.push({
            features: this.deepCopyFeatures(features),
            frameCount: frameCount,
            confidence: track.confidence || 1.0
        });
        
        // Keep only recent features (sliding window)
        // 仅保留最近的特征（滑动窗口）
        const maxFeatures = 10;
        if (bankEntry.features.length > maxFeatures) {
            bankEntry.features = bankEntry.features.slice(-maxFeatures);
        }
        
        bankEntry.lastUpdated = frameCount;
        bankEntry.trackInfo.lastPosition = { x: track.cx, y: track.cy };
        bankEntry.trackInfo.avgSize = { w: track.w, h: track.h };
    }
    
    /**
     * Attempt to re-identify lost tracks with new detections
     * 尝试使用新检测重识别丢失的轨迹
     */
    attemptReidentification(lostTracks, detections, appearanceExtractor, ctx, frameCount) {
        const startTime = performance.now();
        const reidentifications = [];
        
        // Filter tracks eligible for re-identification
        // 过滤符合重识别条件的轨迹
        const eligibleTracks = lostTracks.filter(track => 
            track.lostFrames >= 30 && 
            track.lostFrames <= this.params.maxFramesForReidentification &&
            this.featureBank.has(track.id)
        );
        
        if (eligibleTracks.length === 0 || detections.length === 0) {
            return reidentifications;
        }
        
        // Build cost matrix for re-identification
        // 构建重识别成本矩阵
        const costMatrix = this.buildReidentificationMatrix(
            eligibleTracks, detections, appearanceExtractor, ctx, frameCount
        );
        
        // Find optimal assignments using Hungarian algorithm
        // 使用匈牙利算法找到最优分配
        const assignments = this.solveReidentificationAssignment(costMatrix);
        
        // Process successful re-identifications
        // 处理成功的重识别
        for (const assignment of assignments) {
            const track = eligibleTracks[assignment.trackIdx];
            const detection = detections[assignment.detectionIdx];
            
            if (assignment.cost <= (1 - this.params.minSimilarityThreshold)) {
                reidentifications.push({
                    track: track,
                    detection: detection,
                    detectionIndex: assignment.detectionIdx,
                    similarity: 1 - assignment.cost,
                    method: 'appearance_motion_fusion'
                });
                
                // Update metrics
                // 更新指标
                this.metrics.successfulReidentifications++;
                
                // Log successful re-identification
                // 记录成功的重识别
                this.logReidentification(track, detection, assignment.cost, frameCount);
            }
        }
        
        // Update performance metrics
        // 更新性能指标
        const endTime = performance.now();
        this.updatePerformanceMetrics(endTime - startTime, reidentifications.length);
        
        return reidentifications;
    }
    
    /**
     * Build cost matrix for track-detection re-identification
     * 构建轨迹-检测重识别的成本矩阵
     */
    buildReidentificationMatrix(tracks, detections, appearanceExtractor, ctx, frameCount) {
        const costMatrix = [];
        
        for (let i = 0; i < tracks.length; i++) {
            costMatrix[i] = [];
            const track = tracks[i];
            const bankEntry = this.featureBank.get(track.id);
            
            for (let j = 0; j < detections.length; j++) {
                const detection = detections[j];
                
                // Check basic constraints
                // 检查基本约束
                if (!this.passesBasicConstraints(track, detection, bankEntry)) {
                    costMatrix[i][j] = 999; // High cost for invalid pairs
                    continue;
                }
                
                // Calculate comprehensive re-identification cost
                // 计算综合重识别成本
                const cost = this.calculateReidentificationCost(
                    track, detection, bankEntry, appearanceExtractor, ctx, frameCount
                );
                
                costMatrix[i][j] = cost;
            }
        }
        
        return costMatrix;
    }
    
    /**
     * Check basic constraints for re-identification
     * 检查重识别的基本约束
     */
    passesBasicConstraints(track, detection, bankEntry) {
        // Class consistency
        // 类别一致性
        if (track.class && detection.class && track.class !== detection.class) {
            return false;
        }
        
        // Distance constraint with expanded search radius
        // 使用扩展搜索半径的距离约束
        const detectionCenter = {
            x: detection.x + detection.w / 2,
            y: detection.y + detection.h / 2
        };
        
        const distance = Math.sqrt(
            (detectionCenter.x - track.cx)**2 + 
            (detectionCenter.y - track.cy)**2
        );
        
        if (distance > this.params.expandedSearchRadius) {
            return false;
        }
        
        // Size constraint (not too different)
        // 尺寸约束（差异不能太大）
        const sizeRatio = Math.min(detection.w / track.w, track.w / detection.w) *
                         Math.min(detection.h / track.h, track.h / detection.h);
        
        if (sizeRatio < 0.3) { // Allow more size variation for long-term re-ID
            return false;
        }
        
        return true;
    }
    
    /**
     * Calculate comprehensive re-identification cost
     * 计算综合重识别成本
     */
    calculateReidentificationCost(track, detection, bankEntry, appearanceExtractor, ctx, frameCount) {
        // Appearance cost (most important for re-identification)
        // 外观成本（重识别最重要的因素）
        const appearanceCost = this.calculateAppearanceCost(
            detection, bankEntry, appearanceExtractor, ctx
        );
        
        // Motion prediction cost
        // 运动预测成本
        const motionCost = this.calculateMotionCost(track, detection, frameCount);
        
        // Size consistency cost
        // 尺寸一致性成本
        const sizeCost = this.calculateSizeCost(track, detection, bankEntry);
        
        // Temporal cost (penalty for longer occlusion)
        // 时间成本（对较长遮挡的惩罚）
        const temporalCost = this.calculateTemporalCost(track, frameCount);
        
        // Weighted combination with emphasis on appearance
        // 强调外观的加权组合
        const totalCost = 
            this.params.appearanceWeight * appearanceCost +
            this.params.motionWeight * motionCost +
            this.params.sizeWeight * sizeCost +
            0.1 * temporalCost;
        
        return Math.min(1.0, totalCost);
    }
    
    /**
     * Calculate appearance cost using feature bank
     * 使用特征库计算外观成本
     */
    calculateAppearanceCost(detection, bankEntry, appearanceExtractor, ctx) {
        if (!appearanceExtractor || !ctx || !bankEntry.features.length) {
            return 0.5; // Neutral cost if no appearance info
        }
        
        try {
            // Extract features for current detection
            // 为当前检测提取特征
            const detectionFeatures = appearanceExtractor.extractFeatures(
                ctx, detection, detection.class
            );
            
            if (!detectionFeatures) {
                return 0.5;
            }
            
            // Calculate similarity with all stored features
            // 计算与所有存储特征的相似度
            let maxSimilarity = 0;
            let weightedSimilarity = 0;
            let totalWeight = 0;
            
            for (const featureEntry of bankEntry.features) {
                const similarity = appearanceExtractor.calculateWeightedSimilarity(
                    featureEntry.features, detectionFeatures, detection.class
                );
                
                // Weight by confidence and recency
                // 按置信度和新近度加权
                const weight = featureEntry.confidence * Math.exp(-0.1 * 
                    Math.abs(featureEntry.frameCount - (Date.now() / 33.33))); // Rough frame estimate
                
                maxSimilarity = Math.max(maxSimilarity, similarity);
                weightedSimilarity += similarity * weight;
                totalWeight += weight;
            }
            
            // Use both max and weighted average for robustness
            // 使用最大值和加权平均值以提高稳健性
            const avgSimilarity = totalWeight > 0 ? weightedSimilarity / totalWeight : 0;
            const finalSimilarity = 0.7 * maxSimilarity + 0.3 * avgSimilarity;
            
            return 1 - finalSimilarity; // Convert to cost
            
        } catch (error) {
            console.warn('Appearance cost calculation failed in re-ID:', error);
            return 0.5;
        }
    }
    
    /**
     * Calculate motion prediction cost
     * 计算运动预测成本
     */
    calculateMotionCost(track, detection, frameCount) {
        // Predict where track should be based on last known motion
        // 基于最后已知运动预测轨迹应该在的位置
        const timeDelta = track.lostFrames / 30.0; // Convert frames to seconds
        const damping = Math.exp(-0.1 * timeDelta); // Velocity damping over time
        
        const predictedX = track.cx + (track.vx || 0) * timeDelta * damping;
        const predictedY = track.cy + (track.vy || 0) * timeDelta * damping;
        
        const detectionCenterX = detection.x + detection.w / 2;
        const detectionCenterY = detection.y + detection.h / 2;
        
        const motionError = Math.sqrt(
            (predictedX - detectionCenterX)**2 + 
            (predictedY - detectionCenterY)**2
        );
        
        // Normalize by search radius
        // 按搜索半径归一化
        return Math.min(1.0, motionError / this.params.expandedSearchRadius);
    }
    
    /**
     * Calculate size consistency cost
     * 计算尺寸一致性成本
     */
    calculateSizeCost(track, detection, bankEntry) {
        const avgSize = bankEntry.trackInfo.avgSize;
        
        const widthRatio = Math.min(detection.w / avgSize.w, avgSize.w / detection.w);
        const heightRatio = Math.min(detection.h / avgSize.h, avgSize.h / detection.h);
        
        const sizeConsistency = widthRatio * heightRatio;
        
        return 1 - sizeConsistency;
    }
    
    /**
     * Calculate temporal cost (penalty for longer occlusion)
     * 计算时间成本（对较长遮挡的惩罚）
     */
    calculateTemporalCost(track, frameCount) {
        const occlusionDuration = track.lostFrames;
        const maxDuration = this.params.maxFramesForReidentification;
        
        // Linear penalty for longer occlusion
        // 对较长遮挡的线性惩罚
        return Math.min(1.0, occlusionDuration / maxDuration);
    }
    
    /**
     * Solve re-identification assignment using Hungarian algorithm
     * 使用匈牙利算法解决重识别分配
     */
    solveReidentificationAssignment(costMatrix) {
        if (costMatrix.length === 0 || costMatrix[0].length === 0) {
            return [];
        }
        
        // Use a simple greedy approach for now (can be replaced with Hungarian)
        // 目前使用简单的贪心方法（可以用匈牙利算法替换）
        const assignments = [];
        const usedTracks = new Set();
        const usedDetections = new Set();
        
        // Create list of all possible assignments with costs
        // 创建所有可能分配及其成本的列表
        const candidates = [];
        for (let i = 0; i < costMatrix.length; i++) {
            for (let j = 0; j < costMatrix[i].length; j++) {
                if (costMatrix[i][j] < 999) {
                    candidates.push({
                        trackIdx: i,
                        detectionIdx: j,
                        cost: costMatrix[i][j]
                    });
                }
            }
        }
        
        // Sort by cost (ascending)
        // 按成本排序（升序）
        candidates.sort((a, b) => a.cost - b.cost);
        
        // Greedily assign best matches
        // 贪心分配最佳匹配
        for (const candidate of candidates) {
            if (!usedTracks.has(candidate.trackIdx) && 
                !usedDetections.has(candidate.detectionIdx)) {
                
                assignments.push(candidate);
                usedTracks.add(candidate.trackIdx);
                usedDetections.add(candidate.detectionIdx);
            }
        }
        
        return assignments;
    }
    
    /**
     * Log successful re-identification for analysis
     * 记录成功的重识别以供分析
     */
    logReidentification(track, detection, cost, frameCount) {
        const reidentificationRecord = {
            trackId: track.id,
            frameCount: frameCount,
            occlusionDuration: track.lostFrames,
            similarity: 1 - cost,
            detectionClass: detection.class,
            timestamp: Date.now()
        };
        
        this.reidentificationHistory.push(reidentificationRecord);
        
        // Keep history limited
        // 限制历史记录
        if (this.reidentificationHistory.length > 100) {
            this.reidentificationHistory = this.reidentificationHistory.slice(-100);
        }
        
        console.log(`✓ Re-identified track ${track.id} after ${track.lostFrames} frames (similarity: ${(1-cost).toFixed(3)})`);
    }
    
    /**
     * Update performance metrics
     * 更新性能指标
     */
    updatePerformanceMetrics(processingTime, successCount) {
        const alpha = 0.1; // Smoothing factor
        this.metrics.avgReidentificationTime = 
            alpha * processingTime + (1 - alpha) * this.metrics.avgReidentificationTime;
        
        if (successCount === 0) {
            this.metrics.failedAttempts++;
        }
    }
    
    /**
     * Clean up old feature bank entries
     * 清理旧的特征库条目
     */
    cleanupFeatureBank(currentFrame, maxAge = 3000) { // 100 seconds at 30fps
        for (const [trackId, bankEntry] of this.featureBank.entries()) {
            if (currentFrame - bankEntry.lastUpdated > maxAge) {
                this.featureBank.delete(trackId);
                console.log(`Cleaned up feature bank entry for track ${trackId}`);
            }
        }
    }
    
    /**
     * Get re-identification statistics
     * 获取重识别统计信息
     */
    getReidentificationStats() {
        const totalAttempts = this.metrics.successfulReidentifications + this.metrics.failedAttempts;
        
        return {
            successfulReidentifications: this.metrics.successfulReidentifications,
            failedAttempts: this.metrics.failedAttempts,
            successRate: totalAttempts > 0 ? 
                this.metrics.successfulReidentifications / totalAttempts : 0,
            avgProcessingTime: this.metrics.avgReidentificationTime,
            featureBankSize: this.featureBank.size,
            recentReidentifications: this.reidentificationHistory.slice(-10)
        };
    }
    
    /**
     * Deep copy features object
     * 深度复制特征对象
     */
    deepCopyFeatures(features) {
        if (!features) return null;
        
        const copy = {};
        for (const [key, value] of Object.entries(features)) {
            if (Array.isArray(value)) {
                copy[key] = [...value];
            } else if (typeof value === 'object' && value !== null) {
                copy[key] = { ...value };
            } else {
                copy[key] = value;
            }
        }
        return copy;
    }
}

// Export for use in other modules
// 导出供其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ReidentificationManager;
} else if (typeof window !== 'undefined') {
    window.ReidentificationManager = ReidentificationManager;
}
