/**
 * Occlusion Handling Manager
 * 遮挡处理管理器
 * 
 * Handles short-term and long-term occlusion scenarios
 * 处理短期和长期遮挡场景
 */
class OcclusionManager {
    constructor(options = {}) {
        // Occlusion state thresholds
        // 遮挡状态阈值
        this.thresholds = {
            shortTermLimit: options.shortTermLimit || 30,    // Frames for short-term occlusion
            longTermLimit: options.longTermLimit || 60,      // Frames before track deletion
            confidenceDecay: options.confidenceDecay || 0.95, // Per-frame confidence decay
            minConfidence: options.minConfidence || 0.1      // Minimum confidence to maintain track
        };
        
        // Search parameters for re-identification
        // 重识别搜索参数
        this.searchParams = {
            shortTermRadius: options.shortTermRadius || 100,  // Search radius for short-term
            longTermRadius: options.longTermRadius || 200,    // Search radius for long-term
            appearanceWeight: options.appearanceWeight || 0.6, // Weight for appearance in re-ID
            motionWeight: options.motionWeight || 0.4         // Weight for motion prediction
        };
        
        // Prediction parameters
        // 预测参数
        this.predictionParams = {
            velocityDamping: options.velocityDamping || 0.95,  // Velocity damping during occlusion
            positionUncertainty: options.positionUncertainty || 1.1, // Uncertainty growth rate
            maxUncertainty: options.maxUncertainty || 500     // Maximum position uncertainty
        };
    }
    
    /**
     * Handle occlusion state for a track
     * 处理轨迹的遮挡状态
     */
    handleOcclusion(track, frameCount) {
        if (!track.occlusionState) {
            this.initializeOcclusionState(track);
        }
        
        const missedFrames = track.lostFrames;
        
        if (missedFrames === 0) {
            // Track is visible, reset occlusion state
            // 轨迹可见，重置遮挡状态
            this.resetOcclusionState(track);
            return { state: 'visible', action: 'none' };
        }
        
        if (missedFrames < this.thresholds.shortTermLimit) {
            // Short-term occlusion handling
            // 短期遮挡处理
            return this.handleShortTermOcclusion(track, frameCount);
        } else if (missedFrames < this.thresholds.longTermLimit) {
            // Long-term occlusion handling
            // 长期遮挡处理
            return this.handleLongTermOcclusion(track, frameCount);
        } else {
            // Track should be deleted
            // 轨迹应被删除
            return { state: 'lost', action: 'delete' };
        }
    }
    
    /**
     * Initialize occlusion state for a track
     * 为轨迹初始化遮挡状态
     */
    initializeOcclusionState(track) {
        track.occlusionState = {
            isOccluded: false,
            occlusionStartFrame: -1,
            lastVisiblePosition: { x: track.cx, y: track.cy },
            predictedPosition: { x: track.cx, y: track.cy },
            confidence: 1.0,
            searchRadius: Math.max(track.w, track.h) * 0.5,
            velocityAtOcclusion: { vx: track.vx || 0, vy: track.vy || 0 },
            positionUncertainty: 10.0
        };
    }
    
    /**
     * Reset occlusion state when track becomes visible
     * 当轨迹变为可见时重置遮挡状态
     */
    resetOcclusionState(track) {
        if (track.occlusionState) {
            track.occlusionState.isOccluded = false;
            track.occlusionState.occlusionStartFrame = -1;
            track.occlusionState.confidence = Math.min(1.0, track.occlusionState.confidence + 0.2);
            track.occlusionState.searchRadius = Math.max(track.w, track.h) * 0.5;
            track.occlusionState.positionUncertainty = 10.0;
        }
    }
    
    /**
     * Handle short-term occlusion (missedFrames < 30)
     * 处理短期遮挡 (missedFrames < 30)
     */
    handleShortTermOcclusion(track, frameCount) {
        const occState = track.occlusionState;
        
        if (!occState.isOccluded) {
            // Just entered occlusion state
            // 刚进入遮挡状态
            occState.isOccluded = true;
            occState.occlusionStartFrame = frameCount;
            occState.lastVisiblePosition = { x: track.cx, y: track.cy };
            occState.velocityAtOcclusion = { vx: track.vx || 0, vy: track.vy || 0 };
            
            console.log(`Track ${track.id} entered short-term occlusion at frame ${frameCount}`);
        }
        
        // Continue prediction using Kalman filter
        // 使用卡尔曼滤波器继续预测
        if (track.kalmanFilter) {
            const prediction = track.kalmanFilter.predict();
            track.cx = prediction.x;
            track.cy = prediction.y;
            track.vx = prediction.vx;
            track.vy = prediction.vy;
        } else {
            // Fallback to simple motion model
            // 回退到简单运动模型
            this.updateWithSimpleMotion(track);
        }
        
        // Update occlusion state
        // 更新遮挡状态
        this.updateOcclusionState(track);
        
        return {
            state: 'short_term_occluded',
            action: 'predict',
            searchRadius: occState.searchRadius,
            confidence: occState.confidence
        };
    }
    
    /**
     * Handle long-term occlusion (30 <= missedFrames < 60)
     * 处理长期遮挡 (30 <= missedFrames < 60)
     */
    handleLongTermOcclusion(track, frameCount) {
        const occState = track.occlusionState;
        
        // Expand search radius for re-identification
        // 扩大搜索半径用于重识别
        occState.searchRadius = Math.min(
            this.searchParams.longTermRadius,
            occState.searchRadius * 1.05 // Gradual expansion
        );
        
        // Continue prediction with higher uncertainty
        // 以更高不确定性继续预测
        if (track.kalmanFilter) {
            const prediction = track.kalmanFilter.predict();
            track.cx = prediction.x;
            track.cy = prediction.y;
            track.vx = prediction.vx * this.predictionParams.velocityDamping;
            track.vy = prediction.vy * this.predictionParams.velocityDamping;
        } else {
            this.updateWithSimpleMotion(track);
        }
        
        // Update occlusion state with faster decay
        // 以更快衰减更新遮挡状态
        this.updateOcclusionState(track, true);
        
        return {
            state: 'long_term_occluded',
            action: 'search_reidentify',
            searchRadius: occState.searchRadius,
            confidence: occState.confidence,
            requiresReidentification: true
        };
    }
    
    /**
     * Update occlusion state parameters
     * 更新遮挡状态参数
     */
    updateOcclusionState(track, isLongTerm = false) {
        const occState = track.occlusionState;
        
        // Decay confidence
        // 衰减置信度
        const decayRate = isLongTerm ? 
            this.thresholds.confidenceDecay * 0.98 : // Faster decay for long-term
            this.thresholds.confidenceDecay;
        
        occState.confidence *= decayRate;
        occState.confidence = Math.max(occState.confidence, this.thresholds.minConfidence);
        
        // Increase position uncertainty
        // 增加位置不确定性
        occState.positionUncertainty *= this.predictionParams.positionUncertainty;
        occState.positionUncertainty = Math.min(
            occState.positionUncertainty, 
            this.predictionParams.maxUncertainty
        );
        
        // Update predicted position
        // 更新预测位置
        occState.predictedPosition = { x: track.cx, y: track.cy };
        
        // Gradually expand search radius based on velocity and uncertainty
        // 根据速度和不确定性逐渐扩大搜索半径
        const velocityMagnitude = Math.sqrt(track.vx**2 + track.vy**2);
        const velocityBonus = Math.min(50, velocityMagnitude * 2);
        const uncertaintyBonus = Math.min(100, occState.positionUncertainty * 0.2);
        
        const baseRadius = Math.max(track.w, track.h) * 0.8;
        occState.searchRadius = baseRadius + velocityBonus + uncertaintyBonus;
    }
    
    /**
     * Simple motion model fallback when Kalman filter is not available
     * 当卡尔曼滤波器不可用时的简单运动模型回退
     */
    updateWithSimpleMotion(track) {
        const dt = 1/30; // Assume 30 FPS
        
        // Apply velocity damping during occlusion
        // 在遮挡期间应用速度阻尼
        track.vx = (track.vx || 0) * this.predictionParams.velocityDamping;
        track.vy = (track.vy || 0) * this.predictionParams.velocityDamping;
        
        // Update position
        // 更新位置
        track.cx += track.vx * dt;
        track.cy += track.vy * dt;
        
        // Add some noise to account for uncertainty
        // 添加一些噪声以考虑不确定性
        const noise = track.occlusionState.positionUncertainty * 0.01;
        track.cx += (Math.random() - 0.5) * noise;
        track.cy += (Math.random() - 0.5) * noise;
    }
    
    /**
     * Attempt to re-identify occluded track with new detections
     * 尝试使用新检测重识别遮挡的轨迹
     */
    attemptReidentification(track, detections, appearanceExtractor, ctx) {
        if (!track.occlusionState || !track.occlusionState.isOccluded) {
            return null;
        }
        
        const occState = track.occlusionState;
        const candidates = [];
        
        // Find detections within search radius
        // 找到搜索半径内的检测
        for (let i = 0; i < detections.length; i++) {
            const detection = detections[i];
            
            // Check class consistency
            // 检查类别一致性
            if (track.class && detection.class && track.class !== detection.class) {
                continue;
            }
            
            // Check distance constraint
            // 检查距离约束
            const detectionCenter = {
                x: detection.x + detection.w / 2,
                y: detection.y + detection.h / 2
            };
            
            const distance = Math.sqrt(
                (detectionCenter.x - track.cx)**2 + 
                (detectionCenter.y - track.cy)**2
            );
            
            if (distance > occState.searchRadius) {
                continue;
            }
            
            // Calculate re-identification score
            // 计算重识别分数
            const score = this.calculateReidentificationScore(
                track, detection, distance, appearanceExtractor, ctx
            );
            
            candidates.push({
                detectionIndex: i,
                detection: detection,
                score: score,
                distance: distance
            });
        }
        
        // Sort candidates by score (higher is better)
        // 按分数排序候选者（越高越好）
        candidates.sort((a, b) => b.score - a.score);
        
        // Return best candidate if score is above threshold
        // 如果分数超过阈值则返回最佳候选者
        const minReidentificationScore = track.lostFrames >= this.thresholds.shortTermLimit ? 0.6 : 0.7;
        
        if (candidates.length > 0 && candidates[0].score >= minReidentificationScore) {
            return candidates[0];
        }
        
        return null;
    }
    
    /**
     * Calculate re-identification score for track-detection pair
     * 计算轨迹-检测对的重识别分数
     */
    calculateReidentificationScore(track, detection, distance, appearanceExtractor, ctx) {
        let appearanceScore = 0.5; // Default neutral score
        let motionScore = 0.5;
        
        // Appearance similarity
        // 外观相似度
        if (appearanceExtractor && ctx && track.appearanceFeatures) {
            try {
                const detectionFeatures = appearanceExtractor.extractFeatures(
                    ctx, detection, detection.class
                );
                
                if (detectionFeatures) {
                    appearanceScore = appearanceExtractor.calculateWeightedSimilarity(
                        track.appearanceFeatures, detectionFeatures, detection.class
                    );
                }
            } catch (error) {
                console.warn('Appearance feature extraction failed during re-ID:', error);
            }
        }
        
        // Motion consistency score
        // 运动一致性分数
        const maxReasonableDistance = track.occlusionState.searchRadius;
        motionScore = Math.max(0, 1 - (distance / maxReasonableDistance));
        
        // Size consistency score
        // 尺寸一致性分数
        const sizeRatio = Math.min(detection.w / track.w, track.w / detection.w) *
                         Math.min(detection.h / track.h, track.h / detection.h);
        const sizeScore = Math.max(0, sizeRatio);
        
        // Weighted combination
        // 加权组合
        const finalScore = 
            this.searchParams.appearanceWeight * appearanceScore +
            this.searchParams.motionWeight * motionScore +
            0.2 * sizeScore; // Size weight
        
        return Math.min(1.0, finalScore);
    }
    
    /**
     * Reset Kalman filter after successful re-identification
     * 成功重识别后重置卡尔曼滤波器
     */
    resetKalmanAfterReidentification(track, detection) {
        if (track.kalmanFilter) {
            // Re-initialize with new detection
            // 使用新检测重新初始化
            const centerX = detection.x + detection.w / 2;
            const centerY = detection.y + detection.h / 2;
            
            track.kalmanFilter.initialize(centerX, centerY, detection.confidence || 0.8);
            
            console.log(`Reset Kalman filter for track ${track.id} after re-identification`);
        }
    }
    
    /**
     * Get occlusion statistics for debugging
     * 获取遮挡统计信息用于调试
     */
    getOcclusionStats(tracks) {
        const stats = {
            total: tracks.length,
            visible: 0,
            shortTermOccluded: 0,
            longTermOccluded: 0,
            avgConfidence: 0,
            avgSearchRadius: 0
        };
        
        let totalConfidence = 0;
        let totalSearchRadius = 0;
        let occludedCount = 0;
        
        for (const track of tracks) {
            if (track.lostFrames === 0) {
                stats.visible++;
            } else if (track.lostFrames < this.thresholds.shortTermLimit) {
                stats.shortTermOccluded++;
            } else {
                stats.longTermOccluded++;
            }
            
            if (track.occlusionState) {
                totalConfidence += track.occlusionState.confidence;
                totalSearchRadius += track.occlusionState.searchRadius;
                occludedCount++;
            }
        }
        
        if (occludedCount > 0) {
            stats.avgConfidence = totalConfidence / occludedCount;
            stats.avgSearchRadius = totalSearchRadius / occludedCount;
        }
        
        return stats;
    }
}

// Export for use in other modules
// 导出供其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = OcclusionManager;
} else if (typeof window !== 'undefined') {
    window.OcclusionManager = OcclusionManager;
}
