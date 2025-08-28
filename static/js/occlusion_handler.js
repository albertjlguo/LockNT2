/**
 * 轻量级遮挡处理模块
 * Lightweight Occlusion Handling Module
 * 
 * 专门处理目标遮挡和恢复的轻量级解决方案
 */

class OcclusionHandler {
    constructor(options = {}) {
        this.maxOcclusionFrames = options.maxOcclusionFrames || 20;
        this.predictionDecay = options.predictionDecay || 0.95;
        this.recoveryThreshold = options.recoveryThreshold || 0.6;
        this.searchExpansion = options.searchExpansion || 1.5;
        
        // 遮挡状态管理
        this.occludedTracks = new Map();
        this.recoveryAttempts = new Map();
    }

    /**
     * 检测并处理遮挡状态
     */
    handleOcclusion(tracks, detections) {
        const activeTrackIds = new Set();
        const occlusionUpdates = [];

        // 标记活跃轨迹
        tracks.forEach(track => {
            if (track.missedFrames === 0) {
                activeTrackIds.add(track.id);
                
                // 如果轨迹从遮挡中恢复
                if (this.occludedTracks.has(track.id)) {
                    occlusionUpdates.push({
                        type: 'recovered',
                        trackId: track.id,
                        track: track
                    });
                    this.occludedTracks.delete(track.id);
                    this.recoveryAttempts.delete(track.id);
                }
            }
        });

        // 检测新的遮挡
        tracks.forEach(track => {
            if (track.missedFrames >= 3 && !this.occludedTracks.has(track.id)) {
                this.startOcclusion(track);
                occlusionUpdates.push({
                    type: 'occluded',
                    trackId: track.id,
                    track: track
                });
            }
        });

        // 更新遮挡轨迹的预测
        this.updateOccludedPredictions();

        // 尝试恢复遮挡轨迹
        const recoveryMatches = this.attemptRecovery(detections);

        return {
            occlusionUpdates,
            recoveryMatches,
            occludedCount: this.occludedTracks.size
        };
    }

    /**
     * 开始遮挡处理
     */
    startOcclusion(track) {
        const occlusionState = {
            startFrame: track.frameCount || 0,
            lastKnownPosition: { x: track.x, y: track.y },
            lastKnownSize: { w: track.w, h: track.h },
            velocity: { vx: track.vx || 0, vy: track.vy || 0 },
            confidence: 1.0,
            searchRadius: Math.max(track.w, track.h) * 0.8,
            predictedPosition: { x: track.x, y: track.y }
        };

        this.occludedTracks.set(track.id, occlusionState);
        this.recoveryAttempts.set(track.id, 0);
    }

    /**
     * 更新遮挡轨迹的预测位置
     */
    updateOccludedPredictions() {
        for (const [trackId, state] of this.occludedTracks.entries()) {
            // 简单的线性预测
            state.predictedPosition.x += state.velocity.vx;
            state.predictedPosition.y += state.velocity.vy;
            
            // 速度衰减
            state.velocity.vx *= this.predictionDecay;
            state.velocity.vy *= this.predictionDecay;
            
            // 置信度衰减
            state.confidence *= 0.98;
            
            // 搜索半径扩展
            state.searchRadius = Math.min(state.searchRadius * 1.02, 150);
        }
    }

    /**
     * 尝试恢复遮挡轨迹
     */
    attemptRecovery(detections) {
        const recoveryMatches = [];
        const usedDetections = new Set();

        for (const [trackId, occlusionState] of this.occludedTracks.entries()) {
            if (occlusionState.confidence < 0.3) continue; // 置信度太低

            let bestMatch = null;
            let bestScore = this.recoveryThreshold;

            for (let i = 0; i < detections.length; i++) {
                if (usedDetections.has(i)) continue;

                const detection = detections[i];
                const score = this.calculateRecoveryScore(occlusionState, detection);

                if (score > bestScore) {
                    bestScore = score;
                    bestMatch = {
                        trackId,
                        detectionIdx: i,
                        detection,
                        score,
                        occlusionState
                    };
                }
            }

            if (bestMatch) {
                recoveryMatches.push(bestMatch);
                usedDetections.add(bestMatch.detectionIdx);
                
                // 增加恢复尝试计数
                const attempts = this.recoveryAttempts.get(trackId) + 1;
                this.recoveryAttempts.set(trackId, attempts);
            }
        }

        return recoveryMatches;
    }

    /**
     * 计算恢复匹配评分
     */
    calculateRecoveryScore(occlusionState, detection) {
        const detCenterX = detection.x + detection.w / 2;
        const detCenterY = detection.y + detection.h / 2;
        
        // 距离评分
        const dx = detCenterX - occlusionState.predictedPosition.x;
        const dy = detCenterY - occlusionState.predictedPosition.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance > occlusionState.searchRadius) return 0;
        
        const distanceScore = 1 - (distance / occlusionState.searchRadius);
        
        // 尺寸一致性评分
        const sizeScore = Math.min(
            occlusionState.lastKnownSize.w / detection.w,
            detection.w / occlusionState.lastKnownSize.w
        ) * Math.min(
            occlusionState.lastKnownSize.h / detection.h,
            detection.h / occlusionState.lastKnownSize.h
        );
        
        // 运动一致性评分
        const expectedVx = occlusionState.velocity.vx;
        const expectedVy = occlusionState.velocity.vy;
        const actualVx = detCenterX - occlusionState.lastKnownPosition.x;
        const actualVy = detCenterY - occlusionState.lastKnownPosition.y;
        
        const motionConsistency = Math.max(0, 1 - Math.abs(expectedVx - actualVx) / 50 - Math.abs(expectedVy - actualVy) / 50);
        
        // 置信度权重
        const confidenceWeight = occlusionState.confidence;
        
        // 综合评分
        return (distanceScore * 0.4 + sizeScore * 0.3 + motionConsistency * 0.3) * confidenceWeight;
    }

    /**
     * 清理长期遮挡的轨迹
     */
    cleanupLongTermOcclusion() {
        const toRemove = [];
        
        for (const [trackId, state] of this.occludedTracks.entries()) {
            const attempts = this.recoveryAttempts.get(trackId) || 0;
            
            // 清理条件：置信度过低或尝试次数过多
            if (state.confidence < 0.1 || attempts > this.maxOcclusionFrames) {
                toRemove.push(trackId);
            }
        }
        
        toRemove.forEach(trackId => {
            this.occludedTracks.delete(trackId);
            this.recoveryAttempts.delete(trackId);
        });
        
        return toRemove;
    }

    /**
     * 获取遮挡状态信息
     */
    getOcclusionStatus() {
        const status = {
            totalOccluded: this.occludedTracks.size,
            occludedTracks: []
        };

        for (const [trackId, state] of this.occludedTracks.entries()) {
            status.occludedTracks.push({
                trackId,
                confidence: state.confidence,
                searchRadius: state.searchRadius,
                predictedPosition: {...state.predictedPosition},
                attempts: this.recoveryAttempts.get(trackId) || 0
            });
        }

        return status;
    }

    /**
     * 重置遮挡处理器
     */
    reset() {
        this.occludedTracks.clear();
        this.recoveryAttempts.clear();
    }

    /**
     * 检查轨迹是否被遮挡
     */
    isTrackOccluded(trackId) {
        return this.occludedTracks.has(trackId);
    }

    /**
     * 获取遮挡轨迹的预测位置
     */
    getOccludedTrackPrediction(trackId) {
        const state = this.occludedTracks.get(trackId);
        return state ? state.predictedPosition : null;
    }

    /**
     * 强制结束轨迹遮挡状态
     */
    forceEndOcclusion(trackId) {
        this.occludedTracks.delete(trackId);
        this.recoveryAttempts.delete(trackId);
    }
}

// 导出遮挡处理器
window.OcclusionHandler = OcclusionHandler;
