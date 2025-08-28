/**
 * 密集人群场景优化模块
 * Crowd Scene Optimization Module
 * 
 * 专门针对密集人群场景的追踪优化策略
 */

class CrowdOptimizer {
    constructor(options = {}) {
        this.densityThreshold = options.densityThreshold || 0.3; // 密度阈值
        this.overlapThreshold = options.overlapThreshold || 0.4; // 重叠阈值
        this.minTrackSize = options.minTrackSize || 20; // 最小追踪尺寸
        this.maxTrackSize = options.maxTrackSize || 200; // 最大追踪尺寸
        
        // 密集场景检测状态
        this.crowdDensity = 0;
        this.isCrowdedScene = false;
        this.adaptiveThresholds = {
            matching: 0.7,
            creation: 0.8,
            deletion: 0.3
        };
    }

    /**
     * 分析场景密度并调整策略
     */
    analyzeCrowdDensity(detections, canvasWidth, canvasHeight) {
        if (!detections || detections.length === 0) {
            this.crowdDensity = 0;
            this.isCrowdedScene = false;
            return;
        }

        // 计算检测密度
        const totalArea = canvasWidth * canvasHeight;
        const detectionArea = detections.reduce((sum, det) => sum + (det.w * det.h), 0);
        this.crowdDensity = detectionArea / totalArea;

        // 计算重叠度
        const overlapScore = this.calculateOverlapScore(detections);
        
        // 综合判断是否为密集场景
        this.isCrowdedScene = (this.crowdDensity > this.densityThreshold) || 
                             (overlapScore > this.overlapThreshold);

        // 动态调整阈值
        this.updateAdaptiveThresholds();
    }

    /**
     * 计算检测框重叠评分
     */
    calculateOverlapScore(detections) {
        if (detections.length < 2) return 0;

        let totalOverlap = 0;
        let pairCount = 0;

        for (let i = 0; i < detections.length - 1; i++) {
            for (let j = i + 1; j < detections.length; j++) {
                const iou = this.calculateIoU(detections[i], detections[j]);
                totalOverlap += iou;
                pairCount++;
            }
        }

        return pairCount > 0 ? totalOverlap / pairCount : 0;
    }

    /**
     * 动态调整匹配阈值
     */
    updateAdaptiveThresholds() {
        if (this.isCrowdedScene) {
            // 密集场景：提高匹配要求，降低新建轨迹概率
            this.adaptiveThresholds.matching = 0.8;
            this.adaptiveThresholds.creation = 0.9;
            this.adaptiveThresholds.deletion = 0.2;
        } else {
            // 稀疏场景：放宽匹配要求
            this.adaptiveThresholds.matching = 0.6;
            this.adaptiveThresholds.creation = 0.7;
            this.adaptiveThresholds.deletion = 0.4;
        }
    }

    /**
     * 密集场景下的智能匹配策略
     */
    optimizeMatching(tracks, detections) {
        if (!this.isCrowdedScene) {
            return { tracks, detections }; // 非密集场景直接返回
        }

        // 1. 过滤异常尺寸的检测
        const filteredDetections = detections.filter(det => 
            det.w >= this.minTrackSize && det.w <= this.maxTrackSize &&
            det.h >= this.minTrackSize && det.h <= this.maxTrackSize
        );

        // 2. 按置信度排序检测
        filteredDetections.sort((a, b) => (b.confidence || 0.5) - (a.confidence || 0.5));

        // 3. 优先处理稳定轨迹
        const stableTracks = tracks.filter(t => t.hits > 5);
        const newTracks = tracks.filter(t => t.hits <= 5);

        return {
            tracks: [...stableTracks, ...newTracks],
            detections: filteredDetections
        };
    }

    /**
     * 密集场景下的冲突解决
     */
    resolveConflicts(matches) {
        if (!this.isCrowdedScene) return matches;

        // 检测一对多匹配冲突
        const trackUsage = new Map();
        const detectionUsage = new Map();

        matches.forEach((match, idx) => {
            const trackId = match.track.id;
            const detIdx = match.detectionIdx;

            if (!trackUsage.has(trackId)) trackUsage.set(trackId, []);
            if (!detectionUsage.has(detIdx)) detectionUsage.set(detIdx, []);

            trackUsage.get(trackId).push({match, idx});
            detectionUsage.get(detIdx).push({match, idx});
        });

        const resolvedMatches = [];
        const usedTracks = new Set();
        const usedDetections = new Set();

        // 优先处理单一匹配
        matches.forEach((match, idx) => {
            const trackId = match.track.id;
            const detIdx = match.detectionIdx;

            if (trackUsage.get(trackId).length === 1 && 
                detectionUsage.get(detIdx).length === 1 &&
                !usedTracks.has(trackId) && 
                !usedDetections.has(detIdx)) {
                
                resolvedMatches.push(match);
                usedTracks.add(trackId);
                usedDetections.add(detIdx);
            }
        });

        // 处理冲突匹配 - 选择最高评分
        for (const [trackId, matchList] of trackUsage) {
            if (usedTracks.has(trackId) || matchList.length === 1) continue;

            const bestMatch = matchList.reduce((best, current) => 
                current.match.score > best.match.score ? current : best
            );

            if (!usedDetections.has(bestMatch.match.detectionIdx)) {
                resolvedMatches.push(bestMatch.match);
                usedTracks.add(trackId);
                usedDetections.add(bestMatch.match.detectionIdx);
            }
        }

        return resolvedMatches;
    }

    /**
     * 智能轨迹创建策略
     */
    shouldCreateTrack(detection, existingTracks) {
        if (!this.isCrowdedScene) return true;

        // 密集场景下更严格的创建条件
        
        // 1. 检查是否与现有轨迹过于接近
        const minDistance = 40;
        for (const track of existingTracks) {
            const dx = (detection.x + detection.w/2) - track.x;
            const dy = (detection.y + detection.h/2) - track.y;
            const distance = Math.sqrt(dx*dx + dy*dy);
            
            if (distance < minDistance) {
                return false; // 太接近现有轨迹
            }
        }

        // 2. 检查检测质量
        const confidence = detection.confidence || 0.5;
        if (confidence < this.adaptiveThresholds.creation) {
            return false; // 置信度不足
        }

        // 3. 检查尺寸合理性
        if (detection.w < this.minTrackSize || detection.w > this.maxTrackSize ||
            detection.h < this.minTrackSize || detection.h > this.maxTrackSize) {
            return false; // 尺寸异常
        }

        return true;
    }

    /**
     * 遮挡恢复优化
     */
    optimizeOcclusionRecovery(lostTracks, newDetections) {
        const recoveredMatches = [];

        for (const track of lostTracks) {
            if (track.missedFrames > 10) continue; // 丢失太久

            // 在预测位置附近搜索
            const searchRadius = Math.min(100, 30 + track.missedFrames * 5);
            
            let bestCandidate = null;
            let bestScore = 0.5;

            for (const detection of newDetections) {
                const dx = (detection.x + detection.w/2) - track.predictedX;
                const dy = (detection.y + detection.h/2) - track.predictedY;
                const distance = Math.sqrt(dx*dx + dy*dy);

                if (distance > searchRadius) continue;

                // 计算恢复评分
                const distanceScore = 1 - (distance / searchRadius);
                const sizeScore = Math.min(track.w/detection.w, detection.w/track.w) *
                                 Math.min(track.h/detection.h, detection.h/track.h);
                
                const score = distanceScore * 0.6 + sizeScore * 0.4;

                if (score > bestScore) {
                    bestScore = score;
                    bestCandidate = detection;
                }
            }

            if (bestCandidate) {
                recoveredMatches.push({
                    track: track,
                    detection: bestCandidate,
                    score: bestScore,
                    type: 'recovery'
                });
            }
        }

        return recoveredMatches;
    }

    /**
     * 获取当前优化状态
     */
    getOptimizationStatus() {
        return {
            crowdDensity: this.crowdDensity,
            isCrowdedScene: this.isCrowdedScene,
            adaptiveThresholds: {...this.adaptiveThresholds}
        };
    }

    /**
     * 工具函数：计算IoU
     */
    calculateIoU(box1, box2) {
        const x1 = Math.max(box1.x, box2.x);
        const y1 = Math.max(box1.y, box2.y);
        const x2 = Math.min(box1.x + box1.w, box2.x + box2.w);
        const y2 = Math.min(box1.y + box1.h, box2.y + box2.h);
        
        if (x2 <= x1 || y2 <= y1) return 0;
        
        const intersection = (x2 - x1) * (y2 - y1);
        const union = box1.w * box1.h + box2.w * box2.h - intersection;
        
        return intersection / union;
    }
}

// 导出密集人群优化器
window.CrowdOptimizer = CrowdOptimizer;
