/**
 * 轻量级高精度目标追踪器 - 专为密集人群场景优化
 * Lightweight High-Precision Object Tracker - Optimized for Dense Crowd Scenarios
 * 
 * 核心设计理念：
 * 1. 最小化计算开销 - 只保留最关键的特征
 * 2. 智能匹配策略 - 分层处理，优先级排序
 * 3. 密集场景优化 - 空间分区，局部搜索
 * 4. 鲁棒遮挡处理 - 轻量级预测恢复
 */

class LightweightTracker {
    constructor(options = {}) {
        // 核心配置 - 保持最小化
        this.maxTracks = options.maxTracks || 10;
        this.searchRadius = options.searchRadius || 80;
        this.confidenceThreshold = options.confidenceThreshold || 0.6;
        this.maxMissedFrames = options.maxMissedFrames || 15;
        
        // 轻量级状态管理
        this.tracks = new Map();
        this.nextId = 1;
        this.frameCount = 0;
        
        // 性能优化配置
        this.spatialGrid = new SpatialGrid(8, 6); // 8x6网格分区
        this.featureCache = new LRUCache(100); // 特征缓存
    }

    /**
     * 主更新函数 - 核心追踪逻辑
     */
    update(detections, canvas) {
        this.frameCount++;
        
        // 1. 预测所有轨迹的新位置
        this.predictTracks();
        
        // 2. 使用空间分区优化匹配
        const matches = this.spatialMatching(detections);
        
        // 3. 更新匹配的轨迹
        this.updateMatches(matches, canvas);
        
        // 4. 处理未匹配的检测（创建新轨迹）
        this.handleUnmatched(detections, matches);
        
        // 5. 清理失效轨迹
        this.pruneTracks();
        
        return Array.from(this.tracks.values());
    }

    /**
     * 轻量级运动预测 - 仅使用位置和速度
     */
    predictTracks() {
        for (const track of this.tracks.values()) {
            // 简单的线性预测，避免复杂的加速度计算
            track.predictedX = track.x + track.vx;
            track.predictedY = track.y + track.vy;
            
            // 速度衰减，防止发散
            track.vx *= 0.9;
            track.vy *= 0.9;
            
            track.missedFrames++;
        }
    }

    /**
     * 空间分区匹配 - 核心优化算法
     * 将检测和轨迹按空间位置分组，只在相邻区域内进行匹配
     */
    spatialMatching(detections) {
        const matches = [];
        const usedDetections = new Set();
        const usedTracks = new Set();
        
        // 将检测按空间位置分组
        this.spatialGrid.clear();
        detections.forEach((det, idx) => {
            this.spatialGrid.add(det.x + det.w/2, det.y + det.h/2, {det, idx});
        });
        
        // 按距离优先级排序轨迹
        const sortedTracks = Array.from(this.tracks.values())
            .sort((a, b) => a.missedFrames - b.missedFrames);
        
        // 为每个轨迹在其预测位置附近搜索最佳匹配
        for (const track of sortedTracks) {
            if (usedTracks.has(track.id)) continue;
            
            const candidates = this.spatialGrid.query(
                track.predictedX, track.predictedY, this.searchRadius
            );
            
            let bestMatch = null;
            let bestScore = this.confidenceThreshold;
            
            for (const {det, idx} of candidates) {
                if (usedDetections.has(idx)) continue;
                
                const score = this.calculateMatchScore(track, det);
                if (score > bestScore) {
                    bestScore = score;
                    bestMatch = {track, detection: det, detectionIdx: idx, score};
                }
            }
            
            if (bestMatch) {
                matches.push(bestMatch);
                usedTracks.add(track.id);
                usedDetections.add(bestMatch.detectionIdx);
            }
        }
        
        return {matches, usedDetections};
    }

    /**
     * 轻量级匹配评分 - 只使用最关键的特征
     */
    calculateMatchScore(track, detection) {
        const dx = track.predictedX - (detection.x + detection.w/2);
        const dy = track.predictedY - (detection.y + detection.h/2);
        const distance = Math.sqrt(dx*dx + dy*dy);
        
        // 距离评分 (0-1)
        const distanceScore = Math.max(0, 1 - distance / this.searchRadius);
        
        // 尺寸一致性评分 (0-1)
        const sizeRatio = Math.min(track.w/detection.w, detection.w/track.w) *
                         Math.min(track.h/detection.h, detection.h/track.h);
        
        // IoU评分 (仅在距离较近时计算，节省计算)
        let iouScore = 0;
        if (distance < this.searchRadius * 0.7) {
            iouScore = this.calculateIoU(
                {x: track.predictedX - track.w/2, y: track.predictedY - track.h/2, 
                 w: track.w, h: track.h},
                detection
            );
        }
        
        // 外观特征评分 (轻量级)
        const appearanceScore = track.feature ? 
            this.compareAppearance(track.feature, detection.feature || null) : 0.5;
        
        // 加权组合 - 针对密集场景优化权重
        return distanceScore * 0.4 + 
               sizeRatio * 0.2 + 
               iouScore * 0.2 + 
               appearanceScore * 0.2;
    }

    /**
     * 轻量级外观特征比较
     */
    compareAppearance(feature1, feature2) {
        if (!feature1 || !feature2) return 0.5;
        
        // 使用缓存的特征比较结果
        const cacheKey = `${feature1.id}_${feature2.id}`;
        if (this.featureCache.has(cacheKey)) {
            return this.featureCache.get(cacheKey);
        }
        
        // 简化的颜色直方图比较
        let similarity = 0;
        const minLen = Math.min(feature1.histogram.length, feature2.histogram.length);
        for (let i = 0; i < minLen; i++) {
            similarity += Math.min(feature1.histogram[i], feature2.histogram[i]);
        }
        
        this.featureCache.set(cacheKey, similarity);
        return similarity;
    }

    /**
     * 更新匹配的轨迹
     */
    updateMatches(matchResult, canvas) {
        for (const match of matchResult.matches) {
            const track = match.track;
            const det = match.detection;
            
            // 更新位置 (使用简单的加权平均)
            const alpha = 0.7; // 学习率
            track.x = track.x * (1 - alpha) + (det.x + det.w/2) * alpha;
            track.y = track.y * (1 - alpha) + (det.y + det.h/2) * alpha;
            track.w = track.w * (1 - alpha) + det.w * alpha;
            track.h = track.h * (1 - alpha) + det.h * alpha;
            
            // 更新速度
            track.vx = (det.x + det.w/2) - track.x;
            track.vy = (det.y + det.h/2) - track.y;
            
            // 重置丢失计数
            track.missedFrames = 0;
            track.hits++;
            
            // 更新轨迹历史 (限制长度)
            track.trajectory.push({x: track.x, y: track.y, frame: this.frameCount});
            if (track.trajectory.length > 30) {
                track.trajectory.shift();
            }
            
            // 更新外观特征 (轻量级)
            if (canvas && det.bbox) {
                track.feature = this.extractLightweightFeature(canvas, det.bbox);
            }
        }
    }

    /**
     * 处理未匹配的检测 - 创建新轨迹
     */
    handleUnmatched(detections, matchResult) {
        if (this.tracks.size >= this.maxTracks) return;
        
        for (let i = 0; i < detections.length; i++) {
            if (matchResult.usedDetections.has(i)) continue;
            
            const det = detections[i];
            
            // 避免在已有轨迹附近创建新轨迹 (防止重复)
            if (this.isNearExistingTrack(det)) continue;
            
            this.createTrack(det);
        }
    }

    /**
     * 创建新轨迹
     */
    createTrack(detection) {
        const track = {
            id: this.nextId++,
            x: detection.x + detection.w/2,
            y: detection.y + detection.h/2,
            w: detection.w,
            h: detection.h,
            vx: 0,
            vy: 0,
            predictedX: 0,
            predictedY: 0,
            missedFrames: 0,
            hits: 1,
            trajectory: [{x: detection.x + detection.w/2, y: detection.y + detection.h/2, frame: this.frameCount}],
            feature: null,
            color: this.getTrackColor(this.nextId - 1),
            locked: false
        };
        
        this.tracks.set(track.id, track);
    }

    /**
     * 清理失效轨迹
     */
    pruneTracks() {
        for (const [id, track] of this.tracks.entries()) {
            if (track.missedFrames > this.maxMissedFrames) {
                this.tracks.delete(id);
            }
        }
    }

    /**
     * 轻量级特征提取
     */
    extractLightweightFeature(canvas, bbox) {
        try {
            const ctx = canvas.getContext('2d');
            const imageData = ctx.getImageData(bbox.x, bbox.y, bbox.w, bbox.h);
            
            // 简化的颜色直方图 (16 bins)
            const histogram = new Array(16).fill(0);
            const data = imageData.data;
            
            for (let i = 0; i < data.length; i += 16) { // 采样降低计算量
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                const gray = Math.floor((r + g + b) / 3 / 16);
                histogram[Math.min(15, gray)]++;
            }
            
            // 归一化
            const total = histogram.reduce((sum, val) => sum + val, 0);
            if (total > 0) {
                for (let i = 0; i < histogram.length; i++) {
                    histogram[i] /= total;
                }
            }
            
            return {
                id: `${bbox.x}_${bbox.y}_${this.frameCount}`,
                histogram: histogram
            };
        } catch (e) {
            return null;
        }
    }

    /**
     * 用户点击锁定目标
     */
    lockTarget(x, y) {
        let bestTrack = null;
        let minDistance = Infinity;
        
        for (const track of this.tracks.values()) {
            const dx = x - track.x;
            const dy = y - track.y;
            const distance = Math.sqrt(dx*dx + dy*dy);
            
            if (distance < 50 && distance < minDistance) {
                minDistance = distance;
                bestTrack = track;
            }
        }
        
        if (bestTrack) {
            bestTrack.locked = !bestTrack.locked;
            return bestTrack.id;
        }
        
        return null;
    }

    /**
     * 工具函数
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

    isNearExistingTrack(detection) {
        const centerX = detection.x + detection.w/2;
        const centerY = detection.y + detection.h/2;
        
        for (const track of this.tracks.values()) {
            const dx = centerX - track.x;
            const dy = centerY - track.y;
            if (Math.sqrt(dx*dx + dy*dy) < 30) {
                return true;
            }
        }
        return false;
    }

    getTrackColor(id) {
        const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd', '#98d8c8'];
        return colors[id % colors.length];
    }
}

/**
 * 空间网格 - 用于快速空间查询
 */
class SpatialGrid {
    constructor(cols, rows) {
        this.cols = cols;
        this.rows = rows;
        this.cellWidth = 0;
        this.cellHeight = 0;
        this.grid = [];
    }

    clear() {
        this.grid = Array(this.cols * this.rows).fill().map(() => []);
    }

    add(x, y, item) {
        if (this.cellWidth === 0) return; // 未初始化
        
        const col = Math.floor(x / this.cellWidth);
        const row = Math.floor(y / this.cellHeight);
        const index = this.getIndex(col, row);
        
        if (index >= 0 && index < this.grid.length) {
            this.grid[index].push(item);
        }
    }

    query(x, y, radius) {
        if (this.cellWidth === 0) return [];
        
        const results = [];
        const cellRadius = Math.ceil(radius / Math.min(this.cellWidth, this.cellHeight));
        
        const centerCol = Math.floor(x / this.cellWidth);
        const centerRow = Math.floor(y / this.cellHeight);
        
        for (let row = centerRow - cellRadius; row <= centerRow + cellRadius; row++) {
            for (let col = centerCol - cellRadius; col <= centerCol + cellRadius; col++) {
                const index = this.getIndex(col, row);
                if (index >= 0 && index < this.grid.length) {
                    results.push(...this.grid[index]);
                }
            }
        }
        
        return results;
    }

    getIndex(col, row) {
        if (col < 0 || col >= this.cols || row < 0 || row >= this.rows) return -1;
        return row * this.cols + col;
    }

    setDimensions(width, height) {
        this.cellWidth = width / this.cols;
        this.cellHeight = height / this.rows;
    }
}

/**
 * LRU缓存 - 用于特征比较缓存
 */
class LRUCache {
    constructor(capacity) {
        this.capacity = capacity;
        this.cache = new Map();
    }

    get(key) {
        if (this.cache.has(key)) {
            const value = this.cache.get(key);
            this.cache.delete(key);
            this.cache.set(key, value);
            return value;
        }
        return null;
    }

    set(key, value) {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        } else if (this.cache.size >= this.capacity) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        this.cache.set(key, value);
    }

    has(key) {
        return this.cache.has(key);
    }
}

// 导出轻量级追踪器
window.LightweightTracker = LightweightTracker;
