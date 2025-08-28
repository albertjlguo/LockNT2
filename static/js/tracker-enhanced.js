/**
 * Enhanced Multi-Object Tracker with Advanced Components
 * 增强的多目标跟踪器，集成先进组件
 * 
 * Integrates:
 * - Kalman Filter for precise motion prediction
 * - Enhanced appearance feature extraction
 * - Hungarian algorithm for optimal data association
 * - Short-term and long-term occlusion handling
 * - Target re-identification
 */

// Enhanced components are loaded via script tags in HTML
// 增强组件通过HTML中的script标签加载

(function () {
  'use strict';

  // ----------------------------- Utils ---------------------------------
  /** Clamp value into [min, max] */
  function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

  /** Compute IoU for boxes {x,y,w,h} */
  function iou(a, b) {
    const ax2 = a.x + a.w, ay2 = a.y + a.h;
    const bx2 = b.x + b.w, by2 = b.y + b.h;
    const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
    const x2 = Math.min(ax2, bx2), y2 = Math.min(ay2, by2);
    const iw = Math.max(0, x2 - x1), ih = Math.max(0, y2 - y1);
    const inter = iw * ih;
    const ua = a.w * a.h + b.w * b.h - inter;
    return ua <= 0 ? 0 : inter / ua;
  }

  /** Center distance */
  function centerDistance(a, b) {
    const acx = a.x + a.w / 2, acy = a.y + a.h / 2;
    const bcx = b.x + b.w / 2, bcy = b.y + b.h / 2;
    const dx = acx - bcx, dy = acy - bcy;
    return Math.hypot(dx, dy);
  }

  // ----------------------------- Enhanced Track Class ---------------------------------
  let COLOR_POOL = [
    '#ff6b6b', '#6bcB77', '#4d96ff', '#ffd166', '#9b5de5', '#00bbf9', '#f15bb5', '#00f5d4', '#fee440', '#ef476f'
  ];

  class EnhancedTrack {
    constructor(id, bbox, opts = {}) {
      this.id = id;
      this.cx = bbox.x + bbox.w / 2;
      this.cy = bbox.y + bbox.h / 2;
      this.w = bbox.w;
      this.h = bbox.h;
      this.vx = 0;
      this.vy = 0;
      this.locked = !!opts.locked;
      this.color = COLOR_POOL[(id - 1) % COLOR_POOL.length];
      this.lostFrames = 0;
      this.hits = 1;
      this.trajectory = [];
      this.class = opts.class || 'object';
      this.confidence = opts.confidence || 1.0;
      
      // Initialize Kalman filter for precise motion prediction
      // 初始化卡尔曼滤波器进行精确运动预测
      this.kalmanFilter = new KalmanFilter({
        processNoise: 0.1,
        measurementNoise: 0.5,
        initialUncertainty: 10.0
      });
      this.kalmanFilter.initialize(this.cx, this.cy, this.confidence);
      
      // Enhanced appearance features storage
      // 增强的外观特征存储
      this.appearanceFeatures = null;
      this.appearanceHistory = [];
      this.maxAppearanceHistory = 5;
      
      // Occlusion state (managed by OcclusionManager)
      // 遮挡状态（由遮挡管理器管理）
      this.occlusionState = null;
      
      // Track quality metrics
      // 轨迹质量指标
      this.quality = {
        stability: 1.0,
        consistency: 1.0,
        reliability: 1.0
      };
      
      this._pushTrajectory();
    }

    get bbox() { 
      return { 
        x: this.cx - this.w / 2, 
        y: this.cy - this.h / 2, 
        w: this.w, 
        h: this.h 
      }; 
    }

    _pushTrajectory() {
      this.trajectory.push({ x: this.cx, y: this.cy, t: Date.now() });
      if (this.trajectory.length > 60) this.trajectory.shift();
    }

    /**
     * Predict next state using Kalman filter
     * 使用卡尔曼滤波器预测下一状态
     */
    predict() {
      if (this.kalmanFilter) {
        const prediction = this.kalmanFilter.predict();
        this.cx = prediction.x;
        this.cy = prediction.y;
        this.vx = prediction.vx;
        this.vy = prediction.vy;
      }
      
      this._pushTrajectory();
    }

    /**
     * Update track with new detection
     * 使用新检测更新轨迹
     */
    update(detection, features = null) {
      const detectionCenterX = detection.x + detection.w / 2;
      const detectionCenterY = detection.y + detection.h / 2;
      
      // Update Kalman filter with measurement
      // 使用测量值更新卡尔曼滤波器
      if (this.kalmanFilter) {
        this.kalmanFilter.update(detectionCenterX, detectionCenterY, detection.confidence || 0.8);
        const state = this.kalmanFilter.getState();
        this.cx = state.x;
        this.cy = state.y;
        this.vx = state.vx;
        this.vy = state.vy;
      } else {
        // Fallback to simple update
        // 回退到简单更新
        this.cx = detectionCenterX;
        this.cy = detectionCenterY;
      }
      
      // Update size with smoothing
      // 平滑更新尺寸
      const sizeAlpha = 0.7;
      this.w = sizeAlpha * this.w + (1 - sizeAlpha) * detection.w;
      this.h = sizeAlpha * this.h + (1 - sizeAlpha) * detection.h;
      
      // Update appearance features
      // 更新外观特征
      if (features) {
        this.updateAppearanceFeatures(features);
      }
      
      // Reset lost frames and update quality
      // 重置丢失帧数并更新质量
      this.lostFrames = 0;
      this.hits++;
      this.confidence = Math.min(1.0, this.confidence + 0.1);
      this.updateQuality();
      
      this._pushTrajectory();
    }

    /**
     * Update appearance features with history
     * 使用历史更新外观特征
     */
    updateAppearanceFeatures(newFeatures) {
      if (!newFeatures) return;
      
      // Store in history
      // 存储到历史中
      this.appearanceHistory.push({
        features: newFeatures,
        timestamp: Date.now(),
        confidence: this.confidence
      });
      
      // Maintain history size
      // 维护历史大小
      if (this.appearanceHistory.length > this.maxAppearanceHistory) {
        this.appearanceHistory.shift();
      }
      
      // Update current features (weighted average of recent features)
      // 更新当前特征（最近特征的加权平均）
      if (this.appearanceFeatures) {
        const alpha = 0.3;
        // Simple feature blending (assumes features are objects with numeric properties)
        // 简单特征混合（假设特征是具有数值属性的对象）
        for (const key in newFeatures) {
          if (typeof newFeatures[key] === 'number' && this.appearanceFeatures[key] !== undefined) {
            this.appearanceFeatures[key] = alpha * newFeatures[key] + (1 - alpha) * this.appearanceFeatures[key];
          }
        }
      } else {
        this.appearanceFeatures = { ...newFeatures };
      }
    }

    /**
     * Update track quality metrics
     * 更新轨迹质量指标
     */
    updateQuality() {
      // Stability based on trajectory smoothness
      // 基于轨迹平滑度的稳定性
      if (this.trajectory.length >= 3) {
        const recent = this.trajectory.slice(-3);
        let totalVariation = 0;
        for (let i = 1; i < recent.length; i++) {
          const dx = recent[i].x - recent[i-1].x;
          const dy = recent[i].y - recent[i-1].y;
          totalVariation += Math.sqrt(dx*dx + dy*dy);
        }
        this.quality.stability = Math.max(0.1, 1.0 - totalVariation / 100);
      }
      
      // Consistency based on hit rate
      // 基于命中率的一致性
      const totalFrames = this.hits + this.lostFrames;
      this.quality.consistency = totalFrames > 0 ? this.hits / totalFrames : 1.0;
      
      // Reliability based on confidence and age
      // 基于置信度和年龄的可靠性
      const ageBonus = Math.min(1.0, this.hits / 30);
      this.quality.reliability = this.confidence * ageBonus;
    }

    /**
     * Mark track as lost (increment lost frames)
     * 标记轨迹为丢失（增加丢失帧数）
     */
    markLost() {
      this.lostFrames++;
      this.confidence = Math.max(0.1, this.confidence * 0.95);
    }
  }

  // ----------------------------- Enhanced Tracker ---------------------------------
  class EnhancedTracker {
    constructor(opts = {}) {
      this.tracks = [];
      this.nextId = 1;
      this.enableReID = opts.enableReID ?? true;
      this.frameCount = 0;
      
      // Initialize enhanced components
      // 初始化增强组件
      this.appearanceExtractor = new EnhancedAppearanceExtractor({
        colorBins: { h: 20, s: 8, v: 6 },
        spatialGrid: { rows: 3, cols: 3 },
        enableTexture: true,
        textureRadius: 2
      });
      
      this.dataAssociation = new DataAssociationManager({
        iouWeight: 0.25,
        distanceWeight: 0.2,
        appearanceWeight: 0.35,
        motionWeight: 0.2,
        maxDistance: 150,
        minIoU: 0.05,
        normalThreshold: 0.7,
        crowdedThreshold: 0.6,
        lockedThreshold: 0.8
      });
      
      this.occlusionManager = new OcclusionManager({
        shortTermLimit: 30,
        longTermLimit: 90,
        confidenceDecay: 0.95,
        shortTermRadius: 100,
        longTermRadius: 200,
        appearanceWeight: 0.6,
        motionWeight: 0.4
      });
      
      this.reidentificationManager = new ReidentificationManager({
        expandedSearchRadius: 200,
        appearanceWeight: 0.6,
        motionWeight: 0.25,
        sizeWeight: 0.15,
        minSimilarityThreshold: 0.65,
        maxFramesForReidentification: 90
      });
      
      // Performance monitoring
      // 性能监控
      this.performance = {
        lastUpdateTime: 0,
        avgProcessingTime: 0,
        frameRate: 30
      };
    }

    /**
     * Main tracking update function
     * 主要跟踪更新函数
     */
    update(detections, ctx = null) {
      const startTime = performance.now();
      this.frameCount++;
      
      // Convert detections to standard format
      // 将检测转换为标准格式
      const standardDetections = this.standardizeDetections(detections);
      
      // Predict all tracks
      // 预测所有轨迹
      this.predictTracks();
      
      // Handle occlusion for all tracks
      // 处理所有轨迹的遮挡
      this.handleOcclusions();
      
      // Extract appearance features for detections
      // 为检测提取外观特征
      const detectionFeatures = this.extractDetectionFeatures(standardDetections, ctx);
      
      // Perform data association
      // 执行数据关联
      const association = this.associateTracksAndDetections(
        standardDetections, detectionFeatures, ctx
      );
      
      // Update matched tracks
      // 更新匹配的轨迹
      this.updateMatchedTracks(association.assignments, standardDetections, detectionFeatures);
      
      // Handle unmatched tracks
      // 处理未匹配的轨迹
      this.handleUnmatchedTracks(association.unmatchedTracks);
      
      // Attempt re-identification for lost tracks
      // 尝试为丢失的轨迹重新识别
      this.attemptReidentification(association.unmatchedDetections, standardDetections, detectionFeatures, ctx);
      
      // Create new tracks from unmatched detections
      // 从未匹配的检测创建新轨迹
      this.createNewTracks(association.unmatchedDetections, standardDetections, detectionFeatures);
      
      // Prune old tracks
      // 修剪旧轨迹
      this.pruneTracks();
      
      // Update performance metrics
      // 更新性能指标
      this.updatePerformanceMetrics(performance.now() - startTime);
      
      // Cleanup feature banks periodically
      // 定期清理特征库
      if (this.frameCount % 300 === 0) {
        this.reidentificationManager.cleanupFeatureBank(this.frameCount);
      }
    }

    /**
     * Standardize detection format
     * 标准化检测格式
     */
    standardizeDetections(detections) {
      return detections.map(d => {
        if (d.bbox) {
          return {
            x: d.bbox[0],
            y: d.bbox[1],
            w: d.bbox[2],
            h: d.bbox[3],
            class: d.class,
            confidence: d.score || 0.8
          };
        }
        return {
          x: d.x,
          y: d.y,
          w: d.w,
          h: d.h,
          class: d.class,
          confidence: d.score || d.confidence || 0.8
        };
      });
    }

    /**
     * Predict all tracks using their motion models
     * 使用运动模型预测所有轨迹
     */
    predictTracks() {
      for (const track of this.tracks) {
        track.predict();
      }
    }

    /**
     * Handle occlusion states for all tracks
     * 处理所有轨迹的遮挡状态
     */
    handleOcclusions() {
      for (const track of this.tracks) {
        const occlusionResult = this.occlusionManager.handleOcclusion(track, this.frameCount);
        
        // Update track based on occlusion state
        // 根据遮挡状态更新轨迹
        if (occlusionResult.state === 'visible') {
          // Track is visible, normal processing
          // 轨迹可见，正常处理
        } else if (occlusionResult.state === 'short_term_occluded') {
          // Continue with Kalman prediction
          // 继续卡尔曼预测
        } else if (occlusionResult.state === 'long_term_occluded') {
          // Prepare for re-identification
          // 准备重新识别
        } else if (occlusionResult.state === 'lost') {
          // Mark for deletion
          // 标记删除
          track.shouldDelete = true;
        }
      }
    }

    /**
     * Extract appearance features for detections
     * 为检测提取外观特征
     */
    extractDetectionFeatures(detections, ctx) {
      if (!ctx || !this.enableReID) return [];
      
      const features = [];
      for (const detection of detections) {
        try {
          const feature = this.appearanceExtractor.extractFeatures(ctx, detection, detection.class);
          features.push(feature);
        } catch (error) {
          console.warn('Feature extraction failed:', error);
          features.push(null);
        }
      }
      return features;
    }

    /**
     * Associate tracks with detections using Hungarian algorithm
     * 使用匈牙利算法关联轨迹与检测
     */
    associateTracksAndDetections(detections, detectionFeatures, ctx) {
      const activeTracks = this.tracks.filter(t => !t.shouldDelete && t.lostFrames < 30);
      
      if (activeTracks.length === 0 || detections.length === 0) {
        return {
          assignments: [],
          unmatchedTracks: activeTracks.map((_, i) => i),
          unmatchedDetections: detections.map((_, i) => i)
        };
      }
      
      return this.dataAssociation.associate(
        activeTracks, 
        detections, 
        this.appearanceExtractor, 
        ctx
      );
    }

    /**
     * Update tracks that were successfully matched
     * 更新成功匹配的轨迹
     */
    updateMatchedTracks(assignments, detections, detectionFeatures) {
      for (const assignment of assignments) {
        const track = this.tracks.find(t => t.id === this.tracks[assignment.trackIdx].id);
        const detection = detections[assignment.detectionIdx];
        const features = detectionFeatures[assignment.detectionIdx];
        
        if (track) {
          track.update(detection, features);
          
          // Store features for re-identification
          // 存储特征用于重新识别
          if (features) {
            this.reidentificationManager.storeTrackFeatures(track, features, this.frameCount);
          }
        }
      }
    }

    /**
     * Handle tracks that were not matched
     * 处理未匹配的轨迹
     */
    handleUnmatchedTracks(unmatchedTrackIndices) {
      for (const trackIdx of unmatchedTrackIndices) {
        const track = this.tracks[trackIdx];
        if (track) {
          track.markLost();
        }
      }
    }

    /**
     * Attempt re-identification for lost tracks
     * 尝试为丢失的轨迹重新识别
     */
    attemptReidentification(unmatchedDetectionIndices, detections, detectionFeatures, ctx) {
      const lostTracks = this.tracks.filter(t => t.lostFrames >= 30 && t.lostFrames < 90);
      
      if (lostTracks.length === 0 || unmatchedDetectionIndices.length === 0) {
        return;
      }
      
      const unmatchedDetections = unmatchedDetectionIndices.map(i => detections[i]);
      
      const reidentifications = this.reidentificationManager.attemptReidentification(
        lostTracks,
        unmatchedDetections,
        this.appearanceExtractor,
        ctx,
        this.frameCount
      );
      
      // Process successful re-identifications
      // 处理成功的重新识别
      for (const reidentification of reidentifications) {
        const track = reidentification.track;
        const detection = reidentification.detection;
        const detectionIdx = unmatchedDetectionIndices[reidentification.detectionIndex];
        const features = detectionFeatures[detectionIdx];
        
        // Reset Kalman filter after re-identification
        // 重新识别后重置卡尔曼滤波器
        this.occlusionManager.resetKalmanAfterReidentification(track, detection);
        
        // Update track
        // 更新轨迹
        track.update(detection, features);
        
        // Remove from unmatched detections
        // 从未匹配检测中移除
        const removeIdx = unmatchedDetectionIndices.indexOf(detectionIdx);
        if (removeIdx !== -1) {
          unmatchedDetectionIndices.splice(removeIdx, 1);
        }
        
        console.log(`✓ Re-identified track ${track.id} with similarity ${reidentification.similarity.toFixed(3)}`);
      }
    }

    /**
     * Create new tracks from remaining unmatched detections
     * 从剩余未匹配检测创建新轨迹
     */
    createNewTracks(unmatchedDetectionIndices, detections, detectionFeatures) {
      for (const detectionIdx of unmatchedDetectionIndices) {
        const detection = detections[detectionIdx];
        const features = detectionFeatures[detectionIdx];
        
        // Only create tracks for high-confidence detections
        // 仅为高置信度检测创建轨迹
        if (detection.confidence > 0.6) {
          const trackId = this.createTrack(detection, features, false);
          if (trackId) {
            console.log(`Created new track ${trackId} for ${detection.class} (conf: ${detection.confidence.toFixed(3)})`);
          }
        }
      }
    }

    /**
     * Create a new track
     * 创建新轨迹
     */
    createTrack(detection, features = null, locked = false) {
      const id = this.nextId++;
      const track = new EnhancedTrack(id, detection, { 
        locked, 
        class: detection.class,
        confidence: detection.confidence 
      });
      
      if (features) {
        track.updateAppearanceFeatures(features);
        this.reidentificationManager.storeTrackFeatures(track, features, this.frameCount);
      }
      
      this.tracks.push(track);
      return id;
    }

    /**
     * Remove old and invalid tracks
     * 移除旧的和无效的轨迹
     */
    pruneTracks() {
      const beforeCount = this.tracks.length;
      
      this.tracks = this.tracks.filter(track => {
        // Remove tracks marked for deletion
        // 移除标记删除的轨迹
        if (track.shouldDelete) return false;
        
        // Keep locked tracks longer
        // 保持锁定轨迹更长时间
        if (track.locked) {
          return track.lostFrames < 150;
        }
        
        // Remove tracks lost for too long
        // 移除丢失太久的轨迹
        return track.lostFrames < 90;
      });
      
      const prunedCount = beforeCount - this.tracks.length;
      if (prunedCount > 0) {
        console.log(`Pruned ${prunedCount} tracks`);
      }
    }

    /**
     * Update performance metrics
     * 更新性能指标
     */
    updatePerformanceMetrics(processingTime) {
      const alpha = 0.1;
      this.performance.avgProcessingTime = 
        alpha * processingTime + (1 - alpha) * this.performance.avgProcessingTime;
      
      this.performance.lastUpdateTime = Date.now();
    }

    /**
     * Lock a track by clicking a point on canvas
     * 通过在画布上点击来锁定轨迹
     */
    lockFromPoint(x, y, detections, ctx) {
      if (!detections || detections.length === 0) return null;
      
      const standardDetections = this.standardizeDetections(detections);
      
      // Find detection containing the point
      // 找到包含该点的检测
      const candidates = standardDetections
        .map((d, i) => ({ i, d }))
        .filter(o => x >= o.d.x && y >= o.d.y && x <= o.d.x + o.d.w && y <= o.d.y + o.d.h)
        .sort((a, b) => (b.d.confidence || 0) - (a.d.confidence || 0));

      let chosen = candidates[0];
      
      // Fallback: nearest center within 40px
      // 回退：40px内最近的中心
      if (!chosen) {
        let best = null, bestDist = 40;
        for (let i = 0; i < standardDetections.length; i++) {
          const d = standardDetections[i];
          const cx = d.x + d.w / 2, cy = d.y + d.h / 2;
          const dist = Math.hypot(cx - x, cy - y);
          if (dist < bestDist) { 
            bestDist = dist; 
            best = { i, d }; 
          }
        }
        if (best) chosen = best;
      }

      if (!chosen) return null;

      // Check if existing track overlaps, lock it
      // 检查现有轨迹是否重叠，锁定它
      for (const track of this.tracks) {
        const trackBbox = track.bbox;
        const detectionBbox = chosen.d;
        const iouValue = iou(trackBbox, detectionBbox);
        const distance = centerDistance(trackBbox, detectionBbox);
        
        if (iouValue > 0.4 || distance < 20) {
          track.locked = true;
          track.lostFrames = 0;
          return track.id;
        }
      }

      // Create new locked track
      // 创建新的锁定轨迹
      const features = (this.enableReID && ctx) ? 
        this.appearanceExtractor.extractFeatures(ctx, chosen.d, chosen.d.class) : null;
      
      return this.createTrack(chosen.d, features, true);
    }

    /**
     * Unlock and optionally remove a track
     * 解锁并可选择移除轨迹
     */
    unlock(id, remove = false) {
      const idx = this.tracks.findIndex(t => t.id === id);
      if (idx < 0) return false;
      
      if (remove) {
        this.tracks.splice(idx, 1);
        return true;
      }
      
      this.tracks[idx].locked = false;
      return true;
    }

    /**
     * Clear all tracks
     * 清除所有轨迹
     */
    clear() {
      this.tracks = [];
      this.nextId = 1;
      this.frameCount = 0;
    }

    /**
     * Get tracking statistics
     * 获取跟踪统计信息
     */
    getStats() {
      const activeTracks = this.tracks.filter(t => t.lostFrames < 5);
      const occludedTracks = this.tracks.filter(t => t.lostFrames >= 5 && t.lostFrames < 30);
      const lostTracks = this.tracks.filter(t => t.lostFrames >= 30);
      
      return {
        total: this.tracks.length,
        active: activeTracks.length,
        occluded: occludedTracks.length,
        lost: lostTracks.length,
        locked: this.tracks.filter(t => t.locked).length,
        avgProcessingTime: this.performance.avgProcessingTime,
        frameCount: this.frameCount,
        reidentificationStats: this.reidentificationManager.getReidentificationStats(),
        occlusionStats: this.occlusionManager.getOcclusionStats(this.tracks)
      };
    }

    /**
     * Get shallow copy of tracks
     * 获取轨迹的浅拷贝
     */
    getTracks() {
      return this.tracks.slice();
    }
  }

  // Export enhanced tracker
  // 导出增强跟踪器
  window.EnhancedTracker = EnhancedTracker;
  window.Tracker = EnhancedTracker; // Replace original tracker
})();
