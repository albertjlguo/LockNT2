/**
 * Tracker.js - Lightweight Multi-Object Tracker with Click-to-Lock and Appearance ReID
 * 轻量级多目标追踪（点击锁定 + 外观重识别）
 *
 * Key features:
 * - Alpha-Beta motion model (constant velocity) for prediction
 * - IoU + appearance (HSV histogram) cost for association
 * - Click-to-lock a detection as a persistent track
 * - Track TTL handling and simple trajectory rendering support
 *
 * Notes:
 * - All coordinates are in canvas space (pixels of videoCanvas/detectionCanvas)
 * - Only a subset of detections (e.g., 'person') may compute appearance to save cost
 */

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

  /** L2 normalize a Float32Array in-place */
  function l2normalize(arr) {
    let sum = 0; for (let i = 0; i < arr.length; i++) sum += arr[i] * arr[i];
    const n = Math.sqrt(sum) || 1;
    for (let i = 0; i < arr.length; i++) arr[i] /= n;
    return arr;
  }

  /** Cosine similarity for Float32Array */
  function cosineSimilarity(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    const denom = Math.sqrt(na) * Math.sqrt(nb) || 1;
    return dot / denom;
  }

  /** Convert RGB (0-255) to HSV with H in [0,360), S,V in [0,1] */
  function rgbToHsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const d = max - min;
    let h = 0, s = max === 0 ? 0 : d / max, v = max;
    if (d !== 0) {
      switch (max) {
        case r: h = ((g - b) / d + (g < b ? 6 : 0)); break;
        case g: h = ((b - r) / d + 2); break;
        case b: h = ((r - g) / d + 4); break;
      }
      h *= 60;
    }
    return { h, s, v };
  }

  // ----------------------- Appearance Encoder --------------------------
  /**
   * HSV histogram encoder (coarse bins) for lightweight appearance features.
   * Default bins: H=16, S=4, V=2 => 128-D feature.
   */
  class AppearanceEncoder {
    constructor(opts = {}) {
      this.binsH = opts.binsH || 16;
      this.binsS = opts.binsS || 4;
      this.binsV = opts.binsV || 2;
      this.innerCrop = clamp(opts.innerCrop ?? 0.8, 0.5, 1.0);
      this.sampleStep = opts.sampleStep || 2; // pixel stride to reduce cost
      const dim = this.binsH * this.binsS * this.binsV;
      this._tmp = new Float32Array(dim);
    }

    /** Extract appearance feature from canvas 2D context within bbox */
    extract(ctx, bbox) {
      const x = bbox.x + (1 - this.innerCrop) * 0.5 * bbox.w;
      const y = bbox.y + (1 - this.innerCrop) * 0.5 * bbox.h;
      const w = bbox.w * this.innerCrop;
      const h = bbox.h * this.innerCrop;

      const ix = Math.round(x), iy = Math.round(y);
      const iw = Math.max(1, Math.round(w)), ih = Math.max(1, Math.round(h));
      if (iw < 4 || ih < 4) return null; // too small

      let img;
      try { img = ctx.getImageData(ix, iy, iw, ih); }
      catch (e) { return null; } // likely tainted canvas or out-of-bounds

      const hist = this._tmp; hist.fill(0);
      const data = img.data; // RGBA of length iw*ih*4
      const step = Math.max(1, this.sampleStep);

      for (let yy = 0; yy < ih; yy += step) {
        for (let xx = 0; xx < iw; xx += step) {
          const p = (yy * iw + xx) * 4;
          const r = data[p], g = data[p + 1], b = data[p + 2];
          const { h: H, s: S, v: V } = rgbToHsv(r, g, b);
          const hbin = clamp(Math.floor((H / 360) * this.binsH), 0, this.binsH - 1);
          const sbin = clamp(Math.floor(S * this.binsS), 0, this.binsS - 1);
          const vbin = clamp(Math.floor(V * this.binsV), 0, this.binsV - 1);
          const idx = hbin * (this.binsS * this.binsV) + sbin * this.binsV + vbin;
          hist[idx] += 1;
        }
      }

      return l2normalize(new Float32Array(hist));
    }

    /** Compute appearance distance in [0,1] (1 - cosine similarity) */
    distance(a, b) { return 1 - cosineSimilarity(a, b); }
  }

  // ----------------------------- Track ---------------------------------
  let COLOR_POOL = [
    '#ff6b6b', '#6bcB77', '#4d96ff', '#ffd166', '#9b5de5', '#00bbf9', '#f15bb5', '#00f5d4', '#fee440', '#ef476f'
  ];

  class Track {
    constructor(id, bbox, opts = {}) {
      this.id = id;
      this.cx = bbox.x + bbox.w / 2; // center
      this.cy = bbox.y + bbox.h / 2;
      this.w = bbox.w; this.h = bbox.h;
      this.vx = 0; this.vy = 0;
      this.alpha = opts.alpha ?? 0.7; // Increased for better responsiveness
      this.beta = opts.beta ?? 0.3;   // Reduced for smoother velocity updates
      this.locked = !!opts.locked;
      this.color = COLOR_POOL[(id - 1) % COLOR_POOL.length];
      this.feature = null; // EMA feature
      this.lostFrames = 0;
      this.hits = 1;
      this.trajectory = [];
      
      // Enhanced tracking state for occlusion handling
      // 增强的追踪状态用于遮挡处理
      this.occlusionState = {
        isOccluded: false,
        occlusionStartFrame: 0,
        lastKnownPosition: { cx: this.cx, cy: this.cy },
        predictedPosition: { cx: this.cx, cy: this.cy },
        confidence: 1.0,
        searchRadius: Math.max(bbox.w, bbox.h) * 0.5
      };
      
      // Motion model improvements
      // 运动模型改进
      this.motionModel = {
        acceleration: { ax: 0, ay: 0 },
        velocityHistory: [],
        positionHistory: [],
        maxHistory: 10
      };
      
      // Appearance model enhancements
      // 外观模型增强
      this.appearanceModel = {
        templates: [], // Multiple appearance templates
        weights: [],   // Template weights
        maxTemplates: 3,
        updateThreshold: 0.8
      };
      
      this._pushTrajectory();
    }

    get bbox() { return { x: this.cx - this.w / 2, y: this.cy - this.h / 2, w: this.w, h: this.h }; }

    _pushTrajectory() {
      this.trajectory.push({ x: this.cx, y: this.cy });
      if (this.trajectory.length > 60) this.trajectory.shift();
    }

    /** Enhanced prediction with motion model and occlusion handling */
    predict() {
      // Update motion history
      // 更新运动历史
      this.motionModel.positionHistory.push({ x: this.cx, y: this.cy, t: Date.now() });
      this.motionModel.velocityHistory.push({ vx: this.vx, vy: this.vy, t: Date.now() });
      
      if (this.motionModel.positionHistory.length > this.motionModel.maxHistory) {
        this.motionModel.positionHistory.shift();
        this.motionModel.velocityHistory.shift();
      }
      
      // Calculate acceleration from velocity history
      // 从速度历史计算加速度
      if (this.motionModel.velocityHistory.length >= 2) {
        const recent = this.motionModel.velocityHistory.slice(-2);
        const dt = (recent[1].t - recent[0].t) / 1000; // Convert to seconds
        if (dt > 0) {
          this.motionModel.acceleration.ax = (recent[1].vx - recent[0].vx) / dt;
          this.motionModel.acceleration.ay = (recent[1].vy - recent[0].vy) / dt;
        }
      }
      
      // Enhanced prediction with acceleration
      // 包含加速度的增强预测
      const dt = 1/30; // Assume 30 FPS
      let xp = this.cx + this.vx * dt + 0.5 * this.motionModel.acceleration.ax * dt * dt;
      let yp = this.cy + this.vy * dt + 0.5 * this.motionModel.acceleration.ay * dt * dt;
      
      // Update velocity with acceleration
      // 使用加速度更新速度
      this.vx += this.motionModel.acceleration.ax * dt;
      this.vy += this.motionModel.acceleration.ay * dt;
      
      // Apply damping to prevent runaway
      // 应用阻尼防止失控
      const dampingFactor = this.occlusionState.isOccluded ? 0.95 : 0.98;
      this.vx *= dampingFactor;
      this.vy *= dampingFactor;
      
      // Handle occlusion state
      // 处理遮挡状态
      if (this.occlusionState.isOccluded) {
        // Expand search radius during occlusion
        // 遮挡期间扩大搜索半径
        this.occlusionState.searchRadius *= 1.05;
        this.occlusionState.searchRadius = Math.min(this.occlusionState.searchRadius, 200);
        
        // Reduce confidence over time
        // 随时间降低置信度
        this.occlusionState.confidence *= 0.95;
        
        // Store predicted position
        // 存储预测位置
        this.occlusionState.predictedPosition = { cx: xp, cy: yp };
      } else {
        // Update last known position when not occluded
        // 未遮挡时更新最后已知位置
        this.occlusionState.lastKnownPosition = { cx: this.cx, cy: this.cy };
        this.occlusionState.confidence = Math.min(1.0, this.occlusionState.confidence + 0.1);
      }
      
      this.cx = xp;
      this.cy = yp;
      this._pushTrajectory();
    }

    /** Enhanced update with occlusion recovery and appearance templates */
    update(detBbox, feature) {
      const zx = detBbox.x + detBbox.w / 2;
      const zy = detBbox.y + detBbox.h / 2;
      
      // Check if this is a recovery from occlusion
      // 检查是否从遮挡中恢复
      const wasOccluded = this.occlusionState.isOccluded;
      
      // prediction step
      const xp = this.cx + this.vx;
      const yp = this.cy + this.vy;
      
      // Calculate innovation (measurement residual)
      // 计算新息（测量残差）
      const rx = zx - xp; 
      const ry = zy - yp;
      const innovationMagnitude = Math.sqrt(rx * rx + ry * ry);
      
      // Adaptive update gains based on innovation and occlusion state
      // 基于新息和遮挡状态的自适应更新增益
      let adaptiveAlpha = this.alpha;
      let adaptiveBeta = this.beta;
      
      if (wasOccluded) {
        // Higher gains for quick recovery from occlusion
        // 遮挡恢复时使用更高增益
        adaptiveAlpha = Math.min(0.9, this.alpha * 1.5);
        adaptiveBeta = Math.min(0.5, this.beta * 1.2);
        
        // Reset occlusion state
        // 重置遮挡状态
        this.occlusionState.isOccluded = false;
        this.occlusionState.searchRadius = Math.max(detBbox.w, detBbox.h) * 0.5;
      } else if (innovationMagnitude > 50) {
        // Large innovation suggests rapid movement, increase responsiveness
        // 大的新息表明快速运动，增加响应性
        adaptiveAlpha = Math.min(0.8, this.alpha * 1.2);
        adaptiveBeta = Math.min(0.4, this.beta * 1.1);
      }
      
      // State update
      // 状态更新
      this.cx = xp + adaptiveAlpha * rx;
      this.cy = yp + adaptiveAlpha * ry;
      this.vx = this.vx + adaptiveBeta * rx;
      this.vy = this.vy + adaptiveBeta * ry;
      
      // Adaptive size smoothing based on confidence
      // 基于置信度的自适应尺寸平滑
      const sizeWeight = Math.max(0.2, this.occlusionState.confidence * 0.4);
      this.w = (1 - sizeWeight) * this.w + sizeWeight * detBbox.w;
      this.h = (1 - sizeWeight) * this.h + sizeWeight * detBbox.h;
      
      this.lostFrames = 0;
      this.hits += 1;
      this._pushTrajectory();
      
      // Enhanced appearance model with multiple templates
      // 增强的外观模型，支持多个模板
      if (feature) {
        this.updateAppearanceModel(feature);
      }
    }
    
    /** Update appearance model with template management */
    updateAppearanceModel(newFeature) {
      if (!this.feature || this.feature.length !== newFeature.length) {
        // Initialize first template
        // 初始化第一个模板
        this.feature = newFeature;
        this.appearanceModel.templates = [newFeature];
        this.appearanceModel.weights = [1.0];
        return;
      }
      
      // Calculate similarity with existing templates
      // 计算与现有模板的相似度
      const similarities = this.appearanceModel.templates.map(template => 
        cosineSimilarity(newFeature, template)
      );
      
      const maxSimilarity = Math.max(...similarities);
      const bestTemplateIdx = similarities.indexOf(maxSimilarity);
      
      if (maxSimilarity > this.appearanceModel.updateThreshold) {
        // Update existing template
        // 更新现有模板
        const mu = 0.2;
        for (let i = 0; i < this.feature.length; i++) {
          this.appearanceModel.templates[bestTemplateIdx][i] = 
            (1 - mu) * this.appearanceModel.templates[bestTemplateIdx][i] + mu * newFeature[i];
        }
        l2normalize(this.appearanceModel.templates[bestTemplateIdx]);
        
        // Update main feature to best template
        // 将主特征更新为最佳模板
        this.feature = this.appearanceModel.templates[bestTemplateIdx];
      } else {
        // Add new template if we have space
        // 如果有空间则添加新模板
        if (this.appearanceModel.templates.length < this.appearanceModel.maxTemplates) {
          this.appearanceModel.templates.push(newFeature);
          this.appearanceModel.weights.push(0.5);
        } else {
          // Replace least weighted template
          // 替换权重最小的模板
          const minWeightIdx = this.appearanceModel.weights.indexOf(
            Math.min(...this.appearanceModel.weights)
          );
          this.appearanceModel.templates[minWeightIdx] = newFeature;
          this.appearanceModel.weights[minWeightIdx] = 0.5;
        }
      }
      
      // Normalize weights
      // 归一化权重
      const totalWeight = this.appearanceModel.weights.reduce((sum, w) => sum + w, 0);
      if (totalWeight > 0) {
        this.appearanceModel.weights = this.appearanceModel.weights.map(w => w / totalWeight);
      }
    }
    
    /** Get best appearance match score with a feature */
    getAppearanceMatchScore(feature) {
      if (!feature || this.appearanceModel.templates.length === 0) return 0;
      
      const similarities = this.appearanceModel.templates.map((template, idx) => 
        cosineSimilarity(feature, template) * this.appearanceModel.weights[idx]
      );
      
      return Math.max(...similarities);
    }
    
    /** Check if track should enter occlusion state */
    checkOcclusionState(frameCount) {
      if (this.lostFrames > 3 && !this.occlusionState.isOccluded) {
        this.occlusionState.isOccluded = true;
        this.occlusionState.occlusionStartFrame = frameCount;
        this.occlusionState.searchRadius = Math.max(this.w, this.h) * 0.8;
      }
    }
  }

  // ----------------------------- Tracker -------------------------------
  class Tracker {
    constructor(opts = {}) {
      this.tracks = [];
      this.nextId = 1;
      this.encoder = new AppearanceEncoder(opts.encoder || {});
      this.enableReID = opts.enableReID !== false; // default true
      this.focusClasses = opts.focusClasses || ['person'];
      this.detectHigh = opts.detectHigh ?? 0.5;
      this.detectLow = opts.detectLow ?? 0.3;
      this.gatingBase = opts.gatingBase ?? 80; // Increased for better occlusion handling
      this.maxLostUnlocked = opts.maxLostUnlocked ?? 30; // Reduced for efficiency
      this.maxLostLocked = opts.maxLostLocked ?? 120; // Increased for occlusion tolerance
      
      // Enhanced matching weights for better tracking
      // 增强的匹配权重以改善追踪效果
      this.wIoU = opts.wIoU ?? 0.35;        // Reduced IoU weight
      this.wApp = opts.wApp ?? 0.40;        // Appearance weight
      this.wCtr = opts.wCtr ?? 0.15;        // Increased center distance weight
      this.wRatio = opts.wRatio ?? 0.05;    // Aspect ratio weight
      this.wMotion = opts.wMotion ?? 0.05;  // New: motion consistency weight
      
      this.costThreshold = opts.costThreshold ?? 0.75; // Lowered for better matching
      this.autoCreate = !!opts.autoCreate; // default false: only create via click lock
      
      // Occlusion handling parameters
      // 遮挡处理参数
      this.occlusionParams = {
        maxOcclusionFrames: 60,    // Max frames to maintain occluded track
        reappearanceRadius: 150,   // Search radius for reappearance
        confidenceDecay: 0.95,     // Confidence decay per frame during occlusion
        minReappearanceScore: 0.6  // Min score to consider reappearance
      };
      
      // Frame counter for occlusion management
      // 用于遮挡管理的帧计数器
      this.frameCount = 0;
    }

    /** Enhanced predict-only step with occlusion state management */
    predictOnly() {
      this.frameCount++;
      
      for (const t of this.tracks) {
        t.predict();
        t.lostFrames += 1;
        
        // Check and update occlusion state
        // 检查并更新遮挡状态
        t.checkOcclusionState(this.frameCount);
        
        // Handle long-term occlusion
        // 处理长期遮挡
        if (t.occlusionState.isOccluded) {
          const occlusionDuration = this.frameCount - t.occlusionState.occlusionStartFrame;
          if (occlusionDuration > this.occlusionParams.maxOcclusionFrames) {
            // Mark for removal if occluded too long
            // 如果遮挡时间过长则标记为删除
            t.lostFrames = this.maxLostLocked + 1;
          }
        }
      }
      
      this._prune();
    }

    /** Main update with detections (canvas-space bboxes). ctx is 2D context for appearance. */
    update(detections, ctx) {
      // Prepare detection objects
      const dets = (detections || []).map(d => ({
        x: d.bbox[0] ?? d.bbox.x ?? d.x,
        y: d.bbox[1] ?? d.bbox.y ?? d.y,
        w: d.bbox[2] ?? d.bbox.w ?? d.width ?? d.w,
        h: d.bbox[3] ?? d.bbox.h ?? d.height ?? d.h,
        score: d.score ?? d.confidence ?? 0,
        class: d.class || d.label || ''
      }));

      // Optionally compute appearance features (only for focus classes and limited count)
      let enriched = dets;
      if (this.enableReID && ctx) {
        const maxFeat = 15; // cap to save cost
        enriched = dets.map((dd, idx) => {
          let feat = null;
          if (!this.focusClasses.length || this.focusClasses.includes(dd.class)) {
            if (idx < maxFeat) feat = this.encoder.extract(ctx, dd) || null;
          }
          return { ...dd, feature: feat };
        });
      } else {
        enriched = dets.map(dd => ({ ...dd, feature: null }));
      }

      // Predict existing tracks first
      for (const t of this.tracks) t.predict();

      // Associate in two passes: locked tracks first, then others
      const locked = this.tracks.filter(t => t.locked);
      const normal = this.tracks.filter(t => !t.locked);

      const unmatchedDetIdx = new Set(enriched.map((_, i) => i));

      // Pass 1: locked tracks with strict association
      this._associateAndUpdate(locked, enriched, unmatchedDetIdx, true);
      // Pass 2: normal tracks
      this._associateAndUpdate(normal, enriched, unmatchedDetIdx, false);

      // Increase lost for unmatched tracks
      for (const t of this.tracks) {
        if (!t._updatedThisRound) {
          t.lostFrames += 1;
        }
        delete t._updatedThisRound;
      }

      // Optionally create new tracks from remaining detections (if enabled)
      if (this.autoCreate) {
        for (const idx of unmatchedDetIdx) {
          const d = enriched[idx];
          if (d.score >= this.detectHigh) this._createTrackFromDet(d, d.feature, false);
        }
      }

      this._prune();
    }

    /** Enhanced association with occlusion-aware matching */
    _associateAndUpdate(trackList, detections, unmatchedDetIdx, isLockedPass) {
      if (trackList.length === 0 || unmatchedDetIdx.size === 0) return;

      const pairs = [];
      for (let ti = 0; ti < trackList.length; ti++) {
        const t = trackList[ti];
        const tb = t.bbox;
        
        // Adaptive gating for occlusion handling
        let gating = Math.max(this.gatingBase, 0.5 * Math.hypot(t.w, t.h));
        if (t.occlusionState && t.occlusionState.isOccluded) {
          gating = Math.max(gating, t.occlusionState.searchRadius || gating * 1.5);
        }
        
        for (const di of unmatchedDetIdx) {
          const d = detections[di];
          const db = { x: d.x, y: d.y, w: d.w, h: d.h };
          const i = iou(tb, db);
          const ctr = centerDistance(tb, db);
          
          const gatingMultiplier = (t.occlusionState && t.occlusionState.isOccluded) ? 2.5 : 2.0;
          if (i < 0.01 && ctr > gating * gatingMultiplier) continue;

          // Enhanced appearance matching
          let app = 1;
          if (this.enableReID && d.feature) {
            if (t.getAppearanceMatchScore && t.appearanceModel && t.appearanceModel.templates.length > 0) {
              app = 1 - t.getAppearanceMatchScore(d.feature);
            } else if (t.feature) {
              app = clamp(1 - cosineSimilarity(t.feature, d.feature), 0, 1);
            } else {
              t.feature = d.feature;
              app = 0.3;
            }
          }

          const ratioT = (t.w / (t.h + 1e-3));
          const ratioD = (d.w / (d.h + 1e-3));
          const ratioDiff = Math.min(1, Math.abs(ratioT - ratioD) / Math.max(ratioT, ratioD));
          const ctrNorm = clamp(ctr / (gating * 1.5), 0, 1);
          
          // Motion consistency
          const predictedX = t.cx + (t.vx || 0);
          const predictedY = t.cy + (t.vy || 0);
          const detCenterX = d.x + d.w / 2;
          const detCenterY = d.y + d.h / 2;
          const motionDist = Math.sqrt((predictedX - detCenterX) ** 2 + (predictedY - detCenterY) ** 2);
          const motionCost = clamp(motionDist / gating, 0, 1);

          // Adaptive weights
          let wIoU = this.wIoU, wApp = this.wApp, wCtr = this.wCtr;
          const wMotion = this.wMotion || 0.05;
          
          if (isLockedPass) {
            wIoU = Math.max(0.25, this.wIoU - 0.1);
            wApp = Math.min(0.5, this.wApp + 0.1);
            wCtr = Math.min(0.2, this.wCtr + 0.05);
          }
          
          if (t.occlusionState && t.occlusionState.isOccluded) {
            wIoU *= 0.5; wApp *= 1.3; wCtr *= 1.2;
          }

          const cost = wIoU * (1 - i) + wApp * app + wCtr * ctrNorm + 
                      (this.wRatio || 0.03) * ratioDiff + wMotion * motionCost;
          
          pairs.push({ cost, ti, di });
        }
      }

      pairs.sort((a, b) => a.cost - b.cost);
      const usedT = new Set();
      
      for (const p of pairs) {
        let threshold = this.costThreshold;
        const track = trackList[p.ti];
        
        if (track.occlusionState && track.occlusionState.isOccluded) threshold *= 1.2;
        if (track.locked) threshold *= 1.1;
        
        if (p.cost > threshold) break;
        if (usedT.has(p.ti) || !unmatchedDetIdx.has(p.di)) continue;
        
        const t = trackList[p.ti];
        const d = detections[p.di];
        t.update({ x: d.x, y: d.y, w: d.w, h: d.h }, d.feature);
        t._updatedThisRound = true;
        unmatchedDetIdx.delete(p.di);
        usedT.add(p.ti);
      }
    }

    /** Remove stale tracks */
    _prune() {
      this.tracks = this.tracks.filter(t => {
        const ttl = t.locked ? this.maxLostLocked : this.maxLostUnlocked;
        return t.lostFrames <= ttl;
      });
    }

    /** Create track from detection */
    _createTrackFromDet(d, feature, locked) {
      const id = this.nextId++;
      const t = new Track(id, { x: d.x, y: d.y, w: d.w, h: d.h }, { locked });
      if (feature) t.feature = feature;
      // Store object class information for display
      // 存储对象类别信息用于显示
      t.class = d.class || 'object';
      this.tracks.push(t);
      return id;
    }

    /** Public: lock a track by clicking a point on canvas */
    lockFromPoint(x, y, detections, ctx) {
      if (!detections || detections.length === 0) return null;
      // find detections containing point
      const candidates = detections
        .map((d, i) => ({ i, d, box: { x: d.bbox[0] ?? d.x, y: d.bbox[1] ?? d.y, w: d.bbox[2] ?? d.w, h: d.bbox[3] ?? d.h } }))
        .filter(o => x >= o.box.x && y >= o.box.y && x <= o.box.x + o.box.w && y <= o.box.y + o.box.h)
        .sort((a, b) => (b.d.score || 0) - (a.d.score || 0));

      let chosen = candidates[0];
      // fallback: nearest center within 40px if no box contains point
      if (!chosen) {
        let best = null, bestDist = 40;
        for (let i = 0; i < detections.length; i++) {
          const d = detections[i];
          const box = { x: d.bbox?.[0] ?? d.x, y: d.bbox?.[1] ?? d.y, w: d.bbox?.[2] ?? d.w, h: d.bbox?.[3] ?? d.h };
          const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
          const dist = Math.hypot(cx - x, cy - y);
          if (dist < bestDist) { bestDist = dist; best = { i, d, box }; }
        }
        if (best) chosen = best;
      }

      if (!chosen) return null;

      // If an existing track already overlaps strongly, lock it
      for (const t of this.tracks) {
        const i = iou(t.bbox, chosen.box);
        if (i > 0.4 || centerDistance(t.bbox, chosen.box) < 20) {
          t.locked = true; t.lostFrames = 0; return t.id;
        }
      }

      // Otherwise create a new locked track
      const feature = (this.enableReID && ctx) ? this.encoder.extract(ctx, chosen.box) : null;
      return this._createTrackFromDet({ ...chosen.box, class: chosen.d.class || 'object' }, feature, true);
    }

    /** Unlock and optionally remove a track */
    unlock(id, remove = false) {
      const idx = this.tracks.findIndex(t => t.id === id);
      if (idx < 0) return false;
      if (remove) { this.tracks.splice(idx, 1); return true; }
      this.tracks[idx].locked = false; return true;
    }

    /** Clear all tracks */
    clear() { this.tracks = []; this.nextId = 1; }

    /** Get shallow copy of tracks */
    getTracks() { return this.tracks.slice(); }
  }

  // expose to window
  window.Tracker = Tracker;
})();
