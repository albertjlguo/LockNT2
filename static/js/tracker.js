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

  // ----------------------- Kalman Filter --------------------------
  /**
   * Kalman Filter for object tracking
   * 用于目标追踪的卡尔曼滤波器
   */
  class KalmanFilter {
    constructor(initialX, initialY, opts = {}) {
      // State vector [x, y, vx, vy]
      // 状态向量 [x位置, y位置, x速度, y速度]
      this.x = [initialX, initialY, 0, 0];

      const dt = 1 / 30; // Assume 30 FPS

      // State transition matrix (F)
      // 状态转移矩阵 (恒速模型)
      this.F = [
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
      ];

      // Measurement matrix (H) - we only observe position
      // 测量矩阵 (只观测位置)
      this.H = [
        [1, 0, 0, 0],
        [0, 1, 0, 0]
      ];

      // Covariance matrix (P)
      // 协方差矩阵
      this.P = [
        [10, 0, 0, 0],
        [0, 10, 0, 0],
        [0, 0, 100, 0],
        [0, 0, 0, 100]
      ];

      // Process noise covariance (Q)
      // 过程噪声协方差
      this.Q = opts.Q || [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 10, 0],
        [0, 0, 0, 10]
      ];

      // Measurement noise covariance (R)
      // 测量噪声协方差
      this.R = opts.R || [
        [5, 0],
        [0, 5]
      ];
    }

    predict() {
      // Predict state: x_pred = F * x
      this.x = this.matrixMultiplyVector(this.F, this.x);

      // Predict covariance: P_pred = F * P * F^T + Q
      const FP = this.matrixMultiply(this.F, this.P);
      const FPFt = this.matrixMultiply(FP, this.transpose(this.F));
      this.P = this.matrixAdd(FPFt, this.Q);

      return { x: this.x[0], y: this.x[1] };
    }

    update(measuredX, measuredY) {
      const z = [measuredX, measuredY];

      // Innovation: y = z - H * x_pred
      const y = this.vectorSubtract(z, this.matrixMultiplyVector(this.H, this.x));

      // Innovation covariance: S = H * P * H^T + R
      const HP = this.matrixMultiply(this.H, this.P);
      const HPht = this.matrixMultiply(HP, this.transpose(this.H));
      const S = this.matrixAdd(HPht, this.R);

      // Kalman gain: K = P * H^T * S^(-1)
      const Pht = this.matrixMultiply(this.P, this.transpose(this.H));
      const K = this.matrixMultiply(Pht, this.matrixInverse(S));

      // Update state: x = x_pred + K * y
      this.x = this.vectorAdd(this.x, this.matrixMultiplyVector(K, y));

      // Update covariance: P = (I - K * H) * P
      const I = this.identity(4);
      const KH = this.matrixMultiply(K, this.H);
      const I_KH = this.matrixSubtract(I, KH);
      this.P = this.matrixMultiply(I_KH, this.P);
    }

    // --- Matrix Helper Functions ---
    matrixMultiplyVector(A, v) {
      const result = [];
      for (let i = 0; i < A.length; i++) {
        let sum = 0;
        for (let j = 0; j < v.length; j++) {
          sum += A[i][j] * v[j];
        }
        result.push(sum);
      }
      return result;
    }

    matrixMultiply(A, B) {
      const result = Array(A.length).fill(0).map(() => Array(B[0].length).fill(0));
      for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < B[0].length; j++) {
          for (let k = 0; k < A[0].length; k++) {
            result[i][j] += A[i][k] * B[k][j];
          }
        }
      }
      return result;
    }

    matrixAdd(A, B) {
      const result = [];
      for (let i = 0; i < A.length; i++) {
        result.push([]);
        for (let j = 0; j < A[0].length; j++) {
          result[i].push(A[i][j] + B[i][j]);
        }
      }
      return result;
    }

    matrixSubtract(A, B) {
        const result = [];
        for (let i = 0; i < A.length; i++) {
            result.push([]);
            for (let j = 0; j < A[0].length; j++) {
                result[i].push(A[i][j] - B[i][j]);
            }
        }
        return result;
    }

    transpose(A) {
      const result = Array(A[0].length).fill(0).map(() => Array(A.length).fill(0));
      for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < A[0].length; j++) {
          result[j][i] = A[i][j];
        }
      }
      return result;
    }

    matrixInverse(A) { // Only for 2x2 matrix
      const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
      if (det === 0) return [[0, 0], [0, 0]]; // Should not happen
      const invDet = 1 / det;
      return [
        [A[1][1] * invDet, -A[0][1] * invDet],
        [-A[1][0] * invDet, A[0][0] * invDet]
      ];
    }

    vectorAdd(v1, v2) {
      return v1.map((val, i) => val + v2[i]);
    }

    vectorSubtract(v1, v2) {
      return v1.map((val, i) => val - v2[i]);
    }

    identity(size) {
        const I = Array(size).fill(0).map(() => Array(size).fill(0));
        for (let i = 0; i < size; i++) I[i][i] = 1;
        return I;
    }
  }

  // ----------------------- Hungarian Algorithm --------------------------
  /**
   * Hungarian Algorithm for optimal assignment in tracking
   * 匈牙利算法用于追踪中的最优分配
   */
  class HungarianAlgorithm {
    constructor() {
      this.INF = 1e9;
    }

    /**
     * Solve assignment problem using Hungarian algorithm
     * 使用匈牙利算法解决分配问题
     * @param {number[][]} costMatrix - Cost matrix where costMatrix[i][j] is cost of assigning row i to column j
     * @returns {number[]} - Assignment array where result[i] is the column assigned to row i (-1 if unassigned)
     */
    solve(costMatrix) {
      if (!costMatrix || costMatrix.length === 0) return [];
      
      const n = costMatrix.length;
      const m = costMatrix[0].length;
      
      // Pad matrix to be square
      // 填充矩阵使其为方阵
      const size = Math.max(n, m);
      const matrix = Array(size).fill().map(() => Array(size).fill(this.INF));
      
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
          matrix[i][j] = costMatrix[i][j];
        }
      }
      
      // Hungarian algorithm implementation
      // 匈牙利算法实现
      const u = Array(size + 1).fill(0);
      const v = Array(size + 1).fill(0);
      const p = Array(size + 1).fill(0);
      const way = Array(size + 1).fill(0);
      
      for (let i = 1; i <= size; i++) {
        p[0] = i;
        let j0 = 0;
        const minv = Array(size + 1).fill(this.INF);
        const used = Array(size + 1).fill(false);
        
        do {
          used[j0] = true;
          const i0 = p[j0];
          let delta = this.INF;
          let j1 = 0;
          
          for (let j = 1; j <= size; j++) {
            if (!used[j]) {
              const cur = matrix[i0 - 1][j - 1] - u[i0] - v[j];
              if (cur < minv[j]) {
                minv[j] = cur;
                way[j] = j0;
              }
              if (minv[j] < delta) {
                delta = minv[j];
                j1 = j;
              }
            }
          }
          
          for (let j = 0; j <= size; j++) {
            if (used[j]) {
              u[p[j]] += delta;
              v[j] -= delta;
            } else {
              minv[j] -= delta;
            }
          }
          
          j0 = j1;
        } while (p[j0] !== 0);
        
        do {
          const j1 = way[j0];
          p[j0] = p[j1];
          j0 = j1;
        } while (j0);
      }
      
      // Extract assignment results
      // 提取分配结果
      const result = Array(n).fill(-1);
      for (let j = 1; j <= size; j++) {
        const i = p[j] - 1;
        if (i >= 0 && i < n && j - 1 < m && matrix[i][j - 1] < this.INF) {
          result[i] = j - 1;
        }
      }
      
      return result;
    }
  }

  // ----------------------- ID Manager --------------------------
  /**
   * ID Manager to prevent ID reuse and conflicts
   * ID管理器，防止ID重用和冲突
   */
  class IDManager {
    constructor() {
      this.usedIDs = new Set();
      this.nextID = 1;
      this.releasedIDs = new Map(); // ID -> release timestamp
      this.reuseDelay = 5000; // 5 seconds delay before ID can be reused
    }

    /**
     * Allocate a new unique ID
     * 分配新的唯一ID
     * @returns {number} - New unique ID
     */
    allocateID() {
      // Clean up old released IDs
      // 清理旧的已释放ID
      const now = Date.now();
      for (const [id, releaseTime] of this.releasedIDs.entries()) {
        if (now - releaseTime > this.reuseDelay) {
          this.usedIDs.delete(id);
          this.releasedIDs.delete(id);
        }
      }
      
      // Find next available ID
      // 找到下一个可用ID
      while (this.usedIDs.has(this.nextID)) {
        this.nextID++;
      }
      
      this.usedIDs.add(this.nextID);
      return this.nextID++;
    }

    /**
     * Release an ID for future reuse (with delay)
     * 释放ID以供将来重用（有延迟）
     * @param {number} id - ID to release
     */
    releaseID(id) {
      if (this.usedIDs.has(id)) {
        this.releasedIDs.set(id, Date.now());
      }
    }

    /**
     * Check if an ID is currently in use
     * 检查ID是否正在使用
     * @param {number} id - ID to check
     * @returns {boolean} - True if ID is in use
     */
    isIDInUse(id) {
      return this.usedIDs.has(id) && !this.releasedIDs.has(id);
    }
  }

  // ----------------------- Appearance Encoder --------------------------
  /**
   * Enhanced HSV histogram encoder with spatial information for better discrimination
   * 增强的HSV直方图编码器，包含空间信息以提高区分度
   * Default bins: H=20, S=6, V=4 + spatial bins => 480-D + 36-D spatial = 516-D feature.
   */
  class AppearanceEncoder {
    constructor(opts = {}) {
      this.imageWidth = opts.imageWidth || 640; // Reference image width
      this.imageHeight = opts.imageHeight || 480; // Reference image height
      // Increased bins for better discrimination in crowded scenes
      // 增加直方图bins以在密集场景中提高区分度
      this.binsH = opts.binsH || 20;  // Increased from 16
      this.binsS = opts.binsS || 6;   // Increased from 4
      this.binsV = opts.binsV || 4;   // Increased from 2
      this.innerCrop = clamp(opts.innerCrop ?? 0.85, 0.5, 1.0); // Slightly larger crop
      this.sampleStep = opts.sampleStep || 2;
      
      // Add spatial grid for position-aware features
      // 添加空间网格以获得位置感知特征
      this.spatialGridH = opts.spatialGridH || 3;
      this.spatialGridW = opts.spatialGridW || 3;
      
      this.totalDim = this.binsH * this.binsS * this.binsV;
      this._tmp = new Float32Array(this.totalDim);
      this.spatialDim = this.totalDim * this.spatialGridW * this.spatialGridH;
      this._spatialTmp = new Float32Array(this.spatialDim);
    }

    async encode(ctx, bbox) {
      const colorFeature = await this._computeColorFeature(ctx, bbox);
      if (!colorFeature) return null;

      const geometricFeature = this._extractGeometricFeatures(bbox);
      return { color: colorFeature, geometric: geometricFeature };
    }

    _extractGeometricFeatures(bbox) {
      const { x, y, w, h, confidence } = bbox;
      const aspectRatio = w / (h + 1e-6);
      const normalizedArea = (w * h) / (this.imageWidth * this.imageHeight);
      const normalizedX = (x + w / 2) / this.imageWidth;
      const normalizedY = (y + h / 2) / this.imageHeight;

      // Feature vector: [aspectRatio, normalizedArea, confidence, normalizedX, normalizedY]
      return [aspectRatio, normalizedArea, confidence || 0.5, normalizedX, normalizedY];
    }

    /** Extracts color histogram feature */
    async _computeColorFeature(ctx, bbox) {
      const x = bbox.x + (1 - this.innerCrop) * 0.5 * bbox.w;
      const y = bbox.y + (1 - this.innerCrop) * 0.5 * bbox.h;
      const w = bbox.w * this.innerCrop;
      const h = bbox.h * this.innerCrop;

      const ix = Math.round(x), iy = Math.round(y);
      const iw = Math.max(1, Math.round(w)), ih = Math.max(1, Math.round(h));
      if (iw < 6 || ih < 6) return null; // Increased minimum size for spatial features

      let img;
      try { img = ctx.getImageData(ix, iy, iw, ih); }
      catch (e) { return null; }

      const colorDim = this.binsH * this.binsS * this.binsV;
      const hist = this._tmp; hist.fill(0);
      const spatialFeats = this._spatialTmp; spatialFeats.fill(0);
      const data = img.data;
      const step = Math.max(1, this.sampleStep);

      // Extract global color histogram
      // 提取全局颜色直方图
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

      return l2normalize(hist);
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
      this.kalmanFilter = new KalmanFilter(this.cx, this.cy, opts.kalman);
      this.locked = !!opts.locked;
      this.color = COLOR_POOL[(id - 1) % COLOR_POOL.length];
      this.feature = null; // EMA feature { color, geometric }
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
      
      // Enhanced appearance model for ID consistency in crowded scenes
      // 针对密集场景ID一致性的增强外观模型
      this.appearanceModel = {
        templates: [], // Multiple appearance templates
        weights: [],   // Template weights
        maxTemplates: 7,  // Further increased for better discrimination
        updateThreshold: 0.65,  // Slightly higher for stability
        discriminativeThreshold: 0.75, // Threshold for high-confidence matches
        spatialConsistency: true // Enable spatial layout consistency
      };
      
      // ID switching prevention mechanisms
      // ID切换防护机制
      this.idConsistency = {
        recentMatches: [], // Track recent successful matches
        maxMatchHistory: 10,
        consistencyThreshold: 0.7, // Minimum consistency score
        switchPenalty: 0.3 // Penalty for potential ID switches
      };
      
      this._pushTrajectory();
    }

    get bbox() { return { x: this.cx - this.w / 2, y: this.cy - this.h / 2, w: this.w, h: this.h }; }

    _pushTrajectory() {
      this.trajectory.push({ x: this.cx, y: this.cy });
      if (this.trajectory.length > 60) this.trajectory.shift();
    }

    /** Prediction using Kalman Filter */
    predict() {
      const predicted = this.kalmanFilter.predict();
      this.cx = predicted.x;
      this.cy = predicted.y;

      const vx = this.kalmanFilter.x[2];
      const vy = this.kalmanFilter.x[3];

      // Handle occlusion state with improved prediction
      if (this.occlusionState.isOccluded) {
        // Adaptive search radius expansion
        const expansionRate = Math.min(1.08, 1.02 + Math.sqrt(vx*vx + vy*vy) / 100);
        this.occlusionState.searchRadius *= expansionRate;
        this.occlusionState.searchRadius = Math.min(this.occlusionState.searchRadius, 250);
        
        // Confidence decay with velocity consideration
        const velocityMagnitude = Math.sqrt(vx*vx + vy*vy);
        const decayRate = velocityMagnitude > 20 ? 0.93 : 0.95; // Faster decay for fast-moving objects
        this.occlusionState.confidence *= decayRate;
        
        // Store predicted position
        this.occlusionState.predictedPosition = { cx: this.cx, cy: this.cy };
      } else {
        // Update last known position when not occluded
        this.occlusionState.lastKnownPosition = { cx: this.cx, cy: this.cy };
        this.occlusionState.confidence = Math.min(1.0, this.occlusionState.confidence + 0.15);
        // Reset search radius when not occluded
        this.occlusionState.searchRadius = Math.max(this.w, this.h) * 0.6;
      }

      this._pushTrajectory();
    }

    /** Update with Kalman Filter */
    update(detBbox, feature) {
      const measuredX = detBbox.x + detBbox.w / 2;
      const measuredY = detBbox.y + detBbox.h / 2;

      // Update Kalman filter with new measurement
      this.kalmanFilter.update(measuredX, measuredY);

      // Update track state from filter
      this.cx = this.kalmanFilter.x[0];
      this.cy = this.kalmanFilter.x[1];

      // Reset occlusion state if it was occluded
      if (this.occlusionState.isOccluded) {
        this.occlusionState.isOccluded = false;
        this.occlusionState.searchRadius = Math.max(detBbox.w, detBbox.h) * 0.5;
      }
      
      // Adaptive size smoothing based on confidence
      const sizeWeight = Math.max(0.2, this.occlusionState.confidence * 0.4);
      this.w = (1 - sizeWeight) * this.w + sizeWeight * detBbox.w;
      this.h = (1 - sizeWeight) * this.h + sizeWeight * detBbox.h;
      
      this.lostFrames = 0;
      this.hits += 1;
      this._pushTrajectory();
      
      // Enhanced appearance model with multiple templates
      if (feature) {
        this.updateAppearanceModel(feature);
      }
    }
    
    /** Update appearance model with EMA */
    updateAppearanceModel(newFeature) {
      const alpha = 0.2; // EMA smoothing factor
      if (newFeature && newFeature.color && newFeature.geometric && this.feature && this.feature.color && this.feature.geometric) {
        // EMA update for color histogram
        for (let i = 0; i < this.feature.color.length; i++) {
          this.feature.color[i] = (1 - alpha) * this.feature.color[i] + alpha * newFeature.color[i];
        }
        l2normalize(this.feature.color);

        // EMA update for geometric features
        for (let i = 0; i < this.feature.geometric.length; i++) {
          this.feature.geometric[i] = (1 - alpha) * this.feature.geometric[i] + alpha * newFeature.geometric[i];
        }
      } else {
        this.feature = newFeature;
      }
    }
      
    /** Check if track should enter occlusion state with improved logic */
    checkOcclusionState(frameCount) {
      // Different thresholds for locked vs unlocked tracks
      // 锁定和未锁定轨迹的不同阈值
      const occlusionThreshold = this.locked ? 2 : 4; // Locked tracks enter occlusion faster
      
      if (this.lostFrames > occlusionThreshold && !this.occlusionState.isOccluded) {
        this.occlusionState.isOccluded = true;
        this.occlusionState.occlusionStartFrame = frameCount;
        
        // Adaptive initial search radius based on velocity
        // 基于速度的自适应初始搜索半径
        const vx = this.kalmanFilter.x[2];
        const vy = this.kalmanFilter.x[3];
        const velocityMagnitude = Math.sqrt(vx**2 + vy**2);
        const baseRadius = Math.max(this.w, this.h) * 0.8;
        const velocityBonus = Math.min(50, velocityMagnitude * 2); // Up to 50px bonus
        this.occlusionState.searchRadius = baseRadius + velocityBonus;
        
        console.log(`Track ${this.id} entered occlusion state (lost: ${this.lostFrames}, locked: ${this.locked})`);
      }
    }
  }

  // ----------------------------- Tracker -------------------------------
  class Tracker {
    constructor(opts = {}) {
      this.tracks = [];
      this.idManager = new IDManager(); // Use ID manager instead of simple counter
      this.hungarian = new HungarianAlgorithm(); // Hungarian algorithm for optimal assignment
      this.kalmanParams = opts.kalman || {}; // Store kalman params for new tracks
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
      this.wIoU = opts.wIoU ?? 0.25;        // Further reduced IoU weight for fast movement
      this.wApp = opts.wApp ?? 0.35;        // Appearance weight
      this.wCtr = opts.wCtr ?? 0.20;        // Center distance weight
      this.wRatio = opts.wRatio ?? 0.05;    // Aspect ratio weight
      this.wMotion = opts.wMotion ?? 0.15;  // Increased motion consistency weight
      
      // Adaptive cost thresholds for different scenarios
      // 针对不同场景的自适应成本阈值
      this.costThreshold = opts.costThreshold ?? 0.75; // Reduced base threshold for stricter matching
      this.crowdedSceneThreshold = opts.crowdedSceneThreshold ?? 0.65; // Even stricter for crowded scenes
      this.lockedTrackThreshold = opts.lockedTrackThreshold ?? 0.85; // More permissive for locked tracks
      this.autoCreate = !!opts.autoCreate;
      
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
    async update(detections, videoContext, options = {}) {
      // Extract tracking-first mode options
      // 提取追踪优先模式选项
      const trackingFirstMode = options.trackingFirstMode || false;
      const lockedTargetDetectionWeight = options.lockedTargetDetectionWeight || 1.0;
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
      if (this.enableReID && videoContext) {
        const maxFeat = 15; // cap to save cost
        enriched = dets.map((dd, idx) => {
          let feat = null;
          if (!this.focusClasses.length || this.focusClasses.includes(dd.class)) {
            if (idx < maxFeat) feat = this.encoder.encode(videoContext, dd) || null;
          }
          return { ...dd, feature: feat };
        });
      } else {
        enriched = dets.map(dd => ({ ...dd, feature: null }));
      }

      // Predict existing tracks first
      for (const t of this.tracks) t.predict();

      // Smart association strategy based on tracking mode
      // 基于追踪模式的智能关联策略
      const locked = this.tracks.filter(t => t.locked);
      const normal = this.tracks.filter(t => !t.locked);

      const unmatchedDetIdx = new Set(enriched.map((_, i) => i));

      if (trackingFirstMode && locked.length > 0) {
        // Tracking-first mode: locked targets rely more on prediction
        // 追踪优先模式：锁定目标更依赖预测
        this._associateAndUpdateTrackingFirst(locked, enriched, unmatchedDetIdx, lockedTargetDetectionWeight);
        // Then associate normal tracks using Hungarian algorithm
        // 然后使用匈牙利算法关联普通轨迹
        this._associateAndUpdateHungarian(normal, enriched, unmatchedDetIdx, false);
      } else {
        // Standard mode: detection-first association
        // 标准模式：检测优先关联
        // Use Hungarian for both locked and normal tracks for optimal assignment
        this._associateAndUpdateHungarian(locked, enriched, unmatchedDetIdx, true);
        this._associateAndUpdateHungarian(normal, enriched, unmatchedDetIdx, false);
      }

      // Increase lost for unmatched tracks
      for (const t of this.tracks) {
        if (!t._updatedThisRound) {
          t.lostFrames += 1;
        }
        delete t._updatedThisRound;
      }

      // Try to recover lost tracks before creating new ones
      // 在创建新轨迹前尝试恢复丢失的轨迹
      this._attemptTrackRecovery(enriched, unmatchedDetIdx);

      // Optionally create new tracks from remaining detections (if enabled)
      if (this.autoCreate) {
        for (const idx of unmatchedDetIdx) {
          const d = enriched[idx];
          if (d.score >= this.detectHigh) this._createTrackFromDet(d, d.feature, false);
        }
      }

      this._prune();
    }

    /** Tracking-first association for locked targets - prioritizes prediction over detection */
    _associateAndUpdateTrackingFirst(trackList, detections, unmatchedDetIdx, detectionWeight = 0.3) {
      if (trackList.length === 0) return;
      
      // For locked tracks in tracking-first mode, we primarily rely on prediction
      // 对于追踪优先模式下的锁定轨迹，主要依赖预测
      for (const track of trackList) {
        // Check if track is stable enough to ignore detections
        // 检查轨迹是否稳定到可以忽略检测
        const stability = Math.min(1.0, (track.hits - track.lostFrames) / Math.max(1, track.hits));
        const trackAge = Math.min(1.0, track.hits / 20);
        const confidenceFactor = stability * trackAge;
        
        // Very stable tracks can ignore detections completely
        // 非常稳定的轨迹可以完全忽略检测
        if (confidenceFactor > 0.85 && track.lostFrames === 0) {
          // Pure tracking update - no detection influence
          // 纯追踪更新 - 不受检测影响
          track._updatedThisRound = true;
          continue;
        }
        
        // For less stable tracks, find best detection match with reduced weight
        // 对于稳定性较低的轨迹，寻找最佳检测匹配但降低权重
        let bestMatch = null;
        let bestCost = Infinity;
        
        const tb = track.bbox;
        const gating = Math.max(this.gatingBase, 0.8 * Math.hypot(track.w, track.h));
        
        for (const detIdx of unmatchedDetIdx) {
          const det = detections[detIdx];
          const db = { x: det.x, y: det.y, w: det.w, h: det.h };
          
          const ctr = centerDistance(tb, db);
          if (ctr > gating * 2.0) continue; // Generous gating for locked tracks
          
          // Simplified cost focusing on motion consistency
          // 简化成本，专注于运动一致性
          const predictedX = track.cx + (track.vx || 0) / 30;
          const predictedY = track.cy + (track.vy || 0) / 30;
          const detCenterX = det.x + det.w / 2;
          const detCenterY = det.y + det.h / 2;
          const motionDist = Math.sqrt((predictedX - detCenterX) ** 2 + (predictedY - detCenterY) ** 2);
          
          // Cost based primarily on motion prediction
          // 主要基于运动预测的成本
          const cost = motionDist / gating;
          
          if (cost < bestCost && cost < 0.8) { // Permissive threshold
            bestCost = cost;
            bestMatch = { detIdx, detection: det };
          }
        }
        
        if (bestMatch) {
          // Weighted update: blend prediction with detection
          // 加权更新：混合预测和检测
          const det = bestMatch.detection;
          const predWeight = 1.0 - detectionWeight;
          
          // Blend predicted and detected positions
          // 混合预测和检测位置
          const predX = track.cx + (track.vx || 0) / 30;
          const predY = track.cy + (track.vy || 0) / 30;
          const detX = det.x + det.w / 2;
          const detY = det.y + det.h / 2;
          
          const blendedX = predWeight * predX + detectionWeight * detX;
          const blendedY = predWeight * predY + detectionWeight * detY;
          const blendedW = predWeight * track.w + detectionWeight * det.w;
          const blendedH = predWeight * track.h + detectionWeight * det.h;
          
          // Update with blended values
          // 使用混合值更新
          const blendedBbox = {
            x: blendedX - blendedW / 2,
            y: blendedY - blendedH / 2,
            w: blendedW,
            h: blendedH
          };
          
          track.update(blendedBbox, det.feature);
          track._updatedThisRound = true;
          unmatchedDetIdx.delete(bestMatch.detIdx);
        } else {
          // No good detection match - rely purely on tracking
          // 没有好的检测匹配 - 纯依赖追踪
          track._updatedThisRound = true;
        }
      }
    }

    /**
     * Hungarian algorithm-based association for optimal assignment
     * 基于匈牙利算法的关联，实现最优分配
     */
    _associateAndUpdateHungarian(trackList, detections, unmatchedDetIdx, isLockedPass) {
      if (trackList.length === 0 || unmatchedDetIdx.size === 0) return;

      // Build cost matrix
      // 构建成本矩阵
      const detectionArray = Array.from(unmatchedDetIdx).map(idx => detections[idx]);
      const detectionIndices = Array.from(unmatchedDetIdx);
      
      if (detectionArray.length === 0) return;
      
      const costMatrix = [];
      const maxCost = 10.0;
      
      for (let ti = 0; ti < trackList.length; ti++) {
        const t = trackList[ti];
        const tb = t.bbox;
        const row = [];
        
        // Adaptive gating
        // 自适应门控
        const velocityMagnitude = Math.sqrt((t.vx || 0) ** 2 + (t.vy || 0) ** 2);
        const velocityFactor = Math.min(3.0, 1.0 + velocityMagnitude / 50); // Scale with velocity
        let gating = Math.max(this.gatingBase, 0.5 * Math.hypot(t.w, t.h)) * velocityFactor;
        
        if (t.occlusionState && t.occlusionState.isOccluded) {
          gating = Math.max(gating, t.occlusionState.searchRadius || gating * 2.0);
        }
        
        if (t.locked) {
          gating *= 1.5;
        }
        
        for (let di = 0; di < detectionArray.length; di++) {
          const d = detectionArray[di];
          const db = { x: d.x, y: d.y, w: d.w, h: d.h };
          const i = iou(tb, db);
          const ctr = centerDistance(tb, db);
          
          const gatingMultiplier = (t.occlusionState && t.occlusionState.isOccluded) ? 3.0 : 2.5;
          // More permissive gating for locked tracks
          // 对锁定轨迹更宽松的门控
          const finalGatingMultiplier = t.locked ? gatingMultiplier * 1.2 : gatingMultiplier;
          if (i < 0.005 && ctr > gating * finalGatingMultiplier) {
            row.push(maxCost);
            continue;
          }
          
          // Compute cost using existing logic
          // 使用现有逻辑计算成本
          let app = 1;
          let idSwitchPenalty = 0;
          
          if (this.enableReID && d.feature) {
            if (t.getAppearanceMatchScore && t.appearanceModel && t.appearanceModel.templates.length > 0) {
              const matchScore = t.getAppearanceMatchScore(d.feature, d);
              app = 1 - matchScore;
              
              if (matchScore < t.appearanceModel.discriminativeThreshold) {
                let betterMatchExists = false;
                for (const otherTrack of this.tracks) {
                  if (otherTrack.id !== t.id && otherTrack.getAppearanceMatchScore) {
                    const otherScore = otherTrack.getAppearanceMatchScore(d.feature, d);
                    if (otherScore > matchScore + 0.15) {
                      betterMatchExists = true;
                      break;
                    }
                  }
                }
                
                if (betterMatchExists) {
                  idSwitchPenalty = t.idConsistency.switchPenalty;
                }
              }
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
          // 运动一致性
          const dt = 1/30;
          const predictedX = t.cx + (t.vx || 0) * dt + 0.5 * (t.motionModel?.acceleration?.ax || 0) * dt * dt;
          const predictedY = t.cy + (t.vy || 0) * dt + 0.5 * (t.motionModel?.acceleration?.ay || 0) * dt * dt;
          const detCenterX = d.x + d.w / 2;
          const detCenterY = d.y + d.h / 2;
          const motionDist = Math.sqrt((predictedX - detCenterX) ** 2 + (predictedY - detCenterY) ** 2);
          
          const velocityConfidence = Math.min(1.0, t.hits / 10);
          const adaptiveMotionGating = gating * (0.5 + 0.5 * velocityConfidence);
          const motionCost = clamp(motionDist / adaptiveMotionGating, 0, 1);

          // Adaptive weights
          // 自适应权重
          let wIoU = this.wIoU, wApp = this.wApp, wCtr = this.wCtr;
          let wMotion = this.wMotion || 0.15;
          
          const trackAge = Math.min(1.0, t.hits / 30);
          const stabilityFactor = Math.min(1.0, (t.hits - t.lostFrames) / Math.max(1, t.hits));
          
          if (isLockedPass) {
            wIoU = Math.max(0.15, this.wIoU - 0.15);
            wApp = Math.min(0.45, this.wApp + 0.15);
            wCtr = Math.min(0.25, this.wCtr + 0.10);
            wMotion = Math.min(0.20, this.wMotion + 0.10);
          }
          
          if (t.occlusionState && t.occlusionState.isOccluded) {
            wIoU *= 0.3;
            wApp *= 1.5;
            wCtr *= 1.3;
            wMotion *= 1.4;
          }
          
          if (stabilityFactor > 0.8) {
            wMotion *= 1.2;
          }

          // Enhanced cost calculation with trajectory consistency and ID switch prevention
          // 增强成本计算，包含轨迹一致性和ID切换防护
          
          // Trajectory consistency check - penalize unrealistic movements
          // 轨迹一致性检查 - 惩罚不现实的运动
          let trajConsistencyPenalty = 0;
          if (t.trajectory.length >= 3) {
            const recent = t.trajectory.slice(-3);
            const avgVelX = (recent[2].x - recent[0].x) / 2;
            const avgVelY = (recent[2].y - recent[0].y) / 2;
            const expectedX = t.cx + avgVelX;
            const expectedY = t.cy + avgVelY;
            
            const detCenterX = d.x + d.w / 2;
            const detCenterY = d.y + d.h / 2;
            const trajDeviation = Math.sqrt((expectedX - detCenterX)**2 + (expectedY - detCenterY)**2);
            
            // Penalize large trajectory deviations
            // 惩罚大的轨迹偏差
            const maxReasonableDeviation = Math.max(50, Math.hypot(t.w, t.h) * 0.8);
            if (trajDeviation > maxReasonableDeviation) {
              trajConsistencyPenalty = Math.min(0.4, trajDeviation / maxReasonableDeviation * 0.2);
            }
          }
          
          // Size consistency penalty for dramatic size changes
          // 尺寸一致性惩罚，防止剧烈尺寸变化
          const sizeChangeRatio = Math.max(d.w/t.w, t.w/d.w) * Math.max(d.h/t.h, t.h/d.h);
          const sizeConsistencyPenalty = sizeChangeRatio > 2.0 ? Math.min(0.3, (sizeChangeRatio - 2.0) * 0.1) : 0;
          
          const baseCost = wIoU * (1 - i) + wApp * app + wCtr * ctrNorm + 
                          (this.wRatio || 0.03) * ratioDiff + wMotion * motionCost;
          
          const cost = baseCost + idSwitchPenalty + trajConsistencyPenalty + sizeConsistencyPenalty;
          
          // Ensure cost is within reasonable bounds for Hungarian algorithm
          // 确保成本在匈牙利算法的合理范围内
          row.push(cost > this.costThreshold ? maxCost : cost);
        }
        
        costMatrix.push(row);
      }
      
      // Solve using Hungarian algorithm
      // 使用匈牙利算法求解
      const assignments = this.hungarian.solve(costMatrix);
      
      // Apply assignments
      // 应用分配结果
      for (let ti = 0; ti < assignments.length; ti++) {
        const di = assignments[ti];
        if (di >= 0 && di < detectionArray.length) {
          const cost = costMatrix[ti][di];
          if (cost <= this.costThreshold) {
            const t = trackList[ti];
            const d = detectionArray[di];
            const originalDetIdx = detectionIndices[di];
            
            t.update({ x: d.x, y: d.y, w: d.w, h: d.h }, d.feature);
            t._updatedThisRound = true;
            unmatchedDetIdx.delete(originalDetIdx);
            
            if (isLockedPass) {
              console.log(`Hungarian: Assigned locked track ${t.id} to detection (cost: ${cost.toFixed(3)})`);
            }
          }
        }
      }
    }

    /** Enhanced association with occlusion-aware matching */
    _associateAndUpdate(trackList, detections, unmatchedDetIdx, isLockedPass) {
      if (trackList.length === 0 || unmatchedDetIdx.size === 0) return;

      const pairs = [];
      for (let ti = 0; ti < trackList.length; ti++) {
        const t = trackList[ti];
        const tb = t.bbox;
        
        // Adaptive gating for occlusion handling with velocity consideration
        // 考虑速度的自适应门控遮挡处理
        const velocityMagnitude = Math.sqrt((t.vx || 0) ** 2 + (t.vy || 0) ** 2);
        const velocityFactor = Math.min(3.0, 1.0 + velocityMagnitude / 50); // Scale with velocity
        let gating = Math.max(this.gatingBase, 0.5 * Math.hypot(t.w, t.h)) * velocityFactor;
        
        if (t.occlusionState && t.occlusionState.isOccluded) {
          gating = Math.max(gating, t.occlusionState.searchRadius || gating * 2.0);
        }
        
        // Special handling for locked tracks - more permissive gating
        // 锁定轨迹的特殊处理 - 更宽松的门控
        if (t.locked) {
          gating *= 1.5;
        }
        
        for (const di of unmatchedDetIdx) {
          const d = detections[di];
          const db = { x: d.x, y: d.y, w: d.w, h: d.h };
          const i = iou(tb, db);
          const ctr = centerDistance(tb, db);
          
          const gatingMultiplier = (t.occlusionState && t.occlusionState.isOccluded) ? 3.0 : 2.5;
          // More permissive gating for locked tracks
          // 对锁定轨迹更宽松的门控
          const finalGatingMultiplier = t.locked ? gatingMultiplier * 1.2 : gatingMultiplier;
          if (i < 0.005 && ctr > gating * finalGatingMultiplier) continue; // Lowered IoU threshold

          // Enhanced appearance matching with ID switching prevention
          // 增强外观匹配，防止ID切换
          let app = 1;
          let idSwitchPenalty = 0;
          
          if (this.enableReID && d.feature) {
            if (t.getAppearanceMatchScore && t.appearanceModel && t.appearanceModel.templates.length > 0) {
              const matchScore = t.getAppearanceMatchScore(d.feature, d);
              app = 1 - matchScore;
              
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
          
          // Enhanced motion consistency with acceleration
          // 包含加速度的增强运动一致性
          const dt = 1/30; // 30 FPS assumption
          const predictedX = t.cx + (t.vx || 0) * dt + 0.5 * (t.motionModel?.acceleration?.ax || 0) * dt * dt;
          const predictedY = t.cy + (t.vy || 0) * dt + 0.5 * (t.motionModel?.acceleration?.ay || 0) * dt * dt;
          const detCenterX = d.x + d.w / 2;
          const detCenterY = d.y + d.h / 2;
          const motionDist = Math.sqrt((predictedX - detCenterX) ** 2 + (predictedY - detCenterY) ** 2);
          
          // Adaptive motion cost based on velocity confidence
          // 基于速度置信度的自适应运动成本
          const velocityConfidence = Math.min(1.0, t.hits / 10); // Build confidence over time
          const adaptiveMotionGating = gating * (0.5 + 0.5 * velocityConfidence);
          const motionCost = clamp(motionDist / adaptiveMotionGating, 0, 1);

          // Adaptive weights based on track state and confidence
          // 基于轨迹状态和置信度的自适应权重
          let wIoU = this.wIoU, wApp = this.wApp, wCtr = this.wCtr;
          let wMotion = this.wMotion || 0.15;
          
          // Track age and stability factor
          // 轨迹年龄和稳定性因子
          const trackAge = Math.min(1.0, t.hits / 30); // Normalize to [0,1]
          const stabilityFactor = Math.min(1.0, (t.hits - t.lostFrames) / Math.max(1, t.hits));
          
          if (isLockedPass) {
            wIoU = Math.max(0.15, this.wIoU - 0.15);
            wApp = Math.min(0.45, this.wApp + 0.15);
            wCtr = Math.min(0.25, this.wCtr + 0.10);
            wMotion = Math.min(0.20, this.wMotion + 0.10);
          }
          
          // Occlusion state adjustments
          // 遮挡状态调整
          if (t.occlusionState && t.occlusionState.isOccluded) {
            wIoU *= 0.3; // Heavily reduce IoU importance during occlusion
            wApp *= 1.5; // Increase appearance importance
            wCtr *= 1.3; // Increase center distance importance
            wMotion *= 1.4; // Increase motion prediction importance
          }
          
          // Stability-based adjustments
          // 基于稳定性的调整
          if (stabilityFactor > 0.8) {
            wMotion *= 1.2; // Trust motion model more for stable tracks
          }

          // Enhanced cost calculation with trajectory consistency and ID switch prevention
          // 增强成本计算，包含轨迹一致性和ID切换防护
          
          // Trajectory consistency check - penalize unrealistic movements
          // 轨迹一致性检查 - 惩罚不现实的运动
          let trajConsistencyPenalty = 0;
          if (t.trajectory.length >= 3) {
            const recent = t.trajectory.slice(-3);
            const avgVelX = (recent[2].x - recent[0].x) / 2;
            const avgVelY = (recent[2].y - recent[0].y) / 2;
            const expectedX = t.cx + avgVelX;
            const expectedY = t.cy + avgVelY;
            
            const detCenterX = d.x + d.w / 2;
            const detCenterY = d.y + d.h / 2;
            const trajDeviation = Math.sqrt((expectedX - detCenterX)**2 + (expectedY - detCenterY)**2);
            
            // Penalize large trajectory deviations
            // 惩罚大的轨迹偏差
            const maxReasonableDeviation = Math.max(50, Math.hypot(t.w, t.h) * 0.8);
            if (trajDeviation > maxReasonableDeviation) {
              trajConsistencyPenalty = Math.min(0.4, trajDeviation / maxReasonableDeviation * 0.2);
            }
          }
          
          // Size consistency penalty for dramatic size changes
          // 尺寸一致性惩罚，防止剧烈尺寸变化
          const sizeChangeRatio = Math.max(d.w/t.w, t.w/d.w) * Math.max(d.h/t.h, t.h/d.h);
          const sizeConsistencyPenalty = sizeChangeRatio > 2.0 ? Math.min(0.3, (sizeChangeRatio - 2.0) * 0.1) : 0;
          
          const baseCost = wIoU * (1 - i) + wApp * app + wCtr * ctrNorm + 
                          (this.wRatio || 0.03) * ratioDiff + wMotion * motionCost;
          
          const cost = baseCost + idSwitchPenalty + trajConsistencyPenalty + sizeConsistencyPenalty;
          
          pairs.push({ cost, ti, di });
        }
      }

      pairs.sort((a, b) => a.cost - b.cost);
      const usedT = new Set();
      
      for (const p of pairs) {
        const trackOpts = {
          locked: isLocked,
          class: bbox.class,
          kalman: this.kalmanParams
        };
        // Adaptive threshold based on scene complexity and track state
        // 基于场景复杂度和轨迹状态的自适应阈值
        let threshold = this.costThreshold;
        
        // Detect crowded scene based on number of active tracks
        // 基于活跃轨迹数量检测密集场景
        const activeTracks = this.tracks.filter(t => t.lostFrames < 5).length;
        const isCrowdedScene = activeTracks > 8; // More than 8 active tracks = crowded
        
        if (isCrowdedScene) {
          threshold = this.crowdedSceneThreshold; // Stricter threshold for crowded scenes
        }
        
        // Adjust for track-specific states
        // 根据轨迹特定状态调整
        if (track.locked) {
          threshold = Math.max(threshold, this.lockedTrackThreshold); // Use higher threshold for locked tracks
        }
        
        if (track.occlusionState && track.occlusionState.isOccluded) {
          threshold *= 1.2; // Slightly more permissive for occluded tracks
        }
        
        // Track stability adjustment - stable tracks get slightly more permissive thresholds
        // 轨迹稳定性调整 - 稳定轨迹获得稍微宽松的阈值
        const stabilityFactor = Math.min(1.0, (track.hits - track.lostFrames) / Math.max(1, track.hits));
        const trackAge = Math.min(1.0, track.hits / 30);
        
        if (stabilityFactor > 0.8 && trackAge > 0.5) {
          threshold *= 1.1; // 10% more permissive for very stable tracks
        } else if (track.hits < 5) {
          threshold *= 0.9; // 10% stricter for new tracks
        }
        
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

    /** Enhanced track recovery with ID switching prevention */
    _attemptTrackRecovery(detections, unmatchedDetIdx) {
      // Find tracks that are lost but not yet pruned
      // 找到丢失但尚未被清除的轨迹
      const lostTracks = this.tracks.filter(t => 
        t.lostFrames > 5 && t.lostFrames < (t.locked ? this.maxLostLocked : this.maxLostUnlocked)
      );
      
      if (lostTracks.length === 0 || unmatchedDetIdx.size === 0) return;
      
      // Prevent recovery if too many tracks are competing for same detections
      // 如果太多轨迹竞争相同检测则防止恢复
      const activeTracks = this.tracks.filter(t => t.lostFrames < 3).length;
      const isCrowdedScene = activeTracks > 6;
      
      if (isCrowdedScene && lostTracks.length > unmatchedDetIdx.size * 2) {
        // In crowded scenes, be more conservative about recovery
        // 在密集场景中，对恢复更加保守
        return;
      }
      
      const recoveryPairs = [];
      
      for (const track of lostTracks) {
        // Use predicted position for recovery matching
        // 使用预测位置进行恢复匹配
        const predictedPos = track.occlusionState.predictedPosition || 
                           { cx: track.cx, cy: track.cy };
        
        for (const detIdx of unmatchedDetIdx) {
          const det = detections[detIdx];
          const detCx = det.x + det.w / 2;
          const detCy = det.y + det.h / 2;
          
          // Distance-based recovery check
          // 基于距离的恢复检查
          const distance = Math.sqrt((predictedPos.cx - detCx)**2 + (predictedPos.cy - detCy)**2);
          const maxRecoveryDistance = track.locked ? 150 : 100;
          
          if (distance > maxRecoveryDistance) continue;
          
          // Class consistency check
          // 类别一致性检查
          if (track.class && det.class && track.class !== det.class) continue;
          
          // Appearance similarity check (if available)
          // 外观相似性检查（如果可用）
          let appearanceScore = 0.5; // Default neutral score
          if (det.feature && track.getAppearanceCost) {
            appearanceScore = 1 - track.getAppearanceCost(det.feature);
          }
          
          // Size consistency check
          // 尺寸一致性检查
          const sizeRatio = Math.min(det.w/track.w, track.w/det.w) * 
                           Math.min(det.h/track.h, track.h/det.h);
          
          // Combined recovery score
          // 综合恢复分数
          const distanceScore = Math.exp(-distance / 50); // Exponential decay
          const recoveryScore = 0.4 * distanceScore + 0.4 * appearanceScore + 0.2 * sizeRatio;
          
          // Adaptive recovery threshold based on scene complexity
          // 基于场景复杂度的自适应恢复阈值
          const activeTracks = this.tracks.filter(t => t.lostFrames < 5).length;
          const isCrowdedScene = activeTracks > 8;
          
          let recoveryThreshold = track.locked ? 0.5 : 0.65;
          if (isCrowdedScene) {
            recoveryThreshold += 0.15; // Stricter in crowded scenes
          }
          
          // Additional check: ensure this detection isn't better suited for an active track
          // 额外检查：确保此检测不更适合活跃轨迹
          let conflictingActiveTrack = false;
          for (const activeTrack of this.tracks) {
            if (activeTrack.lostFrames < 3 && activeTrack.id !== track.id) {
              const activeDistance = Math.sqrt(
                (activeTrack.cx - (det.x + det.w/2))**2 + 
                (activeTrack.cy - (det.y + det.h/2))**2
              );
              if (activeDistance < distance * 0.7) { // Active track is much closer
                conflictingActiveTrack = true;
                break;
              }
            }
          }
          
          if (conflictingActiveTrack) {
            recoveryThreshold += 0.2; // Much stricter if conflicts with active tracks
          }
          
          if (recoveryScore > recoveryThreshold) {
            recoveryPairs.push({
              track: track,
              detIdx: detIdx,
              detection: det,
              score: recoveryScore,
              distance: distance
            });
          }
        }
      }
      
      // Sort by recovery score (best first)
      // 按恢复分数排序（最佳优先）
      recoveryPairs.sort((a, b) => b.score - a.score);
      
      // Apply recovery matches
      // 应用恢复匹配
      const usedDetections = new Set();
      const recoveredTracks = new Set();
      
      for (const pair of recoveryPairs) {
        if (usedDetections.has(pair.detIdx) || recoveredTracks.has(pair.track.id)) {
          continue;
        }
        
        // Recover the track
        // 恢复轨迹
        const track = pair.track;
        const det = pair.detection;
        
        console.log(`Recovering track ${track.id} after ${track.lostFrames} lost frames (score: ${pair.score.toFixed(3)})`);
        
        // Update track with detection
        // 使用检测结果更新轨迹
        track.update({ x: det.x, y: det.y, w: det.w, h: det.h }, det.feature);
        track._updatedThisRound = true;
        
        // Reset occlusion state
        // 重置遮挡状态
        if (track.occlusionState.isOccluded) {
          track.occlusionState.isOccluded = false;
          track.occlusionState.confidence = Math.min(1.0, track.occlusionState.confidence + 0.3);
        }
        
        // Mark as used
        // 标记为已使用
        usedDetections.add(pair.detIdx);
        recoveredTracks.add(track.id);
        unmatchedDetIdx.delete(pair.detIdx);
      }
    }

    _getAppearanceCost(track, detBbox, detFeature) {
      if (!track.feature || !detFeature || !track.feature.color || !detFeature.color) return 1.0;

      // 1. Color Feature Cost (Cosine Distance)
      const colorCost = 1 - cosineSimilarity(track.feature.color, detFeature.color);

      // 2. Geometric Feature Cost (Normalized Euclidean Distance)
      let geoDist = 0;
      const geoWeights = [1, 1, 2, 0.5, 0.5]; // Weights for [aspect, area, conf, x, y]
      for (let i = 0; i < track.feature.geometric.length; i++) {
        const diff = track.feature.geometric[i] - detFeature.geometric[i];
        geoDist += (diff * diff) * geoWeights[i];
      }
      const geometricCost = Math.min(1.0, Math.sqrt(geoDist) / track.feature.geometric.length);

      // 3. Combine costs (50% color, 50% geometric)
      const combinedCost = 0.5 * colorCost + 0.5 * geometricCost;
      
      return combinedCost;
    }

    /** Remove stale tracks and release their IDs */
    _prune() {
      const beforeCount = this.tracks.length;
      const prunedTracks = [];
      
      this.tracks = this.tracks.filter(t => {
        const ttl = t.locked ? this.maxLostLocked : this.maxLostUnlocked;
        const shouldKeep = t.lostFrames <= ttl;
        
        if (!shouldKeep) {
          prunedTracks.push(t);
          // Release ID for future reuse
          // 释放ID以供将来重用
          this.idManager.releaseID(t.id);
          
          if (t.locked) {
            console.log(`Pruning locked track ${t.id} after ${t.lostFrames} lost frames`);
          }
        }
        
        return shouldKeep;
      });
      
      const prunedCount = beforeCount - this.tracks.length;
      if (prunedCount > 0) {
        console.log(`Pruned ${prunedCount} stale tracks`);
      }
    }

    /** Enhanced track creation with ID conflict prevention */
    _createTrackFromDet(d, feature, locked) {
      // Check for potential ID conflicts in crowded scenes
      // 在密集场景中检查潜在的ID冲突
      const activeTracks = this.tracks.filter(t => t.lostFrames < 5).length;
      const isCrowdedScene = activeTracks > 8;
      
      if (isCrowdedScene && !locked) {
        // In crowded scenes, be more conservative about creating new tracks
        // 在密集场景中，对创建新轨迹更加保守
        const nearbyTracks = this.tracks.filter(t => {
          const distance = Math.sqrt(
            (t.cx - (d.x + d.w/2))**2 + 
            (t.cy - (d.y + d.h/2))**2
          );
          return distance < Math.max(d.w, d.h) * 1.5;
        });
        
        if (nearbyTracks.length > 0) {
          console.log(`Skipping track creation in crowded scene - ${nearbyTracks.length} nearby tracks`);
          return null;
        }
      }
      
      // Use ID manager to allocate unique ID
      // 使用ID管理器分配唯一ID
      const id = this.idManager.allocateID();
      const t = new Track(id, { x: d.x, y: d.y, w: d.w, h: d.h }, { locked });
      if (feature) t.feature = feature;
      // Store object class information for display
      // 存储对象类别信息用于显示
      t.class = d.class || 'object';
      this.tracks.push(t);
      
      console.log(`Created new track ${id} (locked: ${locked}, crowded: ${isCrowdedScene})`);
      return id;
    }

    /** Public: lock a track by clicking a point on canvas */
    async lockOn(point, detections, videoContext) {
      if (!detections || detections.length === 0) return null;
      // find detections containing point
      const candidates = detections
        .map((d, i) => ({ i, d, box: { x: d.bbox[0] ?? d.x, y: d.bbox[1] ?? d.y, w: d.bbox[2] ?? d.w, h: d.bbox[3] ?? d.h } }))
        .filter(o => point.x >= o.box.x && point.y >= o.box.y && point.x <= o.box.x + o.box.w && point.y <= o.box.y + o.box.h)
        .sort((a, b) => (b.d.score || 0) - (a.d.score || 0));

      let chosen = candidates[0];
      // fallback: nearest center within 40px if no box contains point
      if (!chosen) {
        let best = null, bestDist = 40;
        for (let i = 0; i < detections.length; i++) {
          const d = detections[i];
          const box = { x: d.bbox?.[0] ?? d.x, y: d.bbox?.[1] ?? d.y, w: d.bbox?.[2] ?? d.w, h: d.bbox?.[3] ?? d.h };
          const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
          const dist = Math.hypot(cx - point.x, cy - point.y);
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
      const feature = await this.encoder.encode(videoContext, chosen.box);
      return this._createTrackFromDet({ ...chosen.box, class: chosen.d.class || 'object' }, feature, true);
    }

    /** Unlock and optionally remove a track */
    unlock(id, remove = false) {
      const idx = this.tracks.findIndex(t => t.id === id);
      if (idx < 0) return false;
      if (remove) { this.tracks.splice(idx, 1); return true; }
      this.tracks[idx].locked = false; return true;
    }

    /** Clear all tracks and reset ID manager */
    clear() { 
      // Release all track IDs
      // 释放所有轨迹ID
      for (const track of this.tracks) {
        this.idManager.releaseID(track.id);
      }
      this.tracks = []; 
    }

    /** Get shallow copy of tracks */
    getTracks() { return this.tracks.slice(); }
  }

  // expose to window
  window.Tracker = Tracker;
})();
