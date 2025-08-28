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

  /** Convert RGB to LAB color space for better perceptual distance */
  function rgbToLab(r, g, b) {
    // Convert RGB to XYZ
    r /= 255; g /= 255; b /= 255;
    r = r > 0.04045 ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = g > 0.04045 ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = b > 0.04045 ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;
    
    let x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
    let y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000;
    let z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;
    
    // Convert XYZ to LAB
    x = x > 0.008856 ? Math.pow(x, 1/3) : (7.787 * x + 16/116);
    y = y > 0.008856 ? Math.pow(y, 1/3) : (7.787 * y + 16/116);
    z = z > 0.008856 ? Math.pow(z, 1/3) : (7.787 * z + 16/116);
    
    return {
      l: (116 * y) - 16,
      a: 500 * (x - y),
      b: 200 * (y - z)
    };
  }

  /** Calculate perceptual color distance using Delta E CIE76 */
  function deltaE(lab1, lab2) {
    const dl = lab1.l - lab2.l;
    const da = lab1.a - lab2.a;
    const db = lab1.b - lab2.b;
    return Math.sqrt(dl*dl + da*da + db*db);
  }

  /** Extract dominant colors using k-means clustering */
  function extractDominantColors(imageData, k = 5) {
    const pixels = [];
    const data = imageData.data;
    
    // Sample pixels (every 4th pixel to reduce computation)
    for (let i = 0; i < data.length; i += 16) {
      pixels.push([data[i], data[i+1], data[i+2]]);
    }
    
    if (pixels.length === 0) return [];
    
    // Simple k-means clustering
    let centroids = [];
    for (let i = 0; i < k; i++) {
      const idx = Math.floor(Math.random() * pixels.length);
      centroids.push([...pixels[idx]]);
    }
    
    // Iterate to find centroids
    for (let iter = 0; iter < 10; iter++) {
      const clusters = Array(k).fill().map(() => []);
      
      // Assign pixels to nearest centroid
      for (const pixel of pixels) {
        let minDist = Infinity;
        let bestCluster = 0;
        
        for (let c = 0; c < k; c++) {
          const dist = Math.sqrt(
            Math.pow(pixel[0] - centroids[c][0], 2) +
            Math.pow(pixel[1] - centroids[c][1], 2) +
            Math.pow(pixel[2] - centroids[c][2], 2)
          );
          if (dist < minDist) {
            minDist = dist;
            bestCluster = c;
          }
        }
        clusters[bestCluster].push(pixel);
      }
      
      // Update centroids
      for (let c = 0; c < k; c++) {
        if (clusters[c].length > 0) {
          centroids[c] = [
            clusters[c].reduce((sum, p) => sum + p[0], 0) / clusters[c].length,
            clusters[c].reduce((sum, p) => sum + p[1], 0) / clusters[c].length,
            clusters[c].reduce((sum, p) => sum + p[2], 0) / clusters[c].length
          ];
        }
      }
    }
    
    // Return dominant colors with their weights
    const result = [];
    for (let c = 0; c < k; c++) {
      const cluster = [];
      for (const pixel of pixels) {
        const dist = Math.sqrt(
          Math.pow(pixel[0] - centroids[c][0], 2) +
          Math.pow(pixel[1] - centroids[c][1], 2) +
          Math.pow(pixel[2] - centroids[c][2], 2)
        );
        if (dist < 50) cluster.push(pixel); // Threshold for assignment
      }
      
      if (cluster.length > 0) {
        result.push({
          color: centroids[c],
          weight: cluster.length / pixels.length,
          lab: rgbToLab(centroids[c][0], centroids[c][1], centroids[c][2])
        });
      }
    }
    
    return result.sort((a, b) => b.weight - a.weight);
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
      const enhancedFeatures = await this._extractEnhancedFeatures(ctx, bbox);
      
      return { 
        color: colorFeature, 
        geometric: geometricFeature,
        enhanced: enhancedFeatures
      };
    }

    _extractGeometricFeatures(bbox) {
      const { x, y, w, h, confidence } = bbox;
      const aspectRatio = w / (h + 1e-6);
      const normalizedArea = (w * h) / (this.imageWidth * this.imageHeight);
      const normalizedX = (x + w / 2) / this.imageWidth;
      const normalizedY = (y + h / 2) / this.imageHeight;
      
      // Enhanced geometric features
      const geometricCentroid = this._calculateGeometricCentroid(bbox);
      const shapeCompactness = (4 * Math.PI * w * h) / Math.pow(2 * (w + h), 2);
      const diagonalRatio = Math.sqrt(w*w + h*h) / Math.max(w, h);
      
      // Feature vector: [aspectRatio, normalizedArea, confidence, normalizedX, normalizedY, 
      //                  centroidX, centroidY, compactness, diagonalRatio]
      return [
        aspectRatio, normalizedArea, confidence || 0.5, normalizedX, normalizedY,
        geometricCentroid.x, geometricCentroid.y, shapeCompactness, diagonalRatio
      ];
    }
    
    _calculateGeometricCentroid(bbox) {
      // For rectangular bounding box, geometric centroid is simply the center
      // In future, this could be enhanced with actual shape analysis
      return {
        x: (bbox.x + bbox.w / 2) / this.imageWidth,
        y: (bbox.y + bbox.h / 2) / this.imageHeight
      };
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
    
    /** Extract enhanced features including color centroid and overall color characteristics */
    async _extractEnhancedFeatures(ctx, bbox) {
      const x = bbox.x + (1 - this.innerCrop) * 0.5 * bbox.w;
      const y = bbox.y + (1 - this.innerCrop) * 0.5 * bbox.h;
      const w = bbox.w * this.innerCrop;
      const h = bbox.h * this.innerCrop;

      const ix = Math.round(x), iy = Math.round(y);
      const iw = Math.max(1, Math.round(w)), ih = Math.max(1, Math.round(h));
      if (iw < 6 || ih < 6) return null;

      let img;
      try { img = ctx.getImageData(ix, iy, iw, ih); }
      catch (e) { return null; }

      const data = img.data;
      const step = Math.max(1, this.sampleStep);
      
      // Extract dominant colors
      const dominantColors = extractDominantColors(img, 3);
      
      // Calculate color centroid (weighted center based on color intensity)
      let colorCentroid = this._calculateColorCentroid(data, iw, ih, step);
      
      // Calculate overall color statistics
      const colorStats = this._calculateColorStatistics(data, step);
      
      // Calculate color distribution entropy
      const colorEntropy = this._calculateColorEntropy(data, step);
      
      return {
        dominantColors: dominantColors,
        colorCentroid: colorCentroid,
        colorStatistics: colorStats,
        colorEntropy: colorEntropy,
        overallBrightness: colorStats.avgBrightness,
        colorVariance: colorStats.variance
      };
    }
    
    /** Calculate color centroid - weighted center based on color intensity and distribution */
    _calculateColorCentroid(data, width, height, step) {
      let totalWeight = 0;
      let weightedX = 0;
      let weightedY = 0;
      
      for (let y = 0; y < height; y += step) {
        for (let x = 0; x < width; x += step) {
          const p = (y * width + x) * 4;
          const r = data[p], g = data[p + 1], b = data[p + 2];
          
          // Use luminance as weight (perceived brightness)
          const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
          
          // Also consider color saturation for weight
          const { s: saturation } = rgbToHsv(r, g, b);
          const weight = luminance * (1 + saturation); // Bright and saturated pixels have more weight
          
          weightedX += x * weight;
          weightedY += y * weight;
          totalWeight += weight;
        }
      }
      
      if (totalWeight === 0) {
        return { x: 0.5, y: 0.5 }; // Default to center if no weight
      }
      
      return {
        x: (weightedX / totalWeight) / width,  // Normalized to [0,1]
        y: (weightedY / totalWeight) / height  // Normalized to [0,1]
      };
    }
    
    /** Calculate overall color statistics */
    _calculateColorStatistics(data, step) {
      let totalR = 0, totalG = 0, totalB = 0;
      let totalBrightness = 0;
      let totalSaturation = 0;
      let count = 0;
      
      const values = [];
      
      for (let i = 0; i < data.length; i += 4 * step) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const brightness = (r + g + b) / 3;
        const { s: saturation } = rgbToHsv(r, g, b);
        
        totalR += r;
        totalG += g;
        totalB += b;
        totalBrightness += brightness;
        totalSaturation += saturation;
        values.push(brightness);
        count++;
      }
      
      if (count === 0) {
        return {
          avgR: 0, avgG: 0, avgB: 0,
          avgBrightness: 0, avgSaturation: 0,
          variance: 0
        };
      }
      
      const avgBrightness = totalBrightness / count;
      
      // Calculate variance
      let variance = 0;
      for (const val of values) {
        variance += Math.pow(val - avgBrightness, 2);
      }
      variance /= count;
      
      return {
        avgR: totalR / count,
        avgG: totalG / count,
        avgB: totalB / count,
        avgBrightness: avgBrightness,
        avgSaturation: totalSaturation / count,
        variance: variance
      };
    }
    
    /** Calculate color entropy to measure color diversity */
    _calculateColorEntropy(data, step) {
      const colorBins = new Map();
      let totalPixels = 0;
      
      // Quantize colors to reduce bins (8 levels per channel = 512 total colors)
      for (let i = 0; i < data.length; i += 4 * step) {
        const r = Math.floor(data[i] / 32) * 32;     // 8 levels
        const g = Math.floor(data[i + 1] / 32) * 32; // 8 levels  
        const b = Math.floor(data[i + 2] / 32) * 32; // 8 levels
        const colorKey = `${r},${g},${b}`;
        
        colorBins.set(colorKey, (colorBins.get(colorKey) || 0) + 1);
        totalPixels++;
      }
      
      if (totalPixels === 0) return 0;
      
      // Calculate entropy: -Σ(p * log2(p))
      let entropy = 0;
      for (const count of colorBins.values()) {
        const probability = count / totalPixels;
        if (probability > 0) {
          entropy -= probability * Math.log2(probability);
        }
      }
      
      return entropy;
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
    
    /** Update appearance model with EMA including enhanced features */
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
        
        // EMA update for enhanced features
        if (newFeature.enhanced && this.feature.enhanced) {
          this.updateEnhancedFeatures(newFeature.enhanced, alpha);
        } else if (newFeature.enhanced) {
          this.feature.enhanced = newFeature.enhanced;
        }
      } else {
        this.feature = newFeature;
      }
    }
    
    /** Update enhanced features with EMA smoothing */
    updateEnhancedFeatures(newEnhanced, alpha) {
      if (!this.feature.enhanced) {
        this.feature.enhanced = newEnhanced;
        return;
      }
      
      const current = this.feature.enhanced;
      
      // Update color centroid
      if (newEnhanced.colorCentroid && current.colorCentroid) {
        current.colorCentroid.x = (1 - alpha) * current.colorCentroid.x + alpha * newEnhanced.colorCentroid.x;
        current.colorCentroid.y = (1 - alpha) * current.colorCentroid.y + alpha * newEnhanced.colorCentroid.y;
      }
      
      // Update color statistics
      if (newEnhanced.colorStatistics && current.colorStatistics) {
        const stats = current.colorStatistics;
        const newStats = newEnhanced.colorStatistics;
        stats.avgR = (1 - alpha) * stats.avgR + alpha * newStats.avgR;
        stats.avgG = (1 - alpha) * stats.avgG + alpha * newStats.avgG;
        stats.avgB = (1 - alpha) * stats.avgB + alpha * newStats.avgB;
        stats.avgBrightness = (1 - alpha) * stats.avgBrightness + alpha * newStats.avgBrightness;
        stats.avgSaturation = (1 - alpha) * stats.avgSaturation + alpha * newStats.avgSaturation;
        stats.variance = (1 - alpha) * stats.variance + alpha * newStats.variance;
      }
      
      // Update other enhanced features
      if (newEnhanced.colorEntropy !== undefined) {
        current.colorEntropy = (1 - alpha) * (current.colorEntropy || 0) + alpha * newEnhanced.colorEntropy;
      }
      
      if (newEnhanced.overallBrightness !== undefined) {
        current.overallBrightness = (1 - alpha) * (current.overallBrightness || 0) + alpha * newEnhanced.overallBrightness;
      }
      
      if (newEnhanced.colorVariance !== undefined) {
        current.colorVariance = (1 - alpha) * (current.colorVariance || 0) + alpha * newEnhanced.colorVariance;
      }
      
      // Update dominant colors (replace with new ones due to complexity of EMA on clusters)
      if (newEnhanced.dominantColors && newEnhanced.dominantColors.length > 0) {
        current.dominantColors = newEnhanced.dominantColors;
      }
    }
    
    /** Get enhanced appearance match score using all feature parameters */
    getEnhancedAppearanceScore(detectionFeature) {
      if (!this.feature || !detectionFeature) return 0;
      
      let totalScore = 0;
      let weightSum = 0;
      
      // 1. Color histogram similarity (weight: 0.3)
      if (this.feature.color && detectionFeature.color) {
        const colorSim = cosineSimilarity(this.feature.color, detectionFeature.color);
        totalScore += colorSim * 0.3;
        weightSum += 0.3;
      }
      
      // 2. Geometric feature similarity (weight: 0.2)
      if (this.feature.geometric && detectionFeature.geometric) {
        const geoSim = this.calculateGeometricSimilarity(this.feature.geometric, detectionFeature.geometric);
        totalScore += geoSim * 0.2;
        weightSum += 0.2;
      }
      
      // 3. Enhanced features similarity (weight: 0.5)
      if (this.feature.enhanced && detectionFeature.enhanced) {
        const enhancedSim = this.calculateEnhancedSimilarity(this.feature.enhanced, detectionFeature.enhanced);
        totalScore += enhancedSim * 0.5;
        weightSum += 0.5;
      }
      
      return weightSum > 0 ? totalScore / weightSum : 0;
    }
    
    /** Calculate geometric feature similarity */
    calculateGeometricSimilarity(geo1, geo2) {
      if (!geo1 || !geo2 || geo1.length !== geo2.length) return 0;
      
      let similarity = 0;
      const weights = [0.15, 0.2, 0.1, 0.1, 0.1, 0.15, 0.15, 0.025, 0.025]; // Weights for each geometric feature
      
      for (let i = 0; i < Math.min(geo1.length, weights.length); i++) {
        const diff = Math.abs(geo1[i] - geo2[i]);
        const maxVal = Math.max(Math.abs(geo1[i]), Math.abs(geo2[i]), 1e-6);
        const featureSim = Math.exp(-diff / maxVal); // Exponential similarity
        similarity += featureSim * weights[i];
      }
      
      return similarity;
    }
    
    /** Calculate enhanced features similarity */
    calculateEnhancedSimilarity(enh1, enh2) {
      let totalSim = 0;
      let count = 0;
      
      // Color centroid similarity (weight: 0.25)
      if (enh1.colorCentroid && enh2.colorCentroid) {
        const centroidDist = Math.sqrt(
          Math.pow(enh1.colorCentroid.x - enh2.colorCentroid.x, 2) +
          Math.pow(enh1.colorCentroid.y - enh2.colorCentroid.y, 2)
        );
        const centroidSim = Math.exp(-centroidDist * 5); // Scale factor for sensitivity
        totalSim += centroidSim * 0.25;
        count += 0.25;
      }
      
      // Color statistics similarity (weight: 0.3)
      if (enh1.colorStatistics && enh2.colorStatistics) {
        const stats1 = enh1.colorStatistics;
        const stats2 = enh2.colorStatistics;
        
        const brightnessSim = 1 - Math.abs(stats1.avgBrightness - stats2.avgBrightness) / 255;
        const saturationSim = 1 - Math.abs(stats1.avgSaturation - stats2.avgSaturation);
        const varianceSim = 1 - Math.abs(stats1.variance - stats2.variance) / Math.max(stats1.variance, stats2.variance, 1);
        
        const statsSim = (brightnessSim + saturationSim + varianceSim) / 3;
        totalSim += statsSim * 0.3;
        count += 0.3;
      }
      
      // Dominant colors similarity (weight: 0.3)
      if (enh1.dominantColors && enh2.dominantColors && enh1.dominantColors.length > 0 && enh2.dominantColors.length > 0) {
        const domColorSim = this.calculateDominantColorSimilarity(enh1.dominantColors, enh2.dominantColors);
        totalSim += domColorSim * 0.3;
        count += 0.3;
      }
      
      // Color entropy similarity (weight: 0.15)
      if (enh1.colorEntropy !== undefined && enh2.colorEntropy !== undefined) {
        const entropySim = 1 - Math.abs(enh1.colorEntropy - enh2.colorEntropy) / Math.max(enh1.colorEntropy, enh2.colorEntropy, 1);
        totalSim += entropySim * 0.15;
        count += 0.15;
      }
      
      return count > 0 ? totalSim / count : 0;
    }
    
    /** Calculate similarity between dominant color sets */
    calculateDominantColorSimilarity(colors1, colors2) {
      if (!colors1 || !colors2 || colors1.length === 0 || colors2.length === 0) return 0;
      
      let totalSimilarity = 0;
      let totalWeight = 0;
      
      // Compare each color in set 1 with best match in set 2
      for (const color1 of colors1) {
        let bestMatch = 0;
        
        for (const color2 of colors2) {
          // Use LAB color space for perceptual distance
          const distance = deltaE(color1.lab, color2.lab);
          const similarity = Math.exp(-distance / 50); // Scale factor for color sensitivity
          bestMatch = Math.max(bestMatch, similarity);
        }
        
        totalSimilarity += bestMatch * color1.weight;
        totalWeight += color1.weight;
      }
      
      return totalWeight > 0 ? totalSimilarity / totalWeight : 0;
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
      
      // Enhanced matching weights for robust tracking with enhanced features
      // 增强特征的鲁棒追踪匹配权重
      this.wIoU = opts.wIoU ?? 0.20;        // Reduced IoU weight to rely more on enhanced features
      this.wApp = opts.wApp ?? 0.25;        // Traditional appearance weight
      this.wEnhanced = opts.wEnhanced ?? 0.35; // Enhanced features weight (highest priority)
      this.wCtr = opts.wCtr ?? 0.15;        // Center distance weight
      this.wRatio = opts.wRatio ?? 0.05;    // Aspect ratio weight
      
      // Adaptive cost thresholds with enhanced feature consideration
      // 考虑增强特征的自适应成本阈值
      this.costThreshold = opts.costThreshold ?? 0.70; // Slightly lower for enhanced features
      this.crowdedSceneThreshold = opts.crowdedSceneThreshold ?? 0.60; // Stricter for crowded scenes
      this.lockedTrackThreshold = opts.lockedTrackThreshold ?? 0.80; // More permissive for locked tracks
      this.enhancedFeatureThreshold = opts.enhancedFeatureThreshold ?? 0.65; // Threshold for enhanced feature matching
      this.autoCreate = !!opts.autoCreate;
      
      // Enhanced feature robustness parameters
      // 增强特征鲁棒性参数
      this.robustnessParams = {
        minFeatureConfidence: 0.3,     // Minimum confidence to use enhanced features
        adaptiveWeighting: true,       // Enable adaptive feature weighting
        colorCentroidWeight: 0.3,      // Weight for color centroid in enhanced matching
        dominantColorWeight: 0.4,      // Weight for dominant color matching
        colorStatsWeight: 0.3,         // Weight for color statistics
        featureStabilityWindow: 10,    // Frames to consider for feature stability
        robustMatchingEnabled: true    // Enable robust matching algorithms
      };
      
      // Occlusion handling parameters with enhanced features
      // 使用增强特征的遮挡处理参数
      this.occlusionParams = {
        maxOcclusionFrames: 80,        // Increased with enhanced features
        reappearanceRadius: 180,       // Larger search radius
        confidenceDecay: 0.96,         // Slower decay with better features
        minReappearanceScore: 0.55,    // Lower threshold with enhanced matching
        enhancedRecoveryEnabled: true  // Use enhanced features for recovery
      };
      
      // Performance monitoring for enhanced features
      // 增强特征的性能监控
      this.performanceStats = {
        enhancedMatches: 0,
        traditionalMatches: 0,
        recoveredTracks: 0,
        featureExtractionTime: 0
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

      // Enhanced feature extraction with performance optimization
      // 性能优化的增强特征提取
      let enriched = dets;
      if (this.enableReID && videoContext) {
        const startTime = performance.now();
        const maxFeat = this.robustnessParams.robustMatchingEnabled ? 20 : 15; // More features for robust matching
        
        // Prioritize detections for feature extraction
        // 优先处理检测结果进行特征提取
        const prioritizedDets = this.prioritizeDetectionsForFeatures(dets);
        
        enriched = prioritizedDets.map((dd, idx) => {
          let feat = null;
          if (!this.focusClasses.length || this.focusClasses.includes(dd.class)) {
            if (idx < maxFeat) {
              try {
                feat = this.encoder.encode(videoContext, dd);
                if (feat && feat.enhanced) {
                  // Validate enhanced features quality
                  feat.quality = this.assessFeatureQuality(feat);
                }
              } catch (error) {
                console.warn('Feature extraction failed:', error);
                feat = null;
              }
            }
          }
          return { ...dd, feature: feat };
        });
        
        // Update performance stats
        this.performanceStats.featureExtractionTime = performance.now() - startTime;
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
        // Use enhanced Hungarian algorithm for optimal assignment with robust features
        this._associateAndUpdateEnhancedHungarian(locked, enriched, unmatchedDetIdx, true);
        this._associateAndUpdateEnhancedHungarian(normal, enriched, unmatchedDetIdx, false);
      }

      // Increase lost for unmatched tracks
      for (const t of this.tracks) {
        if (!t._updatedThisRound) {
          t.lostFrames += 1;
        }
        delete t._updatedThisRound;
      }

      // Enhanced track recovery with robust features
      // 使用鲁棒特征的增强轨迹恢复
      this._attemptEnhancedTrackRecovery(enriched, unmatchedDetIdx);

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
            if (t.getEnhancedAppearanceScore) {
              // Use enhanced appearance scoring
              const matchScore = t.getEnhancedAppearanceScore(d.feature);
              app = 1 - matchScore;
              
              // ID switching prevention with enhanced features
              if (matchScore < 0.7) { // Threshold for enhanced features
                let betterMatchExists = false;
                for (const otherTrack of this.tracks) {
                  if (otherTrack.id !== t.id && otherTrack.getEnhancedAppearanceScore) {
                    const otherScore = otherTrack.getEnhancedAppearanceScore(d.feature);
                    if (otherScore > matchScore + 0.2) {
                      betterMatchExists = true;
                      break;
                    }
                  }
                }
                
                if (betterMatchExists) {
                  idSwitchPenalty = t.idConsistency ? t.idConsistency.switchPenalty || 0.3 : 0.3;
                }
              }
            } else if (t.getAppearanceMatchScore && t.appearanceModel && t.appearanceModel.templates.length > 0) {
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
            } else if (t.feature && t.feature.color && d.feature.color) {
              app = clamp(1 - cosineSimilarity(t.feature.color, d.feature.color), 0, 1);
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

    /** Prioritize detections for feature extraction based on tracking needs */
    prioritizeDetectionsForFeatures(detections) {
      // Sort by confidence and existing track proximity
      // 根据置信度和现有轨迹接近度排序
      return detections.sort((a, b) => {
        let scoreA = a.score || 0;
        let scoreB = b.score || 0;
        
        // Boost score if detection is near existing tracks
        // 如果检测结果靠近现有轨迹则提升分数
        for (const track of this.tracks) {
          const distA = Math.sqrt(Math.pow(track.cx - (a.x + a.w/2), 2) + Math.pow(track.cy - (a.y + a.h/2), 2));
          const distB = Math.sqrt(Math.pow(track.cx - (b.x + b.w/2), 2) + Math.pow(track.cy - (b.y + b.h/2), 2));
          
          if (distA < 100) scoreA += 0.2; // Proximity bonus
          if (distB < 100) scoreB += 0.2;
        }
        
        return scoreB - scoreA; // Higher score first
      });
    }
    
    /** Assess quality of extracted features */
    assessFeatureQuality(features) {
      let quality = 0.5; // Base quality
      
      if (features.enhanced) {
        const enh = features.enhanced;
        
        // Color diversity indicates good feature quality
        if (enh.colorEntropy > 2.0) quality += 0.2;
        
        // Dominant colors presence
        if (enh.dominantColors && enh.dominantColors.length >= 2) quality += 0.15;
        
        // Color variance indicates texture richness
        if (enh.colorVariance > 500) quality += 0.1;
        
        // Color centroid stability (not at extreme edges)
        const centroid = enh.colorCentroid;
        if (centroid && centroid.x > 0.1 && centroid.x < 0.9 && centroid.y > 0.1 && centroid.y < 0.9) {
          quality += 0.05;
        }
      }
      
      return Math.min(1.0, quality);
    }
    
    /** Enhanced Hungarian algorithm with robust feature matching */
    _associateAndUpdateEnhancedHungarian(trackList, detections, unmatchedDetIdx, isLockedPass) {
      if (trackList.length === 0 || unmatchedDetIdx.size === 0) return;

      // Build enhanced cost matrix with robust features
      // 使用鲁棒特征构建增强成本矩阵
      const detectionArray = Array.from(unmatchedDetIdx).map(idx => detections[idx]);
      const detectionIndices = Array.from(unmatchedDetIdx);
      
      if (detectionArray.length === 0) return;
      
      const costMatrix = [];
      const maxCost = 10.0;
      
      for (let ti = 0; ti < trackList.length; ti++) {
        const t = trackList[ti];
        const tb = t.bbox;
        const row = [];
        
        // Enhanced adaptive gating with feature-based expansion
        // 基于特征的增强自适应门控
        let gating = this.calculateAdaptiveGating(t);
        
        for (let di = 0; di < detectionArray.length; di++) {
          const d = detectionArray[di];
          const db = { x: d.x, y: d.y, w: d.w, h: d.h };
          const i = iou(tb, db);
          const ctr = centerDistance(tb, db);
          
          // Enhanced gating check
          if (i < 0.005 && ctr > gating) {
            row.push(maxCost);
            continue;
          }
          
          // Calculate enhanced cost with robust features
          const cost = this.calculateEnhancedCost(t, d, i, ctr, gating, isLockedPass);
          row.push(cost > this.getAdaptiveThreshold(t, isLockedPass) ? maxCost : cost);
        }
        
        costMatrix.push(row);
      }
      
      // Solve using Hungarian algorithm
      const assignments = this.hungarian.solve(costMatrix);
      
      // Apply assignments with enhanced validation
      this.applyEnhancedAssignments(assignments, trackList, detectionArray, detectionIndices, costMatrix, unmatchedDetIdx, isLockedPass);
    }
    
    /** Calculate adaptive gating based on track state and features */
    calculateAdaptiveGating(track) {
      const velocityMagnitude = Math.sqrt((track.vx || 0) ** 2 + (track.vy || 0) ** 2);
      const velocityFactor = Math.min(3.0, 1.0 + velocityMagnitude / 50);
      let gating = Math.max(this.gatingBase, 0.5 * Math.hypot(track.w, track.h)) * velocityFactor;
      
      // Enhanced gating based on feature quality
      if (track.feature && track.feature.quality > 0.7) {
        gating *= 1.3; // More permissive for high-quality features
      }
      
      // Occlusion state adjustments
      if (track.occlusionState && track.occlusionState.isOccluded) {
        gating = Math.max(gating, track.occlusionState.searchRadius || gating * 2.5);
      }
      
      // Locked track adjustments
      if (track.locked) {
        gating *= 1.4;
      }
      
      return gating;
    }
    
    /** Calculate enhanced cost with robust feature integration */
    calculateEnhancedCost(track, detection, iou_val, centerDist, gating, isLockedPass) {
      let enhancedCost = 0;
      let traditionalCost = 0;
      let hasEnhancedFeatures = false;
      
      // Enhanced appearance cost using robust features
      if (this.enableReID && detection.feature && track.getEnhancedAppearanceScore) {
        const similarity = track.getEnhancedAppearanceScore(detection.feature);
        enhancedCost = 1.0 - similarity;
        hasEnhancedFeatures = true;
        this.performanceStats.enhancedMatches++;
      } else if (track.feature && detection.feature) {
        // Fallback to traditional appearance
        enhancedCost = this._getAppearanceCost(track, detection, detection.feature);
        this.performanceStats.traditionalMatches++;
      } else {
        enhancedCost = 0.5; // Neutral cost when no features available
      }
      
      // Geometric costs
      const ratioT = (track.w / (track.h + 1e-3));
      const ratioD = (detection.w / (detection.h + 1e-3));
      const ratioDiff = Math.min(1, Math.abs(ratioT - ratioD) / Math.max(ratioT, ratioD));
      const ctrNorm = clamp(centerDist / (gating * 1.5), 0, 1);
      
      // Motion consistency with enhanced prediction
      const motionCost = this.calculateMotionCost(track, detection, gating);
      
      // Adaptive weight calculation
      const weights = this.calculateAdaptiveWeights(track, detection, hasEnhancedFeatures, isLockedPass);
      
      // Combined cost with enhanced features priority
      const baseCost = weights.iou * (1 - iou_val) + 
                      weights.appearance * enhancedCost + 
                      weights.center * ctrNorm + 
                      weights.ratio * ratioDiff + 
                      weights.motion * motionCost;
      
      // Add robustness penalties
      const robustnessPenalty = this.calculateRobustnessPenalty(track, detection);
      
      return baseCost + robustnessPenalty;
    }
    
    /** Calculate adaptive weights based on feature availability and track state */
    calculateAdaptiveWeights(track, detection, hasEnhancedFeatures, isLockedPass) {
      let weights = {
        iou: this.wIoU,
        appearance: hasEnhancedFeatures ? this.wEnhanced : this.wApp,
        center: this.wCtr,
        ratio: this.wRatio,
        motion: 0.15
      };
      
      // Adjust for locked tracks
      if (isLockedPass) {
        weights.iou *= 0.8;
        weights.appearance *= 1.3;
        weights.center *= 1.2;
        weights.motion *= 1.2;
      }
      
      // Adjust for occlusion
      if (track.occlusionState && track.occlusionState.isOccluded) {
        weights.iou *= 0.4;
        weights.appearance *= 1.6;
        weights.center *= 1.4;
        weights.motion *= 1.5;
      }
      
      // Adjust for feature quality
      if (hasEnhancedFeatures && track.feature && track.feature.quality > 0.8) {
        weights.appearance *= 1.2; // Trust high-quality features more
      }
      
      return weights;
    }
    
    /** Calculate motion cost with enhanced prediction */
    calculateMotionCost(track, detection, gating) {
      const dt = 1/30;
      const predictedX = track.cx + (track.vx || 0) * dt;
      const predictedY = track.cy + (track.vy || 0) * dt;
      const detCenterX = detection.x + detection.w / 2;
      const detCenterY = detection.y + detection.h / 2;
      const motionDist = Math.sqrt((predictedX - detCenterX) ** 2 + (predictedY - detCenterY) ** 2);
      
      const velocityConfidence = Math.min(1.0, track.hits / 10);
      const adaptiveMotionGating = gating * (0.5 + 0.5 * velocityConfidence);
      
      return clamp(motionDist / adaptiveMotionGating, 0, 1);
    }
    
    /** Calculate robustness penalty for unrealistic associations */
    calculateRobustnessPenalty(track, detection) {
      let penalty = 0;
      
      // Size change penalty
      const sizeChangeRatio = Math.max(detection.w/track.w, track.w/detection.w) * 
                             Math.max(detection.h/track.h, track.h/detection.h);
      if (sizeChangeRatio > 2.5) {
        penalty += Math.min(0.4, (sizeChangeRatio - 2.5) * 0.15);
      }
      
      // Trajectory consistency penalty
      if (track.trajectory.length >= 3) {
        const recent = track.trajectory.slice(-3);
        const avgVelX = (recent[2].x - recent[0].x) / 2;
        const avgVelY = (recent[2].y - recent[0].y) / 2;
        const expectedX = track.cx + avgVelX;
        const expectedY = track.cy + avgVelY;
        
        const detCenterX = detection.x + detection.w / 2;
        const detCenterY = detection.y + detection.h / 2;
        const trajDeviation = Math.sqrt((expectedX - detCenterX)**2 + (expectedY - detCenterY)**2);
        
        const maxReasonableDeviation = Math.max(60, Math.hypot(track.w, track.h) * 0.9);
        if (trajDeviation > maxReasonableDeviation) {
          penalty += Math.min(0.3, trajDeviation / maxReasonableDeviation * 0.15);
        }
      }
      
      return penalty;
    }
    
    /** Get adaptive threshold based on track state */
    getAdaptiveThreshold(track, isLockedPass) {
      let threshold = this.costThreshold;
      
      // Adjust for scene complexity
      const activeTracks = this.tracks.filter(t => t.lostFrames < 5).length;
      const isCrowdedScene = activeTracks > 8;
      
      if (isCrowdedScene) {
        threshold = this.crowdedSceneThreshold;
      }
      
      // Adjust for track type
      if (track.locked) {
        threshold = Math.max(threshold, this.lockedTrackThreshold);
      }
      
      // Adjust for feature quality
      if (track.feature && track.feature.quality > 0.8) {
        threshold *= 1.1; // More permissive for high-quality features
      }
      
      // Adjust for occlusion
      if (track.occlusionState && track.occlusionState.isOccluded) {
        threshold *= 1.15;
      }
      
      return threshold;
    }
    
    /** Apply enhanced assignments with validation */
    applyEnhancedAssignments(assignments, trackList, detectionArray, detectionIndices, costMatrix, unmatchedDetIdx, isLockedPass) {
      for (let ti = 0; ti < assignments.length; ti++) {
        const di = assignments[ti];
        if (di >= 0 && di < detectionArray.length) {
          const cost = costMatrix[ti][di];
          const threshold = this.getAdaptiveThreshold(trackList[ti], isLockedPass);
          
          if (cost <= threshold) {
            const t = trackList[ti];
            const d = detectionArray[di];
            const originalDetIdx = detectionIndices[di];
            
            // Enhanced update with feature validation
            this.updateTrackWithEnhancedFeatures(t, d);
            t._updatedThisRound = true;
            unmatchedDetIdx.delete(originalDetIdx);
            
            if (isLockedPass) {
              console.log(`Enhanced Hungarian: Assigned locked track ${t.id} (cost: ${cost.toFixed(3)}, quality: ${d.feature?.quality?.toFixed(2) || 'N/A'})`);
            }
          }
        }
      }
    }
    
    /** Update track with enhanced feature validation */
    updateTrackWithEnhancedFeatures(track, detection) {
      // Validate feature consistency before update
      if (detection.feature && track.feature && this.robustnessParams.adaptiveWeighting) {
        const featureConsistency = this.validateFeatureConsistency(track.feature, detection.feature);
        if (featureConsistency < this.robustnessParams.minFeatureConfidence) {
          console.warn(`Low feature consistency for track ${track.id}: ${featureConsistency.toFixed(3)}`);
          // Still update but with reduced confidence
          detection.feature.quality = Math.min(detection.feature.quality || 0.5, 0.6);
        }
      }
      
      track.update({ x: detection.x, y: detection.y, w: detection.w, h: detection.h }, detection.feature);
    }
    
    /** Validate consistency between track and detection features */
    validateFeatureConsistency(trackFeature, detectionFeature) {
      if (!trackFeature.enhanced || !detectionFeature.enhanced) return 0.5;
      
      let consistency = 0;
      let count = 0;
      
      // Color centroid consistency
      if (trackFeature.enhanced.colorCentroid && detectionFeature.enhanced.colorCentroid) {
        const centroidDist = Math.sqrt(
          Math.pow(trackFeature.enhanced.colorCentroid.x - detectionFeature.enhanced.colorCentroid.x, 2) +
          Math.pow(trackFeature.enhanced.colorCentroid.y - detectionFeature.enhanced.colorCentroid.y, 2)
        );
        consistency += Math.exp(-centroidDist * 8); // Higher sensitivity
        count++;
      }
      
      // Brightness consistency
      if (trackFeature.enhanced.overallBrightness !== undefined && detectionFeature.enhanced.overallBrightness !== undefined) {
        const brightnessDiff = Math.abs(trackFeature.enhanced.overallBrightness - detectionFeature.enhanced.overallBrightness) / 255;
        consistency += 1 - brightnessDiff;
        count++;
      }
      
      // Dominant color consistency
      if (trackFeature.enhanced.dominantColors && detectionFeature.enhanced.dominantColors) {
        const colorSim = this.calculateDominantColorSimilarity(
          trackFeature.enhanced.dominantColors, 
          detectionFeature.enhanced.dominantColors
        );
        consistency += colorSim;
        count++;
      }
      
      return count > 0 ? consistency / count : 0.5;
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
            if (t.getEnhancedAppearanceScore) {
              // Use enhanced appearance scoring
              const matchScore = t.getEnhancedAppearanceScore(d.feature);
              app = 1 - matchScore;
            } else if (t.getAppearanceMatchScore && t.appearanceModel && t.appearanceModel.templates.length > 0) {
              const matchScore = t.getAppearanceMatchScore(d.feature, d);
              app = 1 - matchScore;
            } else if (t.feature && t.feature.color && d.feature.color) {
              app = clamp(1 - cosineSimilarity(t.feature.color, d.feature.color), 0, 1);
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

    /** Enhanced track recovery using robust features */
    _attemptEnhancedTrackRecovery(detections, unmatchedDetIdx) {
      if (!this.occlusionParams.enhancedRecoveryEnabled) {
        return this._attemptTrackRecovery(detections, unmatchedDetIdx);
      }
      
      const lostTracks = this.tracks.filter(t => 
        t.lostFrames > 0 && 
        t.lostFrames <= this.occlusionParams.maxOcclusionFrames &&
        !t._updatedThisRound
      );
      
      if (lostTracks.length === 0 || unmatchedDetIdx.size === 0) return;
      
      const unmatchedDetections = Array.from(unmatchedDetIdx).map(idx => detections[idx]);
      const recoveryPairs = [];
      
      // Enhanced recovery matching with robust features
      for (const track of lostTracks) {
        for (let i = 0; i < unmatchedDetections.length; i++) {
          const detection = unmatchedDetections[i];
          const detIdx = Array.from(unmatchedDetIdx)[i];
          
          // Calculate enhanced recovery score
          const recoveryScore = this.calculateEnhancedRecoveryScore(track, detection);
          
          if (recoveryScore > this.occlusionParams.minReappearanceScore) {
            recoveryPairs.push({
              track,
              detection,
              detIdx,
              score: recoveryScore,
              distance: this.calculateTrackDetectionDistance(track, detection)
            });
          }
        }
      }
      
      // Sort by recovery score and apply best matches
      recoveryPairs.sort((a, b) => b.score - a.score);
      
      const usedTracks = new Set();
      const usedDetections = new Set();
      
      for (const pair of recoveryPairs) {
        if (!usedTracks.has(pair.track.id) && !usedDetections.has(pair.detIdx)) {
          // Successful recovery
          this.performEnhancedRecovery(pair.track, pair.detection);
          pair.track._updatedThisRound = true;
          unmatchedDetIdx.delete(pair.detIdx);
          
          usedTracks.add(pair.track.id);
          usedDetections.add(pair.detIdx);
          
          this.performanceStats.recoveredTracks++;
          console.log(`Enhanced Recovery: Track ${pair.track.id} recovered (score: ${pair.score.toFixed(3)}, lost: ${pair.track.lostFrames} frames)`);
        }
      }
    }
    
    /** Calculate enhanced recovery score using robust features */
    calculateEnhancedRecoveryScore(track, detection) {
      let score = 0;
      let components = 0;
      
      // Distance component (inverse relationship)
      const distance = this.calculateTrackDetectionDistance(track, detection);
      const maxDistance = this.occlusionParams.reappearanceRadius;
      if (distance <= maxDistance) {
        score += (1 - distance / maxDistance) * 0.3;
        components++;
      } else {
        return 0; // Too far for recovery
      }
      
      // Enhanced appearance similarity
      if (this.enableReID && detection.feature && track.getEnhancedAppearanceScore) {
        const appearanceSim = track.getEnhancedAppearanceScore(detection.feature);
        score += appearanceSim * 0.5; // High weight for enhanced features
        components++;
      } else if (track.feature && detection.feature) {
        // Fallback to traditional appearance
        const appearanceCost = this._getAppearanceCost(track, detection, detection.feature);
        score += (1 - appearanceCost) * 0.4;
        components++;
      }
      
      // Size consistency
      const sizeRatio = Math.min(detection.w/track.w, track.w/detection.w) * 
                       Math.min(detection.h/track.h, track.h/detection.h);
      if (sizeRatio > 0.5) {
        score += sizeRatio * 0.2;
        components++;
      }
      
      // Motion prediction consistency
      if (track.vx !== undefined && track.vy !== undefined) {
        const predictedX = track.cx + track.vx * (track.lostFrames / 30);
        const predictedY = track.cy + track.vy * (track.lostFrames / 30);
        const detCenterX = detection.x + detection.w / 2;
        const detCenterY = detection.y + detection.h / 2;
        const predictionError = Math.sqrt((predictedX - detCenterX)**2 + (predictedY - detCenterY)**2);
        
        const maxPredictionError = Math.max(100, Math.hypot(track.w, track.h));
        if (predictionError <= maxPredictionError) {
          score += (1 - predictionError / maxPredictionError) * 0.15;
          components++;
        }
      }
      
      // Confidence decay based on lost frames
      const confidenceDecay = Math.pow(this.occlusionParams.confidenceDecay, track.lostFrames);
      score *= confidenceDecay;
      
      return components > 0 ? score : 0;
    }
    
    /** Calculate distance between track and detection */
    calculateTrackDetectionDistance(track, detection) {
      const trackCenterX = track.cx;
      const trackCenterY = track.cy;
      const detCenterX = detection.x + detection.w / 2;
      const detCenterY = detection.y + detection.h / 2;
      
      return Math.sqrt((trackCenterX - detCenterX)**2 + (trackCenterY - detCenterY)**2);
    }
    
    /** Perform enhanced recovery with feature validation */
    performEnhancedRecovery(track, detection) {
      // Reset lost frames
      track.lostFrames = 0;
      
      // Update position with enhanced validation
      this.updateTrackWithEnhancedFeatures(track, detection);
      
      // Reset occlusion state if present
      if (track.occlusionState) {
        track.occlusionState.isOccluded = false;
        track.occlusionState.occlusionFrames = 0;
      }
      
      // Boost confidence for successful recovery
      if (track.confidence !== undefined) {
        track.confidence = Math.min(1.0, track.confidence + 0.1);
      }
    }
    
    /** Calculate dominant color similarity for enhanced features */
    calculateDominantColorSimilarity(colors1, colors2) {
      if (!colors1 || !colors2 || colors1.length === 0 || colors2.length === 0) {
        return 0;
      }
      
      let totalSimilarity = 0;
      let comparisons = 0;
      
      // Compare each color in colors1 with best match in colors2
      for (const color1 of colors1) {
        let bestSimilarity = 0;
        
        for (const color2 of colors2) {
          // Convert to LAB for perceptual comparison
          const lab1 = this.rgbToLab(color1.r, color1.g, color1.b);
          const lab2 = this.rgbToLab(color2.r, color2.g, color2.b);
          
          // Calculate Delta E (CIE76)
          const deltaE = Math.sqrt(
            Math.pow(lab1.L - lab2.L, 2) +
            Math.pow(lab1.a - lab2.a, 2) +
            Math.pow(lab1.b - lab2.b, 2)
          );
          
          // Convert Delta E to similarity (0-1)
          const similarity = Math.exp(-deltaE / 50); // Adjust sensitivity
          bestSimilarity = Math.max(bestSimilarity, similarity);
        }
        
        totalSimilarity += bestSimilarity;
        comparisons++;
      }
      
      return comparisons > 0 ? totalSimilarity / comparisons : 0;
    }
    
    /** Convert RGB to LAB color space */
    rgbToLab(r, g, b) {
      // Normalize RGB values
      r /= 255;
      g /= 255;
      b /= 255;
      
      // Convert to XYZ
      r = r > 0.04045 ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
      g = g > 0.04045 ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
      b = b > 0.04045 ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;
      
      let x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
      let y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000;
      let z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;
      
      x = x > 0.008856 ? Math.pow(x, 1/3) : (7.787 * x) + 16/116;
      y = y > 0.008856 ? Math.pow(y, 1/3) : (7.787 * y) + 16/116;
      z = z > 0.008856 ? Math.pow(z, 1/3) : (7.787 * z) + 16/116;
      
      return {
        L: (116 * y) - 16,
        a: 500 * (x - y),
        b: 200 * (y - z)
      };
    }
    
    /** Fallback to original track recovery method */
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
      if (!track.feature || !detFeature) return 1.0;
      
      // Use enhanced appearance scoring if available
      if (track.getEnhancedAppearanceScore) {
        const similarity = track.getEnhancedAppearanceScore(detFeature);
        return 1.0 - similarity; // Convert similarity to cost
      }
      
      // Fallback to basic appearance cost calculation
      if (!track.feature.color || !detFeature.color) return 1.0;

      // 1. Color Feature Cost (Cosine Distance)
      const colorCost = 1 - cosineSimilarity(track.feature.color, detFeature.color);

      // 2. Geometric Feature Cost (Normalized Euclidean Distance)
      let geoDist = 0;
      const geoWeights = [1, 1, 2, 0.5, 0.5, 0.8, 0.8, 0.3, 0.3]; // Extended weights for enhanced geometric features
      const geoLength = Math.min(track.feature.geometric.length, detFeature.geometric.length, geoWeights.length);
      
      for (let i = 0; i < geoLength; i++) {
        const diff = track.feature.geometric[i] - detFeature.geometric[i];
        geoDist += (diff * diff) * geoWeights[i];
      }
      const geometricCost = Math.min(1.0, Math.sqrt(geoDist) / geoLength);

      // 3. Enhanced features cost (if available)
      let enhancedCost = 0;
      let hasEnhanced = false;
      
      if (track.feature.enhanced && detFeature.enhanced) {
        // Color centroid cost
        if (track.feature.enhanced.colorCentroid && detFeature.enhanced.colorCentroid) {
          const centroidDist = Math.sqrt(
            Math.pow(track.feature.enhanced.colorCentroid.x - detFeature.enhanced.colorCentroid.x, 2) +
            Math.pow(track.feature.enhanced.colorCentroid.y - detFeature.enhanced.colorCentroid.y, 2)
          );
          enhancedCost += centroidDist * 0.4; // Weight for color centroid
        }
        
        // Overall brightness cost
        if (track.feature.enhanced.overallBrightness !== undefined && detFeature.enhanced.overallBrightness !== undefined) {
          const brightnessDiff = Math.abs(track.feature.enhanced.overallBrightness - detFeature.enhanced.overallBrightness) / 255;
          enhancedCost += brightnessDiff * 0.3; // Weight for brightness
        }
        
        // Color variance cost
        if (track.feature.enhanced.colorVariance !== undefined && detFeature.enhanced.colorVariance !== undefined) {
          const maxVariance = Math.max(track.feature.enhanced.colorVariance, detFeature.enhanced.colorVariance, 1);
          const varianceDiff = Math.abs(track.feature.enhanced.colorVariance - detFeature.enhanced.colorVariance) / maxVariance;
          enhancedCost += varianceDiff * 0.3; // Weight for color variance
        }
        
        hasEnhanced = true;
      }

      // 4. Combine costs with adaptive weighting
      if (hasEnhanced) {
        // Use enhanced features when available: 30% color, 30% geometric, 40% enhanced
        return 0.3 * colorCost + 0.3 * geometricCost + 0.4 * Math.min(1.0, enhancedCost);
      } else {
        // Fallback to basic features: 50% color, 50% geometric
        return 0.5 * colorCost + 0.5 * geometricCost;
      }
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
