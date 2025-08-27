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
      this.alpha = opts.alpha ?? 0.6;
      this.beta = opts.beta ?? 0.4;
      this.locked = !!opts.locked;
      this.color = COLOR_POOL[(id - 1) % COLOR_POOL.length];
      this.feature = null; // EMA feature
      this.lostFrames = 0;
      this.hits = 1;
      this.trajectory = [];
      this._pushTrajectory();
    }

    get bbox() { return { x: this.cx - this.w / 2, y: this.cy - this.h / 2, w: this.w, h: this.h }; }

    _pushTrajectory() {
      this.trajectory.push({ x: this.cx, y: this.cy });
      if (this.trajectory.length > 60) this.trajectory.shift();
    }

    /** Constant-velocity alpha-beta prediction */
    predict() {
      const xp = this.cx + this.vx;
      const yp = this.cy + this.vy;
      // small damping to avoid runaway velocity
      this.vx *= 0.98; this.vy *= 0.98;
      this.cx = xp; this.cy = yp;
      this._pushTrajectory();
    }

    /** Update with new detection bbox and optional appearance feature */
    update(detBbox, feature) {
      const zx = detBbox.x + detBbox.w / 2;
      const zy = detBbox.y + detBbox.h / 2;
      // prediction step
      const xp = this.cx + this.vx;
      const yp = this.cy + this.vy;
      // innovation
      const rx = zx - xp; const ry = zy - yp;
      // update
      this.cx = xp + this.alpha * rx;
      this.cy = yp + this.alpha * ry;
      this.vx = this.vx + this.beta * rx;
      this.vy = this.vy + this.beta * ry;
      // smooth size with EMA
      this.w = 0.7 * this.w + 0.3 * detBbox.w;
      this.h = 0.7 * this.h + 0.3 * detBbox.h;
      this.lostFrames = 0;
      this.hits += 1;
      this._pushTrajectory();
      // EMA update for appearance
      if (feature) {
        if (!this.feature || this.feature.length !== feature.length) {
          this.feature = feature;
        } else {
          const mu = 0.3;
          for (let i = 0; i < this.feature.length; i++) {
            this.feature[i] = (1 - mu) * this.feature[i] + mu * feature[i];
          }
          l2normalize(this.feature);
        }
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
      this.gatingBase = opts.gatingBase ?? 60; // px baseline for gating radius
      this.maxLostUnlocked = opts.maxLostUnlocked ?? 45; // frames
      this.maxLostLocked = opts.maxLostLocked ?? 150; // frames
      // Matching weights
      this.wIoU = opts.wIoU ?? 0.45;
      this.wApp = opts.wApp ?? 0.45;
      this.wCtr = opts.wCtr ?? 0.07;
      this.wRatio = opts.wRatio ?? 0.03;
      this.costThreshold = opts.costThreshold ?? 0.85;
      this.autoCreate = !!opts.autoCreate; // default false: only create via click lock
    }

    /** Predict-only step for frames without new detections */
    predictOnly() {
      for (const t of this.tracks) {
        t.predict();
        t.lostFrames += 1;
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

    /** Internal: associate a set of tracks with detections using greedy min-cost */
    _associateAndUpdate(trackList, detections, unmatchedDetIdx, isLockedPass) {
      if (trackList.length === 0 || unmatchedDetIdx.size === 0) return;

      // Precompute cost matrix
      const pairs = [];
      for (let ti = 0; ti < trackList.length; ti++) {
        const t = trackList[ti];
        const tb = t.bbox;
        const gating = Math.max(this.gatingBase, 0.5 * Math.hypot(t.w, t.h));
        for (const di of unmatchedDetIdx) {
          const d = detections[di];
          const db = { x: d.x, y: d.y, w: d.w, h: d.h };
          const i = iou(tb, db);
          const ctr = centerDistance(tb, db);
          if (i < 0.01 && ctr > gating * 3) continue; // gate out far candidates

          // appearance distance
          let app = 1;
          if (this.enableReID && t.feature && d.feature) {
            app = clamp(1 - cosineSimilarity(t.feature, d.feature), 0, 1);
          } else if (this.enableReID && !t.feature && d.feature) {
            // initialize track feature on first opportunity
            t.feature = d.feature;
            app = 0.5; // neutral distance
          }

          const ratioT = (t.w / (t.h + 1e-3));
          const ratioD = (d.w / (d.h + 1e-3));
          const ratioDiff = Math.min(1, Math.abs(ratioT - ratioD));

          const ctrNorm = clamp(ctr / (gating * 2), 0, 1);

          // Adjust weights if locked
          const wIoU = isLockedPass ? Math.max(0.35, this.wIoU - 0.1) : this.wIoU;
          const wApp = isLockedPass ? Math.min(0.6, this.wApp + 0.15) : this.wApp;

          const cost = wIoU * (1 - i) + wApp * app + this.wCtr * ctrNorm + this.wRatio * ratioDiff;
          pairs.push({ cost, ti, di });
        }
      }

      // Greedy assignment by increasing cost
      pairs.sort((a, b) => a.cost - b.cost);
      const usedT = new Set();
      for (const p of pairs) {
        if (p.cost > this.costThreshold) break;
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
      return this._createTrackFromDet({ ...chosen.box }, feature, true);
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
