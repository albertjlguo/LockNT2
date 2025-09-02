# YouTube Live AI Object Tracking System

Looking for the Chinese documentation? See `README.zh.md`.

## Project Overview

This project is a web-based, real-time object detection and tracking system optimized for YouTube Live streams. It adopts an innovative split architecture of frontend AI inference plus backend video proxy, delivering a low-cost, high-performance multi-object tracking solution.

### Core Features

🎯 Real-Time Object Detection
- TensorFlow.js COCO‑SSD model; supports people and vehicle detection
- Adaptive confidence threshold to balance precision and recall
- Perspective-aware bounding-box refinement to improve results in complex scenes

🔄 Interactive Object Tracking
- Click any detection to lock it and create a persistent tracking relationship
- Global optimal assignment with the Hungarian algorithm for multi-target association
- Occlusion recovery: robust re-identification after short-term disappearance

📺 Smooth Video Experience
- MJPEG streaming for low-latency real-time playback
- Adaptive frame buffering smooths network jitter
- Smart frame skipping to maintain a stable frame rate

🎮 Intuitive UI
- Real-time trajectory visualization with optional history playback
- Multi-target status panel with confidence and tracking state
- Tunable detection parameters for different scenarios

### Technical Highlights
- Zero server-side AI cost: all inference on the client
- Unlimited horizontal scale: each user handles their own compute
- Privacy-friendly: no server-side video storage
- Cross-platform: modern browsers with WebGL acceleration

## Quickstart (macOS)

### Requirements
- macOS 10.15+ (macOS 12+ recommended)
- Python 3.11+ (aligns with `pyproject.toml`)
- Modern browser (Chrome 100+, Safari 15+, Firefox 100+)
- Internet access for YouTube Live

### Fast Install

```bash
# 1) Clone the project
git clone https://github.com/albertjlguo/LockNT2.git
cd LockNT2

# 2) (Optional) Create a virtualenv
python3 -m venv venv
source venv/bin/activate

# 3) Install dependencies
pip install uv && uv pip install -r uv.lock
# Or
# pip install -r requirements.txt

# 4) Start the app
python main.py
```

### Verify
On success you should see something like:

```
* Running on http://127.0.0.1:5000
* Debug mode: on
INFO:werkzeug: * Running on all addresses (0.0.0.0)
```

### Get Started
1) Open `http://127.0.0.1:5000`
2) Paste any YouTube Live URL
3) Click “Start Streaming”
4) Click a detection box in the video to start tracking

### Troubleshooting

Common fixes:

```bash
# Python version
python3 --version  # ensure ≥ 3.11

# Dependency install errors
pip install --upgrade pip
pip install opencv-python-headless  # if GUI libs are missing

# Port 5000 in use
lsof -ti:5000 | xargs kill -9

# macOS permissions/tools
sudo xcode-select --install
```

Performance tips:
- Use Chrome for best WebGL performance
- Close other GPU-heavy applications
- Keep a stable network (10+ Mbps recommended)

## Architecture & Design

### Core Architectural Decisions

The system uses a split architecture: frontend AI inference in the browser and a backend video proxy. This design follows deep technical research and trade-off analysis.

#### 1) Frontend AI Inference

Background: Traditional server-side inference is costly and hard to scale, especially for multi-user, real-time video analysis.

Evaluated options:
- Server-side: NVIDIA T4/V100 + TensorRT — high GPU cost ($0.5–2/hr/GPU)
- Edge devices: complex deployments and maintenance
- Browser-side: TensorFlow.js + WebGL — near-zero marginal cost

Final choice: TensorFlow.js COCO‑SSD in the browser.

```javascript
// Model loading and inference tuning
const model = await cocoSsd.load({
  base: 'mobilenet_v2',  // balance between accuracy and speed
  modelUrl: undefined    // leverage CDN cache
});
```

Advantages:
- Zero marginal cost: each user runs their own inference
- Horizontal scale: theoretically supports massive concurrency
- Privacy: video never leaves the device for analysis
- Low latency: no round-trip for inference

Trade-offs:
- Dependent on client device performance
- Accuracy constrained by lightweight models

#### 2) Live Stream Acquisition Challenges & Solutions

Challenge: Browser same-origin policy prevents direct pixel access from YouTube iframes; CORS limits apply.

Explored approaches:

1) Direct iframe embedding ❌
```html
<!-- YouTube iframe cannot expose frames for analysis -->
<iframe src="https://youtube.com/embed/live_stream"></iframe>
```
Limitation: No access to raw frames; cannot run AI.

2) WebRTC ❌
```javascript
const pc = new RTCPeerConnection();
```
Limitation: YouTube does not provide WebRTC endpoints for direct frame access.

3) Backend proxy ✅ (chosen)
```python
# yt-dlp + OpenCV proxy pipeline
import yt_dlp
import cv2

# Resolve the actual stream URL
ydl = yt_dlp.YoutubeDL()
info = ydl.extract_info(youtube_url)
stream_url = info['url']

# Read and transcode via OpenCV
cap = cv2.VideoCapture(stream_url)
```

Final pipeline:

YouTube Live → yt‑dlp resolve → OpenCV capture → MJPEG encode → Browser display
  ↓             ↓                ↓                ↓                 ↓
 HLS        Stream URL        Frames           HTTP MJPEG       Canvas

Key implementation:

```python
# routes.py — MJPEG generator
def video_feed_mjpeg():
    def generate():
        while is_processing:
            frame = stream_processor.get_buffered_frame(max_age_ms=50)
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
```

Performance optimizations:
- 3-frame ring buffer to reduce memory usage
- JPEG quality ~75% for balance of speed and quality
- Adaptive frame skipping to keep ~30 FPS steady
- Exponential backoff reconnection for network issues

Update Notes:
- Deprecated the old polling endpoint (`/video_feed`)
- Added `/video_feed_mjpeg` for smoother playback

### Tracking Algorithm: Design & Details

Design goals:
- Real-time: ~30 FPS experience
- Robustness: occlusion, fast motion, target crossings
- Interactivity: click-to-lock on any target
- Lightweight: suitable for browsers

Algorithm architecture:

```javascript
// tracker.js — core structure
class Tracker {
  constructor() {
    this.tracks = [];
    this.idManager = new IDManager();        // ID lifecycle
    this.hungarian = new HungarianAlgorithm();
    this.encoder = new AppearanceEncoder();  // appearance features
  }

  async update(detections, videoContext) {
    // 1) Predict with motion model
    for (const track of this.tracks) track.predict();

    // 2) Associate with Hungarian algorithm
    this._associateDetectionsToTracks(detections);

    // 3) Update matched tracks; create new ones
    this._updateTracksAndCreateNew(detections);

    // 4) Prune dead tracks
    this._pruneDeadTracks();
  }
}
```

Key components:

1) Motion model (simplified Kalman filter)

```javascript
// State vector: [x, y, vx, vy]
class KalmanFilter {
  predict() {
    // constant-velocity model
    this.x[0] += this.x[2] * dt;
    this.x[1] += this.x[3] * dt;
    return { x: this.x[0], y: this.x[1] };
  }
}
```

2) Appearance features

```javascript
class AppearanceEncoder {
  encode(ctx, bbox) {
    // Extract HSV histogram for the region
    const hist = this._computeHSVHistogram(ctx, bbox);
    return l2normalize(hist);
  }

  distance(feat1, feat2) {
    return 1 - cosineSimilarity(feat1, feat2);
  }
}
```

3) Hungarian assignment

```javascript
_associateDetectionsToTracks(detections) {
  const costMatrix = this._buildCostMatrix(this.tracks, detections);
  const assignments = this.hungarian.solve(costMatrix);
  this._applyAssignments(assignments, costMatrix);
}
```

Cost function (example):

```text
cost = w_iou * (1 - IoU) +
       w_app * appearance_distance +
       w_center * normalized_center_distance +
       w_motion * motion_inconsistency
```

### Handling Loss and Occlusion

Occlusion detection and state management:

```javascript
class Track {
  checkOcclusionState(frameCount) {
    const occlusionThreshold = this.locked ? 2 : 4;

    if (this.lostFrames > occlusionThreshold) {
      this.occlusionState.isOccluded = true;
      this.occlusionState.occlusionStartFrame = frameCount;

      // adaptive search radius based on velocity
      const velocityMagnitude = Math.sqrt(vx*vx + vy*vy);
      this.occlusionState.searchRadius = baseRadius + velocityMagnitude * 2;
    }
  }
}
```

State transitions:
- Tracking → Short-term lost (2–4 frames unmatched)
- Short-term lost → Occluded (keep predicting and widen search)
- Occluded → Recovered (matched again)
- Long occlusion → Remove track (e.g., > 80 frames)

Predictive tracking:

```javascript
predict() {
  const predicted = this.kalmanFilter.predict();
  this.cx = predicted.x;
  this.cy = predicted.y;

  if (this.occlusionState.isOccluded) {
    this.occlusionState.searchRadius *= 1.08;   // expand search
    this.occlusionState.confidence *= 0.95;     // decay confidence
  }
}
```

Search strategy:
- Normal: fixed radius (~0.8× target size)
- Occluded: dynamically expand (max ~250 px)
- Fast motion: directionally biased by velocity vector

Recovery mechanism (multi-cue scoring):

```javascript
_attemptTrackRecovery(detections, unmatchedDetIdx) {
  for (const track of this.lostTracks) {
    for (const detIdx of unmatchedDetIdx) {
      const det = detections[detIdx];
      const score = this._calculateRecoveryScore(track, det);
      if (score > this.recoveryThreshold) {
        this._recoverTrack(track, det);
      }
    }
  }
}

_calculateRecoveryScore(track, detection) {
  let score = 0;
  const predictedPos = track.getPredictedPosition();
  const positionScore = this._calculatePositionScore(predictedPos, detection); // 40%
  score += positionScore * 0.4;

  if (track.feature && detection.feature) { // 35%
    const appearanceScore = cosineSimilarity(track.feature, detection.feature);
    score += appearanceScore * 0.35;
  }

  const sizeScore = this._calculateSizeConsistency(track, detection); // 15%
  score += sizeScore * 0.15;

  const motionScore = this._calculateMotionConsistency(track, detection); // 10%
  score += motionScore * 0.10;

  return score;
}
```

Complex scenarios:
- Crowded scenes: raise matching threshold (e.g., 0.65 → 0.75), limit new tracks, increase appearance weight
- Fast motion: enlarge gating radius, increase motion weight, shorten occlusion threshold
- Lighting changes: use LAB space, adaptive appearance learning rate, multi-template appearance model

---

## Technical Q&A

### Project Overview & Goals

Q: What’s the core goal of the project? What can users do and see?

A: Provide a zero-cost, high-performance real-time tracking solution for any YouTube Live stream.

User experience:
- Paste a YouTube Live URL and start analysis with one click
- Click a detection box to lock and persist tracking
- Observe trajectories, confidence, speed, and status per target
- Tune detection thresholds for different scenarios
- View stats like counts, dwell time, heatmaps

Visuals:
- Colored bounding boxes for different targets
- Trajectory lines for history
- Highlighted style for locked targets
- Floating labels with confidence and IDs

Q: Why run detection and tracking in the browser? Trade-offs vs server-side?

A: After extensive evaluation, browser-side inference is a strategic decision.

Cost comparison:

| Approach | Server Cost | Scalability | Privacy | Latency |
|---------:|------------:|------------:|--------:|--------:|
| Server inference | $0.5–2/hr/GPU | Limited by hardware | Upload data | Network + compute |
| Browser inference | $0 | Virtually unlimited | Local only | Compute only |

Trade-offs:
- Pros: zero marginal cost, strong privacy, massive scale
- Cons: device-dependent performance; lightweight models may trade off some accuracy

Q: Supported classes and baseline performance?

A: Supported classes: Person and Vehicles (car, truck, motorcycle, etc.).

Baseline guidance:

| Metric | Minimum | Recommended | Best |
|-------:|--------:|------------:|-----:|
| Resolution | 480p | 720p | 1080p |
| FPS | 15 | 30 | 30 |
| Concurrent tracks | 3 | 5 | 10 |
| Browser | Chrome 80+ | Chrome 90+ | Chrome 100+ |
| Device RAM | 4 GB | 8 GB | 16 GB + GPU |

### Live Stream Acquisition & Processing

Q: How do you fetch YouTube Live frames and render them in a web app?

A: Backend proxy + MJPEG streaming.

Stack:

```python
yt-dlp     # YouTube stream resolving (youtube-dl replacement)
OpenCV     # Video processing and JPEG encoding
Flask      # Web server
```

Data flow:

YouTube HLS → yt‑dlp → OpenCV → JPEG → MJPEG → Browser

Key snippet:

```python
# stream_processor.py (excerpt)
class StreamProcessor:
    def start_processing(self):
        ydl_opts = {'format': 'best[height<=720]'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(self.youtube_url, download=False)
            stream_url = info['url']

        self.cap = cv2.VideoCapture(stream_url)

        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 75])
                self.frame_buffer.append(buffer.tobytes())
```

Q: How to handle CORS/CSP/iframe constraints?

A: Use a backend proxy to resolve and re-stream frames over HTTP MJPEG; the browser consumes MJPEG without cross-origin pixel access. [... omitted 145 of 401 lines ...]

Inference pipeline optimization:

```javascript
// detection.js — pipeline
class DetectionPipeline {
  constructor() {
    this.model = null;
    this.inferenceQueue = [];
    this.isInferring = false;
    this.targetFPS = 30;
    this.detectionInterval = 100; // ms
  }

  async processFrame(videoElement) {
    if (this.isInferring) return this.lastDetections; // reuse

    this.isInferring = true;
    try {
      const detections = await this.model.detect(videoElement);
      this.lastDetections = this.postProcessDetections(detections);
      return this.lastDetections;
    } finally {
      this.isInferring = false;
    }
  }

  postProcessDetections(rawDetections) {
    return rawDetections
      .filter(det => det.score > this.confidenceThreshold)
      .filter(det => ['person', 'car'].includes(det.class))
      .map(det => this.normalizeDetection(det));
  }
}
```

Performance strategies:
- Smart frame skipping: reuse last results when inference is busy
- Async pipeline: decouple inference from rendering
- Batch post-processing: handle multiple boxes efficiently
- Memory pool: reuse objects to reduce GC

Detection quality management:

```javascript
class DetectionQualityManager {
  constructor() {
    this.qualityHistory = [];
    this.adaptiveThreshold = 0.3;
    this.stabilityWindow = 10;
  }

  assessDetectionQuality(detections) {
    const avgConfidence = this.calculateAvgConfidence(detections);
    const detectionStability = this.calculateStability(detections);
    const spatialConsistency = this.calculateSpatialConsistency(detections);

    const qualityScore = (
      avgConfidence * 0.4 +
      detectionStability * 0.3 +
      spatialConsistency * 0.3
    );

    this.updateAdaptiveThreshold(qualityScore);

    return {
      score: qualityScore,
      threshold: this.adaptiveThreshold,
      shouldFilter: qualityScore < 0.6
    };
  }
}
```

False positive handling:
- Temporal consistency checks
- Spatial sanity checks on box positions/sizes
- Class-specific dynamic thresholds
- Outlier removal for obvious errors

### Interactive Tracking System

Q: After clicking a detection, how is persistent tracking established?

A: Click-to-lock mechanism.

```javascript
canvas.addEventListener('click', (event) => {
  const clickPoint = this.getCanvasCoordinates(event);
  const hitDetection = this.findHitDetection(clickPoint, detections);
  if (!hitDetection) return;

  const existingTrack = this.tracker.findTrackByDetection(hitDetection);
  if (existingTrack) {
    existingTrack.setLocked(true);
    this.showLockConfirmation(existingTrack.id);
  } else {
    const newTrack = this.tracker.createLockedTrack(hitDetection);
    this.showTrackCreated(newTrack.id);
  }
});
```

Locked-track policy:
- Extended lifetime: tolerate longer losses (e.g., 80 vs 30 frames)
- Priority matching: prefer locked tracks
- Stronger appearance learning: update templates more often
- Enhanced recovery: dedicated logic for locked tracks

ID switch prevention:

```javascript
class IDSwitchPrevention {
  constructor() {
    this.trackHistory = new Map();
    this.crossValidationWindow = 5;
  }

  validateAssignment(trackId, detectionId, assignment) {
    const history = this.trackHistory.get(trackId) || [];
    const recent = history.slice(-this.crossValidationWindow);
    const consistencyScore = this.calculateConsistencyScore(recent, assignment);
    const crossValidationScore = this.performCrossValidation(trackId, detectionId, assignment);
    return {
      isValid: consistencyScore > 0.7 && crossValidationScore > 0.6,
      confidence: Math.min(consistencyScore, crossValidationScore),
      reason: this.getValidationReason(consistencyScore, crossValidationScore)
    };
  }
}
```

Guards:
- Historical motion consistency
- Cross-validation to avoid better matches elsewhere
- Size consistency checks
- Physics-based motion sanity

### Trajectory Visualization & Coordinate Sync

Coordinate transform manager:

```javascript
class CoordinateManager {
  constructor(videoElement, canvasElement) {
    this.video = videoElement;
    this.canvas = canvasElement;
    this.scaleX = 1; this.scaleY = 1;
    this.offsetX = 0; this.offsetY = 0;
  }

  updateTransform() {
    this.scaleX = this.canvas.width / this.video.videoWidth;
    this.scaleY = this.canvas.height / this.video.videoHeight;

    const videoAspect = this.video.videoWidth / this.video.videoHeight;
    const canvasAspect = this.canvas.width / this.canvas.height;
    if (videoAspect > canvasAspect) {
      this.scaleY = this.scaleX;
      this.offsetY = (this.canvas.height - this.video.videoHeight * this.scaleY) / 2;
    } else {
      this.scaleX = this.scaleY;
      this.offsetX = (this.canvas.width - this.video.videoWidth * this.scaleX) / 2;
    }
  }

  videoToCanvas(x, y) {
    return { x: x * this.scaleX + this.offsetX,
             y: y * this.scaleY + this.offsetY };
  }
}
```

Trajectory renderer and predictive recovery strategies are implemented to keep drawings and coordinates in sync across aspect ratios and letterboxing. [... omitted 145 of 401 lines ...]

Robustness highlights:
- Multi-modal recovery (position, appearance, motion, size)
- Adaptive search radius during occlusion
- Confidence management with gradual decay
- Long-term memory of appearance templates

### Performance, Resources & Stability

Resource management:
- Track history capped (e.g., 100 points per track)
- Appearance template cleanup (remove low-similarity templates)
- Detection history capped (e.g., last 50 frames)
- Periodic memory cleanup

Performance monitoring:
- FPS monitor (goal 25+)
- Per-stage timing (detection, tracking, rendering)
- Auto-degrade when under load
- Resource usage reporting (memory/CPU)

Error recovery:
- Layered error handling for detection/tracking/rendering
- Safe mode with simplified processing on severe errors
- State reset when necessary

Complexity analysis:

| Component | Time | Space |
|----------:|-----:|------:|
| Detection inference | O(1) | O(1) |
| Feature extraction | O(n) | O(n×d) |
| Hungarian | O(n^3) | O(n^2) |
| Track update | O(n) | O(n×h) |

Device benchmarks:

| Device | FPS | Max tracks | Memory |
|-------:|----:|-----------:|-------:|
| High-end desktop | 30 | 15+ | <200 MB |
| Mid-range laptop | 25 | 8–10 | <150 MB |
| Low-end device | 15 | 3–5 | <100 MB |
| Mobile device | 20 | 5–8 | <120 MB |

### Security, Compliance & Copyright

Legal compliance:
- Live streams only; no recorded videos
- No storage; frames processed in memory only
- Analysis-only; no downloading/recording features
- Session duration limits (e.g., 1 hour)

Privacy:
- Local inference in the browser
- Data minimization: only necessary tracking data
- Auto cleanup after session ends
- No user tracking or personal data collection

Technical compliance:
- Respect YouTube terms and API usage
- Reasonable rate limiting
- Graceful handling of restricted or removed content

GDPR/Privacy measures:
- Lawful basis: explicit user consent
- Data subject rights: access/delete/correct
- Privacy by design and default
- Regular privacy impact assessments

### Testing, Deployment & Docs

Testing strategy:
- Unit tests for core algorithmic components (>85% target)
- Integration tests for end-to-end tracking
- Performance benchmarks across devices
- Compatibility tests across browsers/resolutions

Scenarios:
- Basic tracking (single/multi-target accuracy)
- Occlusion handling (partial/complete, recovery ability)
- Edge cases (enter/exit frame, rapid motion)
- Stress (dense scenes, long runs)

Deployment docs:
- Environment requirements (Python 3.11+, modern browser)
- Dependency management and versions
- Server configuration and performance tuning
- Troubleshooting

Monitoring & maintenance:
- Real-time FPS and timing metrics
- Error logs and analysis
- Model updates and algorithm rollouts

### Optional Advanced Features & Roadmap

Advanced analytics:
- Behavior analysis (dwell time, movement patterns)
- Heatmaps of activity
- Stats export (CSV/JSON)
- Anomaly detection

Technical enhancements:
- More classes (bicycle, animals, etc.)
- 3D tracking via depth estimation
- Multi-camera fusion
- Edge deployment

UX improvements:
- Customizable UI (layout/themes)
- Data export
- Real-time notifications
- RESTful APIs for integrations

Future directions:
- Stronger detection and ReID models
- Real-time semantic segmentation
- Predictive analytics
- Cloud-assisted updates and optimization

---

## Summary

This project delivers a zero-cost, high-performance real-time tracking system for YouTube Live with:

🎯 Technical innovation
- Browser-side AI inference (zero server cost, massive scale)
- Multi-modal tracking (appearance + motion + geometry)
- Intelligent occlusion handling (predictive tracking, multi-stage recovery)
- Adaptive performance tuning based on device capability

🛡️ Robustness
- ID switch prevention via multi-check safeguards
- Layered error recovery
- Smart resource management and monitoring
- Stable long-duration operation

📊 Practical value
- Plug-and-play: paste a URL and go
- Interactive: click-to-lock for persistent tracking
- Real-time visualization and stats
- Works across desktop and mobile

## FAQ (Highlights)

1) Playback latency?
- MJPEG aims for smoothness by default; adjust JPEG quality or detection cadence to trade CPU/bandwidth.

2) “MJPEG stream error” when stopping?
- Fixed: stopping no longer triggers error prompts or reconnection.

3) Install issues?
- Prefer `uv pip install -r uv.lock`. If problems persist, install key deps individually.

## Technical Architecture (Addendum)

Backend optimizations
- Frame buffer: 3-frame ring buffer to reduce latency
- Adaptive encoding: JPEG quality ~75%, progressive enabled
- Hardware accel: supports H.264 hardware acceleration (where applicable)
- Performance monitoring: processing time stats and drop counts

Frontend optimizations
- Smart canvas rendering
- Adaptive detection throttling (300–600 ms) based on target stability
- Tracking-first mode: rely more on prediction for locked targets
- Exponential backoff reconnection
- Lightweight prediction mode to reduce redraws
- Fusion of predicted and detected positions to reduce jitter

## Latest Major Updates

v2.2 — Hungarian Algorithm Optimal Assignment
- Integrated Hungarian algorithm for globally optimal assignments
- ID manager with delayed release to prevent reuse
- Significant accuracy gains in crowded scenes
- Multi-term cost matrix (IoU, appearance, motion consistency)
- Strong conflict protection (gating and thresholds)
- Backward-compatible with existing UI/UX and click-to-lock
- Dual modes: standard (Hungarian) and tracking-first retained

v2.1 — ID Switch Prevention System
- Dramatically reduces ID switches in dense crowds
- Enhanced appearance features (higher-dim features, multi-template model)
- Cross-validation, trajectory consistency, and size checks
- Scene-adaptive thresholds (e.g., 0.65–0.85)
- Intelligent conflict resolution and conservative track creation
- Consistency rewards via historical matching checks

v2.0 — Robust Tracking Overhaul
- Solves “locked target drifts after ~2s” issue
- Tracking-first mode for long-term stability
- Predictive/detection fusion; adaptive detection cadence (300–600 ms)
- Improved motion prediction with smoothed acceleration and confidence weighting
- Multi-template appearance memory and smart recovery

v1.x — Foundations
- Enhanced tracking (occlusion handling, predictive tracking, multi-template appearance matching)
- Bounding box optimization for person and vehicle classes
- UI controls for per-class confidence thresholds
- Second-order motion model with acceleration and history
- MJPEG improvements for smoother playback
- Frame buffer management to reduce jitter
- Bilingual UI support for labels
- Real-time FPS and timing analysis

---

For the original Chinese documentation, see `README.zh.md`.
