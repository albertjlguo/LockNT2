/**
 * Kalman Filter for Object Tracking
 * 用于目标追踪的卡尔曼滤波器
 * 
 * State vector: [x, y, vx, vy, w, h]
 * - (x, y): center position / 中心位置
 * - (vx, vy): velocity / 速度
 * - (w, h): width and height / 宽度和高度
 */

class KalmanFilter {
    constructor(initialBbox, options = {}) {
        // State dimension: 6 (x, y, vx, vy, w, h)
        // Measurement dimension: 4 (x, y, w, h)
        this.stateDim = 6;
        this.measureDim = 4;
        
        // Configuration parameters
        this.processNoise = options.processNoise || 2.0;
        this.measurementNoise = options.measurementNoise || 8.0;
        this.velocityNoise = options.velocityNoise || 10.0;
        
        // Initialize state vector [x, y, vx, vy, w, h]
        this.state = new Float32Array([
            initialBbox.x + initialBbox.w / 2, // x center
            initialBbox.y + initialBbox.h / 2, // y center
            0, // vx
            0, // vy
            initialBbox.w, // width
            initialBbox.h  // height
        ]);
        
        // State transition matrix (constant velocity model)
        this.F = this.createTransitionMatrix(1/30); // Assume 30 FPS
        
        // Measurement matrix (we observe x, y, w, h)
        this.H = new Float32Array([
            1, 0, 0, 0, 0, 0,  // x
            0, 1, 0, 0, 0, 0,  // y
            0, 0, 0, 0, 1, 0,  // w
            0, 0, 0, 0, 0, 1   // h
        ]);
        
        // Process noise covariance matrix Q
        this.Q = this.createProcessNoiseMatrix();
        
        // Measurement noise covariance matrix R
        this.R = this.createMeasurementNoiseMatrix();
        
        // Error covariance matrix P (initial uncertainty)
        this.P = this.createInitialCovarianceMatrix();
        
        // Identity matrix
        this.I = this.createIdentityMatrix(this.stateDim);
        
        // Temporary matrices for calculations
        this.tempState = new Float32Array(this.stateDim);
        this.tempMeasure = new Float32Array(this.measureDim);
        
        // Tracking quality metrics
        this.confidence = 1.0;
        this.innovationHistory = [];
        this.maxHistory = 10;
    }
    
    createTransitionMatrix(dt) {
        const F = new Float32Array(this.stateDim * this.stateDim);
        
        // Identity matrix
        for (let i = 0; i < this.stateDim; i++) {
            F[i * this.stateDim + i] = 1.0;
        }
        
        // Position update: x = x + vx * dt, y = y + vy * dt
        F[0 * this.stateDim + 2] = dt; // x += vx * dt
        F[1 * this.stateDim + 3] = dt; // y += vy * dt
        
        return F;
    }
    
    createProcessNoiseMatrix() {
        const Q = new Float32Array(this.stateDim * this.stateDim);
        
        Q[0 * this.stateDim + 0] = this.processNoise; // x
        Q[1 * this.stateDim + 1] = this.processNoise; // y
        Q[2 * this.stateDim + 2] = this.velocityNoise; // vx
        Q[3 * this.stateDim + 3] = this.velocityNoise; // vy
        Q[4 * this.stateDim + 4] = this.processNoise * 0.1; // w
        Q[5 * this.stateDim + 5] = this.processNoise * 0.1; // h
        
        return Q;
    }
    
    createMeasurementNoiseMatrix() {
        const R = new Float32Array(this.measureDim * this.measureDim);
        
        R[0 * this.measureDim + 0] = this.measurementNoise; // x
        R[1 * this.measureDim + 1] = this.measurementNoise; // y
        R[2 * this.measureDim + 2] = this.measurementNoise * 0.3; // w
        R[3 * this.measureDim + 3] = this.measurementNoise * 0.3; // h
        
        return R;
    }
    
    createInitialCovarianceMatrix() {
        const P = new Float32Array(this.stateDim * this.stateDim);
        
        P[0 * this.stateDim + 0] = 50; // x
        P[1 * this.stateDim + 1] = 50; // y
        P[2 * this.stateDim + 2] = 500; // vx
        P[3 * this.stateDim + 3] = 500; // vy
        P[4 * this.stateDim + 4] = 50; // w
        P[5 * this.stateDim + 5] = 50; // h
        
        return P;
    }
    
    createIdentityMatrix(size) {
        const I = new Float32Array(size * size);
        for (let i = 0; i < size; i++) {
            I[i * size + i] = 1.0;
        }
        return I;
    }
    
    predict(dt = 1/30) {
        // Update transition matrix
        this.F = this.createTransitionMatrix(dt);
        
        // Predict state: x = F * x
        this.matrixVectorMultiply(this.F, this.state, this.tempState);
        this.state.set(this.tempState);
        
        // Predict covariance: P = F * P * F^T + Q
        this.predictCovariance();
        
        // Apply velocity damping
        this.state[2] *= 0.98; // vx damping
        this.state[3] *= 0.98; // vy damping
        
        // Ensure positive dimensions
        this.state[4] = Math.max(1, this.state[4]);
        this.state[5] = Math.max(1, this.state[5]);
    }
    
    update(measurement, confidence = 1.0) {
        const z = new Float32Array([
            measurement.x + measurement.w / 2, // center x
            measurement.y + measurement.h / 2, // center y
            measurement.w,
            measurement.h
        ]);
        
        // Calculate innovation: y = z - H * x
        this.matrixVectorMultiply(this.H, this.state, this.tempMeasure);
        const innovation = new Float32Array(this.measureDim);
        for (let i = 0; i < this.measureDim; i++) {
            innovation[i] = z[i] - this.tempMeasure[i];
        }
        
        // Simplified Kalman gain calculation for real-time performance
        const K = this.calculateSimplifiedKalmanGain(innovation);
        
        // Update state: x = x + K * y
        for (let i = 0; i < this.stateDim; i++) {
            this.state[i] += K[i] * confidence;
        }
        
        // Ensure positive dimensions
        this.state[4] = Math.max(1, this.state[4]);
        this.state[5] = Math.max(1, this.state[5]);
        
        // Update confidence
        const innovationMagnitude = Math.sqrt(innovation[0] * innovation[0] + innovation[1] * innovation[1]);
        this.confidence = Math.max(0.1, Math.min(1.0, 
            this.confidence * 0.95 + 0.05 * Math.exp(-innovationMagnitude / 30)
        ));
    }
    
    calculateSimplifiedKalmanGain(innovation) {
        // Simplified gain calculation for real-time performance
        const K = new Float32Array(this.stateDim);
        
        // Position gains (higher for position, lower for velocity)
        K[0] = innovation[0] * 0.3; // x
        K[1] = innovation[1] * 0.3; // y
        K[2] = innovation[0] * 0.1; // vx
        K[3] = innovation[1] * 0.1; // vy
        K[4] = innovation[2] * 0.2; // w
        K[5] = innovation[3] * 0.2; // h
        
        return K;
    }
    
    predictCovariance() {
        // Simplified covariance prediction
        for (let i = 0; i < this.stateDim; i++) {
            for (let j = 0; j < this.stateDim; j++) {
                if (i === j) {
                    this.P[i * this.stateDim + j] = Math.min(1000, this.P[i * this.stateDim + j] * 1.01 + this.Q[i * this.stateDim + j]);
                }
            }
        }
    }
    
    matrixVectorMultiply(A, x, y) {
        for (let i = 0; i < this.stateDim; i++) {
            let sum = 0;
            for (let j = 0; j < this.stateDim; j++) {
                sum += A[i * this.stateDim + j] * x[j];
            }
            y[i] = sum;
        }
    }
    
    getBbox() {
        return {
            x: this.state[0] - this.state[4] / 2,
            y: this.state[1] - this.state[5] / 2,
            w: this.state[4],
            h: this.state[5]
        };
    }
    
    getVelocity() {
        return {
            vx: this.state[2],
            vy: this.state[3]
        };
    }
    
    getConfidence() {
        return this.confidence;
    }
}

// Export for use in tracker
window.KalmanFilter = KalmanFilter;
