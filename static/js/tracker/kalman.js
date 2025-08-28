/**
 * Enhanced Kalman Filter for Object Tracking
 * 增强的卡尔曼滤波器用于目标追踪
 * 
 * State vector: [x, y, vx, vy, ax, ay] - position, velocity, acceleration
 * 状态向量: [x, y, vx, vy, ax, ay] - 位置、速度、加速度
 */
class KalmanFilter {
    constructor(initialState = null) {
        // State dimension: 6 (x, y, vx, vy, ax, ay)
        // 状态维度: 6 (x, y, vx, vy, ax, ay)
        this.stateDim = 6;
        this.measureDim = 2; // Only observe position (x, y)
        
        // State vector [x, y, vx, vy, ax, ay]
        // 状态向量 [x, y, vx, vy, ax, ay]
        this.state = initialState || new Float32Array(this.stateDim);
        
        // State covariance matrix (6x6)
        // 状态协方差矩阵 (6x6)
        this.covariance = this.createIdentityMatrix(this.stateDim);
        this.scaleMatrix(this.covariance, 1000); // Initial uncertainty
        
        // Process noise covariance (6x6)
        // 过程噪声协方差 (6x6)
        this.processNoise = this.createProcessNoiseMatrix();
        
        // Measurement noise covariance (2x2)
        // 测量噪声协方差 (2x2)
        this.measurementNoise = this.createMeasurementNoiseMatrix();
        
        // State transition matrix (6x6)
        // 状态转移矩阵 (6x6)
        this.transitionMatrix = this.createTransitionMatrix();
        
        // Measurement matrix (2x6) - only observe position
        // 测量矩阵 (2x6) - 仅观测位置
        this.measurementMatrix = this.createMeasurementMatrix();
        
        // Adaptive parameters for dynamic noise adjustment
        // 自适应参数用于动态噪声调整
        this.adaptiveParams = {
            baseProcessNoise: 0.1,
            basePositionNoise: 5.0,
            velocityFactor: 0.02,
            accelerationFactor: 0.01,
            confidenceFactor: 1.0
        };
        
        // Temporary matrices for calculations
        // 计算用临时矩阵
        this.tempMatrix1 = this.createMatrix(this.stateDim, this.stateDim);
        this.tempMatrix2 = this.createMatrix(this.stateDim, this.measureDim);
        this.tempVector = new Float32Array(this.stateDim);
    }
    
    /**
     * Initialize filter with first detection
     * 使用首次检测初始化滤波器
     */
    initialize(x, y, confidence = 1.0) {
        this.state[0] = x; // x position
        this.state[1] = y; // y position
        this.state[2] = 0; // x velocity
        this.state[3] = 0; // y velocity
        this.state[4] = 0; // x acceleration
        this.state[5] = 0; // y acceleration
        
        // Adjust initial covariance based on confidence
        // 根据置信度调整初始协方差
        const initialUncertainty = 100 / Math.max(0.1, confidence);
        this.covariance = this.createIdentityMatrix(this.stateDim);
        this.scaleMatrix(this.covariance, initialUncertainty);
        
        this.adaptiveParams.confidenceFactor = confidence;
    }
    
    /**
     * Predict next state using motion model
     * 使用运动模型预测下一状态
     */
    predict(dt = 1/30) {
        // Update transition matrix with current time step
        // 使用当前时间步长更新转移矩阵
        this.updateTransitionMatrix(dt);
        
        // Predict state: x_k = F * x_{k-1}
        // 预测状态: x_k = F * x_{k-1}
        this.matrixVectorMultiply(this.transitionMatrix, this.state, this.tempVector);
        this.state.set(this.tempVector);
        
        // Predict covariance: P_k = F * P_{k-1} * F^T + Q
        // 预测协方差: P_k = F * P_{k-1} * F^T + Q
        this.updateProcessNoise(dt);
        this.predictCovariance();
        
        return {
            x: this.state[0],
            y: this.state[1],
            vx: this.state[2],
            vy: this.state[3],
            ax: this.state[4],
            ay: this.state[5]
        };
    }
    
    /**
     * Update filter with new measurement
     * 使用新测量值更新滤波器
     */
    update(x, y, confidence = 1.0) {
        // Measurement vector
        // 测量向量
        const measurement = new Float32Array([x, y]);
        
        // Adapt measurement noise based on confidence
        // 根据置信度调整测量噪声
        this.adaptMeasurementNoise(confidence);
        
        // Innovation: y = z - H * x_predicted
        // 新息: y = z - H * x_predicted
        const innovation = this.calculateInnovation(measurement);
        
        // Innovation covariance: S = H * P * H^T + R
        // 新息协方差: S = H * P * H^T + R
        const innovationCovariance = this.calculateInnovationCovariance();
        
        // Kalman gain: K = P * H^T * S^(-1)
        // 卡尔曼增益: K = P * H^T * S^(-1)
        const kalmanGain = this.calculateKalmanGain(innovationCovariance);
        
        // Update state: x = x + K * innovation
        // 更新状态: x = x + K * innovation
        this.updateState(kalmanGain, innovation);
        
        // Update covariance: P = (I - K * H) * P
        // 更新协方差: P = (I - K * H) * P
        this.updateCovariance(kalmanGain);
        
        // Store confidence for adaptive parameters
        // 存储置信度用于自适应参数
        this.adaptiveParams.confidenceFactor = confidence;
        
        return {
            x: this.state[0],
            y: this.state[1],
            vx: this.state[2],
            vy: this.state[3],
            ax: this.state[4],
            ay: this.state[5],
            uncertainty: this.getPositionUncertainty()
        };
    }
    
    /**
     * Get current state and uncertainty
     * 获取当前状态和不确定性
     */
    getState() {
        return {
            position: { x: this.state[0], y: this.state[1] },
            velocity: { x: this.state[2], y: this.state[3] },
            acceleration: { x: this.state[4], y: this.state[5] },
            uncertainty: this.getPositionUncertainty(),
            velocityMagnitude: Math.sqrt(this.state[2]**2 + this.state[3]**2)
        };
    }
    
    /**
     * Get position uncertainty (standard deviation)
     * 获取位置不确定性（标准差）
     */
    getPositionUncertainty() {
        return Math.sqrt(this.covariance[0] + this.covariance[7]); // sqrt(var_x + var_y)
    }
    
    /**
     * Create state transition matrix F
     * 创建状态转移矩阵 F
     */
    createTransitionMatrix() {
        const F = this.createIdentityMatrix(this.stateDim);
        // Will be updated in updateTransitionMatrix with actual dt
        return F;
    }
    
    /**
     * Update transition matrix with time step
     * 使用时间步长更新转移矩阵
     */
    updateTransitionMatrix(dt) {
        // Reset to identity
        this.setIdentityMatrix(this.transitionMatrix);
        
        // Position = position + velocity*dt + 0.5*acceleration*dt^2
        this.transitionMatrix[0 * this.stateDim + 2] = dt;     // x += vx*dt
        this.transitionMatrix[1 * this.stateDim + 3] = dt;     // y += vy*dt
        this.transitionMatrix[0 * this.stateDim + 4] = 0.5*dt*dt; // x += 0.5*ax*dt^2
        this.transitionMatrix[1 * this.stateDim + 5] = 0.5*dt*dt; // y += 0.5*ay*dt^2
        
        // Velocity = velocity + acceleration*dt
        this.transitionMatrix[2 * this.stateDim + 4] = dt;     // vx += ax*dt
        this.transitionMatrix[3 * this.stateDim + 5] = dt;     // vy += ay*dt
    }
    
    /**
     * Create process noise matrix Q
     * 创建过程噪声矩阵 Q
     */
    createProcessNoiseMatrix() {
        const Q = this.createMatrix(this.stateDim, this.stateDim);
        // Will be updated dynamically
        return Q;
    }
    
    /**
     * Update process noise based on current state
     * 根据当前状态更新过程噪声
     */
    updateProcessNoise(dt) {
        // Clear matrix
        this.processNoise.fill(0);
        
        // Adaptive noise based on velocity and acceleration
        // 基于速度和加速度的自适应噪声
        const velocityMag = Math.sqrt(this.state[2]**2 + this.state[3]**2);
        const accelerationMag = Math.sqrt(this.state[4]**2 + this.state[5]**2);
        
        const posNoise = this.adaptiveParams.basePositionNoise * 
                        (1 + velocityMag * this.adaptiveParams.velocityFactor);
        const velNoise = this.adaptiveParams.baseProcessNoise * 
                        (1 + accelerationMag * this.adaptiveParams.accelerationFactor);
        const accNoise = this.adaptiveParams.baseProcessNoise * 0.5;
        
        // Position noise
        this.processNoise[0 * this.stateDim + 0] = posNoise * dt * dt;
        this.processNoise[1 * this.stateDim + 1] = posNoise * dt * dt;
        
        // Velocity noise
        this.processNoise[2 * this.stateDim + 2] = velNoise * dt;
        this.processNoise[3 * this.stateDim + 3] = velNoise * dt;
        
        // Acceleration noise
        this.processNoise[4 * this.stateDim + 4] = accNoise;
        this.processNoise[5 * this.stateDim + 5] = accNoise;
    }
    
    /**
     * Create measurement noise matrix R
     * 创建测量噪声矩阵 R
     */
    createMeasurementNoiseMatrix() {
        const R = this.createMatrix(this.measureDim, this.measureDim);
        R[0] = 25.0; // x measurement noise variance
        R[3] = 25.0; // y measurement noise variance
        return R;
    }
    
    /**
     * Adapt measurement noise based on detection confidence
     * 根据检测置信度调整测量噪声
     */
    adaptMeasurementNoise(confidence) {
        const baseNoise = 25.0;
        const adaptedNoise = baseNoise / Math.max(0.1, confidence);
        
        this.measurementNoise[0] = adaptedNoise;
        this.measurementNoise[3] = adaptedNoise;
    }
    
    /**
     * Create measurement matrix H (2x6)
     * 创建测量矩阵 H (2x6)
     */
    createMeasurementMatrix() {
        const H = this.createMatrix(this.measureDim, this.stateDim);
        H[0 * this.stateDim + 0] = 1; // Observe x position
        H[1 * this.stateDim + 1] = 1; // Observe y position
        return H;
    }
    
    /**
     * Calculate innovation (measurement residual)
     * 计算新息（测量残差）
     */
    calculateInnovation(measurement) {
        const predicted = new Float32Array(this.measureDim);
        
        // H * x_predicted
        for (let i = 0; i < this.measureDim; i++) {
            predicted[i] = 0;
            for (let j = 0; j < this.stateDim; j++) {
                predicted[i] += this.measurementMatrix[i * this.stateDim + j] * this.state[j];
            }
        }
        
        // innovation = measurement - predicted
        const innovation = new Float32Array(this.measureDim);
        for (let i = 0; i < this.measureDim; i++) {
            innovation[i] = measurement[i] - predicted[i];
        }
        
        return innovation;
    }
    
    /**
     * Calculate innovation covariance S = H * P * H^T + R
     * 计算新息协方差 S = H * P * H^T + R
     */
    calculateInnovationCovariance() {
        // H * P
        const HP = this.createMatrix(this.measureDim, this.stateDim);
        this.matrixMultiply(this.measurementMatrix, this.measureDim, this.stateDim,
                           this.covariance, this.stateDim, this.stateDim, HP);
        
        // H * P * H^T
        const HPHt = this.createMatrix(this.measureDim, this.measureDim);
        this.matrixMultiplyTranspose(HP, this.measureDim, this.stateDim,
                                   this.measurementMatrix, this.measureDim, this.stateDim, HPHt);
        
        // S = H * P * H^T + R
        const S = this.createMatrix(this.measureDim, this.measureDim);
        for (let i = 0; i < this.measureDim * this.measureDim; i++) {
            S[i] = HPHt[i] + this.measurementNoise[i];
        }
        
        return S;
    }
    
    /**
     * Calculate Kalman gain K = P * H^T * S^(-1)
     * 计算卡尔曼增益 K = P * H^T * S^(-1)
     */
    calculateKalmanGain(innovationCovariance) {
        // P * H^T
        const PHt = this.createMatrix(this.stateDim, this.measureDim);
        this.matrixMultiplyTranspose(this.covariance, this.stateDim, this.stateDim,
                                   this.measurementMatrix, this.measureDim, this.stateDim, PHt);
        
        // Invert S (2x2 matrix)
        const Sinv = this.invertMatrix2x2(innovationCovariance);
        
        // K = P * H^T * S^(-1)
        const K = this.createMatrix(this.stateDim, this.measureDim);
        this.matrixMultiply(PHt, this.stateDim, this.measureDim,
                           Sinv, this.measureDim, this.measureDim, K);
        
        return K;
    }
    
    /**
     * Update state with Kalman gain and innovation
     * 使用卡尔曼增益和新息更新状态
     */
    updateState(kalmanGain, innovation) {
        for (let i = 0; i < this.stateDim; i++) {
            let correction = 0;
            for (let j = 0; j < this.measureDim; j++) {
                correction += kalmanGain[i * this.measureDim + j] * innovation[j];
            }
            this.state[i] += correction;
        }
    }
    
    /**
     * Update covariance P = (I - K * H) * P
     * 更新协方差 P = (I - K * H) * P
     */
    updateCovariance(kalmanGain) {
        // K * H
        const KH = this.createMatrix(this.stateDim, this.stateDim);
        this.matrixMultiply(kalmanGain, this.stateDim, this.measureDim,
                           this.measurementMatrix, this.measureDim, this.stateDim, KH);
        
        // I - K * H
        const IKH = this.createIdentityMatrix(this.stateDim);
        for (let i = 0; i < this.stateDim * this.stateDim; i++) {
            IKH[i] -= KH[i];
        }
        
        // (I - K * H) * P
        const newCovariance = this.createMatrix(this.stateDim, this.stateDim);
        this.matrixMultiply(IKH, this.stateDim, this.stateDim,
                           this.covariance, this.stateDim, this.stateDim, newCovariance);
        
        this.covariance.set(newCovariance);
    }
    
    /**
     * Predict covariance P = F * P * F^T + Q
     * 预测协方差 P = F * P * F^T + Q
     */
    predictCovariance() {
        // F * P
        const FP = this.createMatrix(this.stateDim, this.stateDim);
        this.matrixMultiply(this.transitionMatrix, this.stateDim, this.stateDim,
                           this.covariance, this.stateDim, this.stateDim, FP);
        
        // F * P * F^T
        const FPFt = this.createMatrix(this.stateDim, this.stateDim);
        this.matrixMultiplyTranspose(FP, this.stateDim, this.stateDim,
                                   this.transitionMatrix, this.stateDim, this.stateDim, FPFt);
        
        // P = F * P * F^T + Q
        for (let i = 0; i < this.stateDim * this.stateDim; i++) {
            this.covariance[i] = FPFt[i] + this.processNoise[i];
        }
    }
    
    // ======================== Matrix Utilities ========================
    
    createMatrix(rows, cols) {
        return new Float32Array(rows * cols);
    }
    
    createIdentityMatrix(size) {
        const matrix = new Float32Array(size * size);
        for (let i = 0; i < size; i++) {
            matrix[i * size + i] = 1;
        }
        return matrix;
    }
    
    setIdentityMatrix(matrix) {
        const size = Math.sqrt(matrix.length);
        matrix.fill(0);
        for (let i = 0; i < size; i++) {
            matrix[i * size + i] = 1;
        }
    }
    
    scaleMatrix(matrix, scale) {
        for (let i = 0; i < matrix.length; i++) {
            matrix[i] *= scale;
        }
    }
    
    matrixVectorMultiply(matrix, vector, result) {
        const size = vector.length;
        for (let i = 0; i < size; i++) {
            result[i] = 0;
            for (let j = 0; j < size; j++) {
                result[i] += matrix[i * size + j] * vector[j];
            }
        }
    }
    
    matrixMultiply(A, rowsA, colsA, B, rowsB, colsB, result) {
        for (let i = 0; i < rowsA; i++) {
            for (let j = 0; j < colsB; j++) {
                result[i * colsB + j] = 0;
                for (let k = 0; k < colsA; k++) {
                    result[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
                }
            }
        }
    }
    
    matrixMultiplyTranspose(A, rowsA, colsA, B, rowsB, colsB, result) {
        // result = A * B^T
        for (let i = 0; i < rowsA; i++) {
            for (let j = 0; j < rowsB; j++) {
                result[i * rowsB + j] = 0;
                for (let k = 0; k < colsA; k++) {
                    result[i * rowsB + j] += A[i * colsA + k] * B[j * colsB + k];
                }
            }
        }
    }
    
    invertMatrix2x2(matrix) {
        const a = matrix[0], b = matrix[1];
        const c = matrix[2], d = matrix[3];
        
        const det = a * d - b * c;
        if (Math.abs(det) < 1e-10) {
            // Singular matrix, return identity
            return new Float32Array([1, 0, 0, 1]);
        }
        
        const invDet = 1.0 / det;
        return new Float32Array([
            d * invDet, -b * invDet,
            -c * invDet, a * invDet
        ]);
    }
}

// Export for use in other modules
// 导出供其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = KalmanFilter;
} else if (typeof window !== 'undefined') {
    window.KalmanFilter = KalmanFilter;
}
