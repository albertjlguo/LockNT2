/**
 * Enhanced Appearance Feature Extractor
 * 增强的外观特征提取器
 * 
 * Multi-level feature extraction for robust person/object discrimination
 * 多层次特征提取，用于鲁棒的人物/目标区分
 */
class EnhancedAppearanceExtractor {
    constructor(opts = {}) {
        // Color histogram configuration
        // 颜色直方图配置
        this.colorConfig = {
            hBins: opts.hBins || 24,     // Hue bins (increased for better discrimination)
            sBins: opts.sBins || 8,      // Saturation bins
            vBins: opts.vBins || 6,      // Value bins
            totalColorBins: 0
        };
        this.colorConfig.totalColorBins = this.colorConfig.hBins * this.colorConfig.sBins * this.colorConfig.vBins;
        
        // Spatial grid configuration for layout-aware features
        // 空间网格配置，用于布局感知特征
        this.spatialConfig = {
            gridRows: opts.gridRows || 4,    // Vertical divisions
            gridCols: opts.gridCols || 3,    // Horizontal divisions
            totalCells: 0
        };
        this.spatialConfig.totalCells = this.spatialConfig.gridRows * this.spatialConfig.gridCols;
        
        // Edge/texture configuration
        // 边缘/纹理配置
        this.textureConfig = {
            enableTexture: opts.enableTexture !== false,
            sobelThreshold: opts.sobelThreshold || 30,
            texturePatterns: 8  // Number of local binary pattern variations
        };
        
        // Feature dimensions
        // 特征维度
        this.dimensions = {
            color: this.colorConfig.totalColorBins,
            spatial: this.spatialConfig.totalCells * 4,  // RGB + brightness per cell
            texture: this.textureConfig.enableTexture ? this.textureConfig.texturePatterns : 0,
            total: 0
        };
        this.dimensions.total = this.dimensions.color + this.dimensions.spatial + this.dimensions.texture;
        
        // Sampling configuration for performance
        // 采样配置，用于性能优化
        this.sampling = {
            step: opts.samplingStep || 2,
            minSize: opts.minSize || 16,
            cropRatio: opts.cropRatio || 0.9  // Use 90% of bbox to avoid edge noise
        };
        
        // Temporary arrays for calculations
        // 计算用临时数组
        this.tempArrays = {
            colorHist: new Float32Array(this.dimensions.color),
            spatialFeats: new Float32Array(this.dimensions.spatial),
            textureFeats: new Float32Array(this.dimensions.texture),
            combined: new Float32Array(this.dimensions.total)
        };
        
        console.log(`Enhanced appearance extractor initialized: ${this.dimensions.total}D features`);
    }
    
    /**
     * Extract comprehensive appearance features from bounding box
     * 从边界框提取综合外观特征
     */
    extractFeatures(ctx, bbox, objectClass = 'unknown') {
        if (!ctx || !bbox || bbox.w < this.sampling.minSize || bbox.h < this.sampling.minSize) {
            return null;
        }
        
        // Calculate cropped region to avoid edge artifacts
        // 计算裁剪区域以避免边缘伪影
        const cropMargin = (1 - this.sampling.cropRatio) / 2;
        const cropX = bbox.x + bbox.w * cropMargin;
        const cropY = bbox.y + bbox.h * cropMargin;
        const cropW = bbox.w * this.sampling.cropRatio;
        const cropH = bbox.h * this.sampling.cropRatio;
        
        // Validate cropped region
        // 验证裁剪区域
        const finalX = Math.max(0, Math.round(cropX));
        const finalY = Math.max(0, Math.round(cropY));
        const finalW = Math.max(this.sampling.minSize, Math.round(cropW));
        const finalH = Math.max(this.sampling.minSize, Math.round(cropH));
        
        let imageData;
        try {
            imageData = ctx.getImageData(finalX, finalY, finalW, finalH);
        } catch (e) {
            console.warn('Failed to extract image data:', e);
            return null;
        }
        
        // Extract multi-level features
        // 提取多层次特征
        const colorFeatures = this.extractColorHistogram(imageData);
        const spatialFeatures = this.extractSpatialFeatures(imageData);
        const textureFeatures = this.textureConfig.enableTexture ? 
            this.extractTextureFeatures(imageData) : new Float32Array(0);
        
        // Combine all features
        // 组合所有特征
        const combinedFeatures = this.combineFeatures(colorFeatures, spatialFeatures, textureFeatures);
        
        // Apply class-specific normalization
        // 应用特定类别的归一化
        this.applyClassSpecificNormalization(combinedFeatures, objectClass);
        
        // L2 normalize final feature vector
        // L2归一化最终特征向量
        this.l2Normalize(combinedFeatures);
        
        return combinedFeatures;
    }
    
    /**
     * Extract enhanced color histogram with improved discrimination
     * 提取增强的颜色直方图，改进区分能力
     */
    extractColorHistogram(imageData) {
        const hist = this.tempArrays.colorHist;
        hist.fill(0);
        
        const data = imageData.data;
        const width = imageData.width;
        const height = imageData.height;
        const step = this.sampling.step;
        
        let totalPixels = 0;
        
        for (let y = 0; y < height; y += step) {
            for (let x = 0; x < width; x += step) {
                const idx = (y * width + x) * 4;
                const r = data[idx];
                const g = data[idx + 1];
                const b = data[idx + 2];
                const alpha = data[idx + 3];
                
                // Skip transparent pixels
                // 跳过透明像素
                if (alpha < 128) continue;
                
                const hsv = this.rgbToHsv(r, g, b);
                
                // Enhanced binning with smooth transitions
                // 增强的分箱，具有平滑过渡
                const hBin = this.getSmoothBin(hsv.h / 360, this.colorConfig.hBins);
                const sBin = this.getSmoothBin(hsv.s, this.colorConfig.sBins);
                const vBin = this.getSmoothBin(hsv.v, this.colorConfig.vBins);
                
                // Distribute weight across neighboring bins for smoothness
                // 在相邻分箱间分配权重以实现平滑性
                this.addSmoothWeight(hist, hBin, sBin, vBin);
                totalPixels++;
            }
        }
        
        // Normalize by total pixels
        // 按总像素数归一化
        if (totalPixels > 0) {
            for (let i = 0; i < hist.length; i++) {
                hist[i] /= totalPixels;
            }
        }
        
        return new Float32Array(hist);
    }
    
    /**
     * Extract spatial layout features for position-aware discrimination
     * 提取空间布局特征，用于位置感知区分
     */
    extractSpatialFeatures(imageData) {
        const spatialFeats = this.tempArrays.spatialFeats;
        spatialFeats.fill(0);
        
        const data = imageData.data;
        const width = imageData.width;
        const height = imageData.height;
        const step = this.sampling.step;
        
        const cellWidth = width / this.spatialConfig.gridCols;
        const cellHeight = height / this.spatialConfig.gridRows;
        
        // Extract features for each spatial cell
        // 为每个空间单元提取特征
        for (let row = 0; row < this.spatialConfig.gridRows; row++) {
            for (let col = 0; col < this.spatialConfig.gridCols; col++) {
                const cellIdx = row * this.spatialConfig.gridCols + col;
                const featureOffset = cellIdx * 4; // 4 features per cell: R, G, B, brightness
                
                const startX = Math.floor(col * cellWidth);
                const endX = Math.min(width, Math.floor((col + 1) * cellWidth));
                const startY = Math.floor(row * cellHeight);
                const endY = Math.min(height, Math.floor((row + 1) * cellHeight));
                
                let sumR = 0, sumG = 0, sumB = 0, sumBrightness = 0;
                let pixelCount = 0;
                
                for (let y = startY; y < endY; y += step) {
                    for (let x = startX; x < endX; x += step) {
                        const idx = (y * width + x) * 4;
                        const r = data[idx];
                        const g = data[idx + 1];
                        const b = data[idx + 2];
                        const alpha = data[idx + 3];
                        
                        if (alpha < 128) continue;
                        
                        sumR += r;
                        sumG += g;
                        sumB += b;
                        sumBrightness += (r + g + b) / 3;
                        pixelCount++;
                    }
                }
                
                if (pixelCount > 0) {
                    spatialFeats[featureOffset] = sumR / (pixelCount * 255);
                    spatialFeats[featureOffset + 1] = sumG / (pixelCount * 255);
                    spatialFeats[featureOffset + 2] = sumB / (pixelCount * 255);
                    spatialFeats[featureOffset + 3] = sumBrightness / (pixelCount * 255);
                }
            }
        }
        
        return new Float32Array(spatialFeats);
    }
    
    /**
     * Extract texture features using simplified local binary patterns
     * 使用简化的局部二值模式提取纹理特征
     */
    extractTextureFeatures(imageData) {
        if (!this.textureConfig.enableTexture) {
            return new Float32Array(0);
        }
        
        const textureFeats = this.tempArrays.textureFeats;
        textureFeats.fill(0);
        
        const data = imageData.data;
        const width = imageData.width;
        const height = imageData.height;
        
        // Convert to grayscale for texture analysis
        // 转换为灰度图进行纹理分析
        const grayData = new Uint8Array(width * height);
        for (let i = 0; i < width * height; i++) {
            const pixelIdx = i * 4;
            grayData[i] = Math.round(0.299 * data[pixelIdx] + 0.587 * data[pixelIdx + 1] + 0.114 * data[pixelIdx + 2]);
        }
        
        // Simplified LBP (Local Binary Pattern) calculation
        // 简化的LBP（局部二值模式）计算
        let totalPatterns = 0;
        
        for (let y = 1; y < height - 1; y += 2) {
            for (let x = 1; x < width - 1; x += 2) {
                const centerIdx = y * width + x;
                const centerValue = grayData[centerIdx];
                
                // 8-neighbor LBP pattern
                // 8邻域LBP模式
                let pattern = 0;
                const neighbors = [
                    grayData[(y-1) * width + (x-1)], // top-left
                    grayData[(y-1) * width + x],     // top
                    grayData[(y-1) * width + (x+1)], // top-right
                    grayData[y * width + (x+1)],     // right
                    grayData[(y+1) * width + (x+1)], // bottom-right
                    grayData[(y+1) * width + x],     // bottom
                    grayData[(y+1) * width + (x-1)], // bottom-left
                    grayData[y * width + (x-1)]      // left
                ];
                
                for (let i = 0; i < 8; i++) {
                    if (neighbors[i] >= centerValue) {
                        pattern |= (1 << i);
                    }
                }
                
                // Simplified pattern classification (uniform patterns only)
                // 简化的模式分类（仅统一模式）
                const uniformPattern = this.getUniformPattern(pattern);
                if (uniformPattern < this.textureConfig.texturePatterns) {
                    textureFeats[uniformPattern]++;
                    totalPatterns++;
                }
            }
        }
        
        // Normalize texture features
        // 归一化纹理特征
        if (totalPatterns > 0) {
            for (let i = 0; i < textureFeats.length; i++) {
                textureFeats[i] /= totalPatterns;
            }
        }
        
        return new Float32Array(textureFeats);
    }
    
    /**
     * Combine all feature types into single vector
     * 将所有特征类型组合为单一向量
     */
    combineFeatures(colorFeatures, spatialFeatures, textureFeatures) {
        const combined = this.tempArrays.combined;
        let offset = 0;
        
        // Copy color features
        // 复制颜色特征
        combined.set(colorFeatures, offset);
        offset += colorFeatures.length;
        
        // Copy spatial features
        // 复制空间特征
        combined.set(spatialFeatures, offset);
        offset += spatialFeatures.length;
        
        // Copy texture features
        // 复制纹理特征
        if (textureFeatures.length > 0) {
            combined.set(textureFeatures, offset);
        }
        
        return new Float32Array(combined);
    }
    
    /**
     * Apply class-specific feature weighting
     * 应用特定类别的特征权重
     */
    applyClassSpecificNormalization(features, objectClass) {
        // Different classes may benefit from different feature emphasis
        // 不同类别可能受益于不同的特征强调
        const classWeights = {
            person: { color: 1.0, spatial: 1.2, texture: 0.8 },
            car: { color: 0.8, spatial: 1.0, texture: 1.2 },
            default: { color: 1.0, spatial: 1.0, texture: 1.0 }
        };
        
        const weights = classWeights[objectClass] || classWeights.default;
        
        let offset = 0;
        
        // Weight color features
        // 权重颜色特征
        for (let i = 0; i < this.dimensions.color; i++) {
            features[offset + i] *= weights.color;
        }
        offset += this.dimensions.color;
        
        // Weight spatial features
        // 权重空间特征
        for (let i = 0; i < this.dimensions.spatial; i++) {
            features[offset + i] *= weights.spatial;
        }
        offset += this.dimensions.spatial;
        
        // Weight texture features
        // 权重纹理特征
        for (let i = 0; i < this.dimensions.texture; i++) {
            features[offset + i] *= weights.texture;
        }
    }
    
    /**
     * Calculate cosine similarity between two feature vectors
     * 计算两个特征向量之间的余弦相似度
     */
    calculateSimilarity(features1, features2) {
        if (!features1 || !features2 || features1.length !== features2.length) {
            return 0;
        }
        
        let dot = 0, norm1 = 0, norm2 = 0;
        
        for (let i = 0; i < features1.length; i++) {
            dot += features1[i] * features2[i];
            norm1 += features1[i] * features1[i];
            norm2 += features2[i] * features2[i];
        }
        
        const denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
        return denominator > 1e-10 ? dot / denominator : 0;
    }
    
    /**
     * Calculate weighted similarity with feature importance
     * 计算带特征重要性的加权相似度
     */
    calculateWeightedSimilarity(features1, features2, objectClass = 'unknown') {
        const baseSimilarity = this.calculateSimilarity(features1, features2);
        
        // Calculate feature-specific similarities for better discrimination
        // 计算特征特定相似度以改进区分能力
        const colorSim = this.calculateSimilarity(
            features1.slice(0, this.dimensions.color),
            features2.slice(0, this.dimensions.color)
        );
        
        const spatialSim = this.calculateSimilarity(
            features1.slice(this.dimensions.color, this.dimensions.color + this.dimensions.spatial),
            features2.slice(this.dimensions.color, this.dimensions.color + this.dimensions.spatial)
        );
        
        let textureSim = 0;
        if (this.dimensions.texture > 0) {
            textureSim = this.calculateSimilarity(
                features1.slice(this.dimensions.color + this.dimensions.spatial),
                features2.slice(this.dimensions.color + this.dimensions.spatial)
            );
        }
        
        // Class-specific weighting
        // 特定类别权重
        const weights = objectClass === 'person' ? 
            { color: 0.4, spatial: 0.4, texture: 0.2 } :
            { color: 0.3, spatial: 0.3, texture: 0.4 };
        
        const weightedSim = weights.color * colorSim + 
                           weights.spatial * spatialSim + 
                           weights.texture * textureSim;
        
        // Combine with base similarity
        // 与基础相似度结合
        return 0.7 * baseSimilarity + 0.3 * weightedSim;
    }
    
    // ======================== Utility Functions ========================
    
    rgbToHsv(r, g, b) {
        r /= 255; g /= 255; b /= 255;
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const diff = max - min;
        
        let h = 0;
        if (diff !== 0) {
            switch (max) {
                case r: h = ((g - b) / diff + (g < b ? 6 : 0)) * 60; break;
                case g: h = ((b - r) / diff + 2) * 60; break;
                case b: h = ((r - g) / diff + 4) * 60; break;
            }
        }
        
        const s = max === 0 ? 0 : diff / max;
        const v = max;
        
        return { h, s, v };
    }
    
    getSmoothBin(value, numBins) {
        const binSize = 1.0 / numBins;
        const binIndex = Math.floor(value / binSize);
        return Math.min(binIndex, numBins - 1);
    }
    
    addSmoothWeight(histogram, hBin, sBin, vBin) {
        const idx = hBin * (this.colorConfig.sBins * this.colorConfig.vBins) + 
                   sBin * this.colorConfig.vBins + vBin;
        if (idx >= 0 && idx < histogram.length) {
            histogram[idx] += 1.0;
        }
    }
    
    getUniformPattern(pattern) {
        // Count bit transitions for uniform pattern detection
        // 计算位转换以检测统一模式
        let transitions = 0;
        for (let i = 0; i < 8; i++) {
            const bit1 = (pattern >> i) & 1;
            const bit2 = (pattern >> ((i + 1) % 8)) & 1;
            if (bit1 !== bit2) transitions++;
        }
        
        // Uniform patterns have at most 2 transitions
        // 统一模式最多有2个转换
        if (transitions <= 2) {
            // Count number of 1s in pattern
            // 计算模式中1的数量
            let ones = 0;
            for (let i = 0; i < 8; i++) {
                if ((pattern >> i) & 1) ones++;
            }
            return Math.min(ones, this.textureConfig.texturePatterns - 1);
        }
        
        return this.textureConfig.texturePatterns - 1; // Non-uniform pattern
    }
    
    l2Normalize(vector) {
        let norm = 0;
        for (let i = 0; i < vector.length; i++) {
            norm += vector[i] * vector[i];
        }
        norm = Math.sqrt(norm);
        
        if (norm > 1e-10) {
            for (let i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
    }
}

// Export for use in other modules
// 导出供其他模块使用
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EnhancedAppearanceExtractor;
} else if (typeof window !== 'undefined') {
    window.EnhancedAppearanceExtractor = EnhancedAppearanceExtractor;
}
