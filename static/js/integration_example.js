/**
 * 轻量级追踪系统集成示例
 * Lightweight Tracking System Integration Example
 * 
 * 展示如何将轻量级追踪器集成到现有的视频流应用中
 */

class LightweightTrackingSystem {
    constructor(videoCanvas, overlayCanvas) {
        this.videoCanvas = videoCanvas;
        this.overlayCanvas = overlayCanvas;
        this.overlayCtx = overlayCanvas.getContext('2d');
        
        // 初始化核心组件
        this.tracker = new LightweightTracker({
            maxTracks: 8,
            searchRadius: 60,
            confidenceThreshold: 0.65,
            maxMissedFrames: 12
        });
        
        this.crowdOptimizer = new CrowdOptimizer({
            densityThreshold: 0.25,
            overlapThreshold: 0.35
        });
        
        this.occlusionHandler = new OcclusionHandler({
            maxOcclusionFrames: 15,
            recoveryThreshold: 0.65
        });
        
        // 性能监控
        this.performanceStats = {
            frameCount: 0,
            avgProcessingTime: 0,
            trackingAccuracy: 0
        };
        
        this.setupEventListeners();
    }

    /**
     * 主处理函数 - 替换原有的复杂追踪逻辑
     */
    async processFrame(detections) {
        const startTime = performance.now();
        
        try {
            // 1. 场景密度分析
            this.crowdOptimizer.analyzeCrowdDensity(
                detections, 
                this.overlayCanvas.width, 
                this.overlayCanvas.height
            );
            
            // 2. 优化检测结果
            const optimized = this.crowdOptimizer.optimizeMatching(
                Array.from(this.tracker.tracks.values()), 
                detections
            );
            
            // 3. 更新空间网格尺寸
            this.tracker.spatialGrid.setDimensions(
                this.overlayCanvas.width, 
                this.overlayCanvas.height
            );
            
            // 4. 执行轻量级追踪
            const tracks = this.tracker.update(optimized.detections, this.videoCanvas);
            
            // 5. 处理遮挡情况
            const occlusionResult = this.occlusionHandler.handleOcclusion(tracks, detections);
            
            // 6. 处理恢复的轨迹
            this.handleRecoveredTracks(occlusionResult.recoveryMatches);
            
            // 7. 绘制结果
            this.drawTrackingResults(tracks, occlusionResult);
            
            // 8. 更新性能统计
            this.updatePerformanceStats(performance.now() - startTime);
            
            return {
                tracks: tracks,
                crowdStatus: this.crowdOptimizer.getOptimizationStatus(),
                occlusionStatus: this.occlusionHandler.getOcclusionStatus(),
                performance: this.performanceStats
            };
            
        } catch (error) {
            console.error('Lightweight tracking error:', error);
            return { tracks: [], error: error.message };
        }
    }

    /**
     * 处理恢复的轨迹
     */
    handleRecoveredTracks(recoveryMatches) {
        for (const match of recoveryMatches) {
            const track = this.tracker.tracks.get(match.trackId);
            if (track) {
                // 更新轨迹位置
                track.x = match.detection.x + match.detection.w / 2;
                track.y = match.detection.y + match.detection.h / 2;
                track.w = match.detection.w;
                track.h = match.detection.h;
                track.missedFrames = 0;
                track.hits++;
                
                // 添加轨迹点
                track.trajectory.push({
                    x: track.x, 
                    y: track.y, 
                    frame: this.tracker.frameCount
                });
                
                console.log(`Track ${match.trackId} recovered from occlusion`);
            }
        }
    }

    /**
     * 绘制追踪结果 - 简洁清晰的可视化
     */
    drawTrackingResults(tracks, occlusionResult) {
        // 清空画布
        this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
        
        // 绘制活跃轨迹
        for (const track of tracks) {
            if (track.locked) {
                this.drawLockedTrack(track);
            } else {
                this.drawActiveTrack(track);
            }
        }
        
        // 绘制遮挡轨迹的预测位置
        for (const occludedInfo of occlusionResult.occlusionStatus.occludedTracks) {
            this.drawOccludedTrack(occludedInfo);
        }
        
        // 绘制场景状态信息
        this.drawSceneInfo();
    }

    /**
     * 绘制锁定的轨迹
     */
    drawLockedTrack(track) {
        const ctx = this.overlayCtx;
        const bbox = {
            x: track.x - track.w/2,
            y: track.y - track.h/2,
            w: track.w,
            h: track.h
        };
        
        // 绘制边界框
        ctx.strokeStyle = track.color;
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 4]);
        ctx.strokeRect(bbox.x, bbox.y, bbox.w, bbox.h);
        
        // 绘制轨迹历史
        if (track.trajectory.length > 1) {
            ctx.strokeStyle = track.color;
            ctx.lineWidth = 2;
            ctx.setLineDash([]);
            ctx.globalAlpha = 0.6;
            
            ctx.beginPath();
            ctx.moveTo(track.trajectory[0].x, track.trajectory[0].y);
            for (let i = 1; i < track.trajectory.length; i++) {
                ctx.lineTo(track.trajectory[i].x, track.trajectory[i].y);
            }
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        }
        
        // 绘制ID标签
        ctx.fillStyle = track.color;
        ctx.fillRect(bbox.x, bbox.y - 25, 60, 20);
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.fillText(`ID ${track.id}`, bbox.x + 5, bbox.y - 10);
    }

    /**
     * 绘制活跃但未锁定的轨迹
     */
    drawActiveTrack(track) {
        const ctx = this.overlayCtx;
        const bbox = {
            x: track.x - track.w/2,
            y: track.y - track.h/2,
            w: track.w,
            h: track.h
        };
        
        // 绘制半透明边界框
        ctx.strokeStyle = track.color;
        ctx.lineWidth = 2;
        ctx.setLineDash([]);
        ctx.globalAlpha = 0.5;
        ctx.strokeRect(bbox.x, bbox.y, bbox.w, bbox.h);
        ctx.globalAlpha = 1.0;
        
        // 绘制中心点
        ctx.fillStyle = track.color;
        ctx.beginPath();
        ctx.arc(track.x, track.y, 3, 0, 2 * Math.PI);
        ctx.fill();
    }

    /**
     * 绘制遮挡轨迹的预测位置
     */
    drawOccludedTrack(occludedInfo) {
        const ctx = this.overlayCtx;
        const pos = occludedInfo.predictedPosition;
        
        // 绘制搜索区域
        ctx.strokeStyle = '#ff6b6b';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.globalAlpha = 0.3;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, occludedInfo.searchRadius, 0, 2 * Math.PI);
        ctx.stroke();
        
        // 绘制预测位置
        ctx.fillStyle = '#ff6b6b';
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 5, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.globalAlpha = 1.0;
        ctx.setLineDash([]);
    }

    /**
     * 绘制场景信息
     */
    drawSceneInfo() {
        const ctx = this.overlayCtx;
        const crowdStatus = this.crowdOptimizer.getOptimizationStatus();
        
        // 绘制信息面板
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(10, 10, 200, 80);
        
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.fillText(`Tracks: ${this.tracker.tracks.size}`, 15, 25);
        ctx.fillText(`Crowd: ${crowdStatus.isCrowdedScene ? 'Dense' : 'Sparse'}`, 15, 40);
        ctx.fillText(`Density: ${(crowdStatus.crowdDensity * 100).toFixed(1)}%`, 15, 55);
        ctx.fillText(`Occluded: ${this.occlusionHandler.occludedTracks.size}`, 15, 70);
        ctx.fillText(`FPS: ${(1000 / this.performanceStats.avgProcessingTime).toFixed(1)}`, 15, 85);
    }

    /**
     * 设置事件监听器
     */
    setupEventListeners() {
        // 点击锁定目标
        this.overlayCanvas.addEventListener('click', (e) => {
            const rect = this.overlayCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const trackId = this.tracker.lockTarget(x, y);
            if (trackId) {
                console.log(`Locked track ${trackId}`);
                this.showNotification(`已锁定目标 ${trackId}`);
            }
        });
        
        // 右键清除所有轨迹
        this.overlayCanvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            this.clearAllTracks();
            this.showNotification('已清除所有轨迹');
        });
    }

    /**
     * 清除所有轨迹
     */
    clearAllTracks() {
        this.tracker.tracks.clear();
        this.occlusionHandler.reset();
        this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
    }

    /**
     * 更新性能统计
     */
    updatePerformanceStats(processingTime) {
        this.performanceStats.frameCount++;
        
        // 计算平均处理时间
        const alpha = 0.1;
        this.performanceStats.avgProcessingTime = 
            this.performanceStats.avgProcessingTime * (1 - alpha) + processingTime * alpha;
    }

    /**
     * 显示通知
     */
    showNotification(message) {
        // 简单的通知实现
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 12px;
            z-index: 1000;
        `;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 2000);
    }

    /**
     * 获取系统状态
     */
    getSystemStatus() {
        return {
            activeTracksCount: this.tracker.tracks.size,
            occludedTracksCount: this.occlusionHandler.occludedTracks.size,
            crowdDensity: this.crowdOptimizer.crowdDensity,
            isCrowdedScene: this.crowdOptimizer.isCrowdedScene,
            averageProcessingTime: this.performanceStats.avgProcessingTime,
            estimatedFPS: 1000 / this.performanceStats.avgProcessingTime
        };
    }
}

// 使用示例
/*
// 初始化轻量级追踪系统
const videoCanvas = document.getElementById('videoCanvas');
const overlayCanvas = document.getElementById('detectionCanvas');
const trackingSystem = new LightweightTrackingSystem(videoCanvas, overlayCanvas);

// 在检测循环中使用
async function detectionLoop() {
    const detections = await window.detectionManager.detectObjects(frameImage);
    const result = await trackingSystem.processFrame(detections);
    
    console.log('Tracking result:', result);
    
    // 更新UI状态
    updateTrackingUI(result);
}
*/

// 导出集成系统
window.LightweightTrackingSystem = LightweightTrackingSystem;
