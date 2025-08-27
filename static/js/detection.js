/**
 * Object Detection Manager
 * Handles TensorFlow.js model loading and object detection
 */
class ObjectDetectionManager {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.detectionCallbacks = [];
        this.detectionStats = {
            totalDetections: 0,
            objectCounts: {},
            recentDetections: []
        };
    }

    /**
     * Initialize and load the COCO-SSD model
     */
    async loadModel() {
        try {
            this.updateModelStatus('loading', 'Loading AI Model...');
            
            // Load the COCO-SSD model
            this.model = await cocoSsd.load();
            this.isModelLoaded = true;
            
            this.updateModelStatus('ready', 'AI Model Ready');
            console.log('COCO-SSD model loaded successfully');
            
            return true;
        } catch (error) {
            console.error('Error loading model:', error);
            this.updateModelStatus('error', 'Failed to load AI Model');
            return false;
        }
    }

    /**
     * Update model status in the UI
     */
    updateModelStatus(status, message) {
        const statusIndicator = document.getElementById('modelStatus');
        const statusText = document.getElementById('modelStatusText');
        const progressBar = document.querySelector('#modelProgress .progress-bar');

        if (statusIndicator && statusText) {
            statusText.textContent = message;
            
            // Update status indicator
            statusIndicator.innerHTML = '';
            switch (status) {
                case 'loading':
                    statusIndicator.innerHTML = '<i class="fas fa-circle text-warning pulse"></i>';
                    if (progressBar) {
                        progressBar.style.width = '50%';
                    }
                    break;
                case 'ready':
                    statusIndicator.innerHTML = '<i class="fas fa-circle text-success"></i>';
                    if (progressBar) {
                        progressBar.style.width = '100%';
                        setTimeout(() => {
                            document.getElementById('modelProgress').style.display = 'none';
                        }, 1000);
                    }
                    break;
                case 'error':
                    statusIndicator.innerHTML = '<i class="fas fa-circle text-danger"></i>';
                    if (progressBar) {
                        progressBar.style.width = '0%';
                    }
                    break;
            }
        }
    }

    /**
     * Detect objects in a video frame
     */
    async detectObjects(videoElement) {
        if (!this.isModelLoaded || !this.model) {
            console.warn('Model not loaded yet');
            return [];
        }

        try {
            const predictions = await this.model.detect(videoElement);
            this.processDetections(predictions);
            return predictions;
        } catch (error) {
            console.error('Error during object detection:', error);
            return [];
        }
    }

    /**
     * Process detection results and update statistics
     */
    processDetections(predictions) {
        // Update total detection count
        this.detectionStats.totalDetections += predictions.length;
        
        // Reset current frame object counts
        const currentFrameCounts = {};
        
        // Process each prediction
        predictions.forEach(prediction => {
            const className = prediction.class;
            const confidence = (prediction.score * 100).toFixed(1);
            
            // Update object counts
            currentFrameCounts[className] = (currentFrameCounts[className] || 0) + 1;
            
            // Add to recent detections
            this.detectionStats.recentDetections.unshift({
                class: className,
                confidence: confidence,
                timestamp: new Date()
            });
            
            // Keep only last 50 detections
            if (this.detectionStats.recentDetections.length > 50) {
                this.detectionStats.recentDetections.pop();
            }
        });
        
        // Update object counts (using current frame counts)
        this.detectionStats.objectCounts = currentFrameCounts;
        
        // Update UI
        this.updateDetectionUI();
        
        // Notify callbacks
        this.detectionCallbacks.forEach(callback => {
            callback(predictions, this.detectionStats);
        });
    }

    /**
     * Update the detection results UI
     */
    updateDetectionUI() {
        // Update object counts
        this.updateObjectCounts();
        
        // Update detection history
        this.updateDetectionHistory();
        
        // Update total count
        const detectionCountElement = document.getElementById('detectionCount');
        if (detectionCountElement) {
            const totalCurrentObjects = Object.values(this.detectionStats.objectCounts)
                .reduce((sum, count) => sum + count, 0);
            detectionCountElement.textContent = totalCurrentObjects;
        }
    }

    /**
     * Update object counts display
     */
    updateObjectCounts() {
        const objectCountsContainer = document.getElementById('objectCounts');
        if (!objectCountsContainer) return;

        const counts = this.detectionStats.objectCounts;
        
        if (Object.keys(counts).length === 0) {
            objectCountsContainer.innerHTML = `
                <div class="no-detections text-center py-4">
                    <i class="fas fa-search fa-2x text-muted mb-2"></i>
                    <p class="text-muted mb-0">No objects detected</p>
                    <small class="text-muted">Objects will appear here when detected</small>
                </div>
            `;
            return;
        }

        // Sort objects by count (descending)
        const sortedObjects = Object.entries(counts)
            .sort(([,a], [,b]) => b - a);

        const objectsHTML = sortedObjects.map(([className, count]) => {
            // Get average confidence for this object class
            const recentForClass = this.detectionStats.recentDetections
                .filter(d => d.class === className)
                .slice(0, 5); // Last 5 detections
            
            const avgConfidence = recentForClass.length > 0 
                ? (recentForClass.reduce((sum, d) => sum + parseFloat(d.confidence), 0) / recentForClass.length).toFixed(1)
                : '0.0';

            return `
                <div class="object-item fade-in">
                    <div>
                        <div class="object-name">${this.formatClassName(className)}</div>
                        <div class="confidence-score">${avgConfidence}% confidence</div>
                    </div>
                    <div class="object-count">${count}</div>
                </div>
            `;
        }).join('');

        objectCountsContainer.innerHTML = objectsHTML;
    }

    /**
     * Update detection history display
     */
    updateDetectionHistory() {
        const historyContainer = document.getElementById('detectionHistory');
        if (!historyContainer) return;

        const recentDetections = this.detectionStats.recentDetections.slice(0, 10);
        
        if (recentDetections.length === 0) {
            historyContainer.innerHTML = `
                <div class="no-history text-center py-3">
                    <i class="fas fa-history fa-lg text-muted mb-2"></i>
                    <p class="text-muted mb-0 small">Detection history will appear here</p>
                </div>
            `;
            return;
        }

        const historyHTML = recentDetections.map(detection => {
            const timeStr = detection.timestamp.toLocaleTimeString();
            return `
                <div class="detection-entry fade-in">
                    <div class="d-flex justify-content-between align-items-center">
                        <span>${this.formatClassName(detection.class)}</span>
                        <span class="confidence-score">${detection.confidence}%</span>
                    </div>
                    <div class="detection-time">${timeStr}</div>
                </div>
            `;
        }).join('');

        historyContainer.innerHTML = historyHTML;
    }

    /**
     * Format class name for display
     */
    formatClassName(className) {
        return className.split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    /**
     * Add a callback for detection events
     */
    addDetectionCallback(callback) {
        this.detectionCallbacks.push(callback);
    }

    /**
     * Clear detection statistics
     */
    clearStats() {
        this.detectionStats = {
            totalDetections: 0,
            objectCounts: {},
            recentDetections: []
        };
        this.updateDetectionUI();
    }

    /**
     * Get current detection statistics
     */
    getStats() {
        return { ...this.detectionStats };
    }
}

// Global detection manager instance
window.detectionManager = new ObjectDetectionManager();

// Global tracker instance
window.tracker = new Tracker();

// Initialize detection manager when page loads
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Loading object detection model...');
    await window.detectionManager.loadModel();
});
