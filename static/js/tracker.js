/**
 * @file tracker.js
 * @description Implements a multi-object tracker for real-time video streams.
 *              Supports interactive target selection, IOU-based matching, and motion prediction.
 * @description 实现一个多目标追踪器，用于实时视频流。支持交互式目标选择、
 *              基于IOU的匹配和运动预测。
 */

class Tracker {
    /**
     * Initializes the multi-object Tracker instance.
     * @param {object} [options={}] - Configuration options.
     * @param {number} [options.iou_threshold=0.5] - IOU threshold for matching.
     * @param {number} [options.max_lost_frames=10] - Max frames to keep a lost track.
     */
    constructor({ iou_threshold = 0.5, max_lost_frames = 10 } = {}) {
        this.iou_threshold = iou_threshold;
        this.max_lost_frames = max_lost_frames;
        this.next_track_id = 0;
        /**
         * @property {Array<object>} tracks - Array of active tracks.
         */
        this.tracks = [];
        /**
         * @property {Array<string>} colors - Color palette for visualizing different tracks.
         */
        this.colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff"];
    }

    _calculate_iou(box1, box2) {
        const [x1, y1, w1, h1] = box1;
        const [x2, y2, w2, h2] = box2;
        const ix = Math.max(x1, x2);
        const iy = Math.max(y1, y2);
        const iw = Math.min(x1 + w1, x2 + w2) - ix;
        const ih = Math.min(y1 + h1, y2 + h2) - iy;
        if (iw <= 0 || ih <= 0) return 0;
        const inter_area = iw * ih;
        const union_area = w1 * h1 + w2 * h2 - inter_area;
        return inter_area / union_area;
    }

    /**
     * Handles user clicks to add or remove tracks.
     * - Clicking a detected object: adds it to tracks if new, or removes it if already tracked.
     * - Clicking empty space: clears all tracks.
     * @param {number} x - Click x-coordinate.
     * @param {number} y - Click y-coordinate.
     * @param {Array<object>} detections - Current frame's detections.
     */
    select_target(x, y, detections) {
        // Check if click is on an existing track to remove it
        const clicked_track_index = this.tracks.findIndex(track => {
            const [tx, ty, tw, th] = track.bbox;
            return x >= tx && x <= tx + tw && y >= ty && y <= ty + th;
        });

        if (clicked_track_index > -1) {
            console.log(`Track ${this.tracks[clicked_track_index].id} removed.`);
            this.tracks.splice(clicked_track_index, 1);
            return;
        }

        // Check if click is on a new detection to add it
        const best_match = detections.find(detection => {
            const [dx, dy, dw, dh] = detection.bbox;
            return x >= dx && x <= dx + dw && y >= dy && y <= dy + dh;
        });

        if (best_match) {
            const centerX = best_match.bbox[0] + best_match.bbox[2] / 2;
            const centerY = best_match.bbox[1] + best_match.bbox[3] / 2;
            const new_track = {
                id: this.next_track_id++,
                bbox: best_match.bbox,
                class: best_match.class,
                frames_lost: 0,
                history: [[centerX, centerY]],
                last_velocity: { x: 0, y: 0 },
                color: this.colors[this.next_track_id % this.colors.length]
            };
            this.tracks.push(new_track);
            console.log(`New track ${new_track.id} added.`);
        } else {
            console.log("Clearing all tracks.");
            this.tracks = []; // Clear all tracks if background is clicked
        }
    }

    /**
     * Updates all active tracks with new detections using a greedy matching algorithm.
     * @param {Array<object>} detections - Detections from the current frame.
     */
    update(detections) {
        if (this.tracks.length === 0) return;

        let remaining_detections = [...detections];
        const matched_indices = new Set();

        // Greedy matching: For each track, find the best-matching detection
        this.tracks.forEach(track => {
            let best_match = { pred: null, iou: -1, index: -1 };

            remaining_detections.forEach((pred, index) => {
                if (matched_indices.has(index)) return;
                const iou = this._calculate_iou(track.bbox, pred.bbox);
                if (iou > best_match.iou) {
                    best_match = { pred, iou, index };
                }
            });

            if (best_match.iou >= this.iou_threshold) {
                // --- Track Matched ---
                const { pred, index } = best_match;
                const old_center_x = track.bbox[0] + track.bbox[2] / 2;
                const old_center_y = track.bbox[1] + track.bbox[3] / 2;
                const new_center_x = pred.bbox[0] + pred.bbox[2] / 2;
                const new_center_y = pred.bbox[1] + pred.bbox[3] / 2;

                track.last_velocity = { x: new_center_x - old_center_x, y: new_center_y - old_center_y };
                track.bbox = pred.bbox;
                track.class = pred.class;
                track.frames_lost = 0;
                track.history.push([new_center_x, new_center_y]);
                matched_indices.add(index);
            } else {
                // --- Track Lost ---
                track.frames_lost++;
                const [bx, by, bw, bh] = track.bbox;
                track.bbox = [bx + track.last_velocity.x, by + track.last_velocity.y, bw, bh];
                const p_center_x = track.bbox[0] + track.bbox[2] / 2;
                const p_center_y = track.bbox[1] + track.bbox[3] / 2;
                track.history.push([p_center_x, p_center_y]);
            }
            if (track.history.length > 30) track.history.shift();
        });

        // Remove tracks that have been lost for too long
        this.tracks = this.tracks.filter(track => track.frames_lost <= this.max_lost_frames);
    }

    /**
     * Draws all tracked objects onto the canvas.
     * @param {CanvasRenderingContext2D} ctx - The canvas rendering context.
     */
    draw(ctx) {
        this.tracks.forEach(track => {
            const { bbox, id, history, 'class': className, frames_lost, color } = track;
            const [x, y, w, h] = bbox;
            const is_predicted = frames_lost > 0;
            const current_color = is_predicted ? '#FFA500' : color; // Use orange for predicted

            // Draw motion trail
            if (history.length > 1) {
                ctx.strokeStyle = color;
                ctx.globalAlpha = 0.7;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(history[0][0], history[0][1]);
                for (let i = 1; i < history.length; i++) {
                    ctx.lineTo(history[i][0], history[i][1]);
                }
                ctx.stroke();
                ctx.globalAlpha = 1.0;
            }

            // Draw bounding box
            ctx.strokeStyle = current_color;
            ctx.lineWidth = is_predicted ? 2 : 4;
            ctx.strokeRect(x, y, w, h);

            // Draw label
            const label = `ID: ${id}`;
            ctx.font = 'bold 16px Arial';
            const textWidth = ctx.measureText(label).width;
            ctx.fillStyle = current_color;
            ctx.fillRect(x, y - 25, textWidth + 10, 25);
            ctx.fillStyle = '#000000';
            ctx.fillText(label, x + 5, y - 8);
        });
    }
}
