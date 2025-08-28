import logging
import json
from flask import render_template, request, jsonify, Response
from app import app
from stream_processor import StreamProcessor
import threading
import time
from urllib.parse import urlparse

# Global stream processor instance
stream_processor = None
stream_thread = None

@app.route('/')
def index():
    """Main page with object detection interface."""
    return render_template('index.html')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Start processing a YouTube live stream."""
    global stream_processor, stream_thread
    
    data = request.get_json()
    youtube_url = data.get('youtube_url')
    
    if not youtube_url:
        return jsonify({"error": "youtube_url is required"}), 400

    if stream_processor and stream_processor.is_running:
        return jsonify({"message": "Stream is already running"}), 200

    logging.info(f"Started processing stream: {youtube_url}")
    
    # Initialize and start the stream processor
    stream_processor = StreamProcessor()
    stream_thread = threading.Thread(target=stream_processor.start_processing, args=(youtube_url,), daemon=True)
    stream_thread.start()
    
    # Wait a moment for the stream to initialize before returning
    time.sleep(2) # Give it time to get the first frames
    
    return jsonify({"message": "Stream processing started successfully"})

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop the current stream processing."""
    global stream_processor, stream_thread
    
    if stream_processor:
        stream_processor.stop()
        if stream_thread:
            stream_thread.join() # Wait for the thread to finish
        stream_processor = None
    
    return jsonify({"message": "Stream stopped"})

@app.route('/stream_status')
def stream_status():
    """Get current stream processing status."""
    global stream_processor
    
    if not stream_processor:
        return jsonify({
            'active': False,
            'url': None,
            'frame_count': 0,
            'fps': 0
        })
    
    return jsonify({
        'active': stream_processor.is_running,
        'url': stream_processor.youtube_url,
        'frame_count': stream_processor.frame_count,
        'fps': stream_processor.fps
    })

@app.route('/video_feed')
def video_feed():
    """Get current video frame as JPEG image."""
    global stream_processor
    
    if not stream_processor or not stream_processor.is_running:
        return Response("No active stream", status=404)
    
    frame = stream_processor.get_latest_frame()
    if frame is not None:
        return Response(frame, mimetype='image/jpeg')
    else:
        return Response("No frame available", status=503)

def gen_frames():
    """Generate frame-by-frame for video stream."""
    frame_count = 0
    none_frame_count = 0
    max_none_frames = 100 # ~3 seconds of no frames

    try:
        while True:
            frame = stream_processor.get_latest_frame()
            if frame:
                none_frame_count = 0 # Reset counter on success
                frame_count += 1
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                none_frame_count += 1
                logging.warning("gen_frames: get_latest_frame() returned None. Consecutive None frames: %d", none_frame_count)
                if none_frame_count > max_none_frames:
                    logging.error("gen_frames: No frame received for too long. Closing MJPEG stream.")
                    break

            time.sleep(1/30) # Limit frame rate
    except Exception as e:
        logging.error(f"Exception in gen_frames: {e}")
    finally:
        logging.info("gen_frames: Exiting generator. Total frames sent: %d", frame_count)

@app.route('/video_feed_mjpeg')
def video_feed_mjpeg():
    """Enhanced MJPEG streaming with adaptive frame delivery and buffer optimization.
    增强的MJPEG流媒体，支持自适应帧传输和缓冲区优化。
    """
    global stream_processor

    if not stream_processor or not stream_processor.is_running:
        return Response("No active stream", status=404)

    def generate():
        boundary = "frame"
        last_frame_sent = None

        while stream_processor and stream_processor.is_running:
            try:
                frame = stream_processor.get_latest_frame()

                # If frame is not available, wait briefly and retry.
                # This handles the initial startup race condition.
                # 如果帧不可用，请稍等片刻然后重试。这可以解决初始启动竞态条件。
                if frame is None:
                    time.sleep(0.1)  # Wait for the first frame to be ready
                    continue

                # Send frame only if it's new to avoid redundant transmissions
                # 仅当帧为新时才发送，以避免冗余传输
                if frame != last_frame_sent:
                    last_frame_sent = frame
                    yield (
                        b"--" + boundary.encode() + b"\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + 
                        frame + b"\r\n"
                    )
                
                # Control the stream's frame rate to about 30 FPS
                # 将流的帧率控制在约 30 FPS
                time.sleep(1 / 30)

            except GeneratorExit:
                break
            except Exception as e:
                logging.error(f"Error in MJPEG generator: {e}")
                time.sleep(0.1)  # Sleep longer on error

    return Response(generate(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0',
                        'Connection': 'keep-alive',
                        'X-Accel-Buffering': 'no'  # Disable nginx buffering for real-time streaming
                    })

def is_valid_youtube_url(url: str) -> bool:
    """Validate YouTube URL strictly by hostname whitelist and scheme.
    通过严格的主机名白名单与协议校验验证 YouTube URL。
    """
    try:
        p = urlparse(url)
        if p.scheme not in {"http", "https"}:
            return False
        host = (p.netloc or '').lower()
        # Allow typical YouTube hosts
        allowed = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
        # Strip port if present
        host = host.split(':', 1)[0]
        return host in allowed and bool(p.path)
    except Exception:
        return False
