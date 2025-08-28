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
processing_thread = None
is_processing = False

@app.route('/')
def index():
    """Main page with object detection interface."""
    return render_template('index.html')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Start processing a YouTube live stream."""
    global stream_processor, processing_thread, is_processing
    
    try:
        data = request.get_json()
        youtube_url = data.get('url', '').strip()
        
        if not youtube_url:
            return jsonify({'error': 'YouTube URL is required'}), 400
        
        # Validate YouTube URL format (strict domain whitelist)
        # 严格的域名白名单校验，防止 SSRF
        if not is_valid_youtube_url(youtube_url):
            return jsonify({'error': 'Invalid YouTube URL format (must be a valid YouTube URL)'}), 400
        
        # Stop existing stream if running
        if is_processing:
            stop_stream_processing()
        
        # Initialize stream processor
        stream_processor = StreamProcessor(youtube_url)
        
        # Validate stream URL
        validation_result, error_message = stream_processor.validate_stream()
        if not validation_result:
            logging.error(f"Stream validation failed for URL {youtube_url}: {error_message}")
            return jsonify({'error': f'Unable to access the YouTube stream. Reason: {error_message}'}), 400
        
        # Start processing in background thread
        is_processing = True
        processing_thread = threading.Thread(target=stream_processor.start_processing)
        processing_thread.daemon = True
        processing_thread.start()
        
        logging.info(f"Started processing stream: {youtube_url}")
        return jsonify({'message': 'Stream processing started successfully'})
        
    except Exception as e:
        logging.error(f"Error starting stream: {str(e)}")
        return jsonify({'error': f'Failed to start stream processing: {str(e)}'}), 500

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop the current stream processing."""
    global is_processing
    
    try:
        stop_stream_processing()
        return jsonify({'message': 'Stream processing stopped'})
    except Exception as e:
        logging.error(f"Error stopping stream: {str(e)}")
        return jsonify({'error': f'Failed to stop stream: {str(e)}'}), 500

@app.route('/stream_status')
def stream_status():
    """Get current stream processing status."""
    global stream_processor, is_processing
    
    if not is_processing or not stream_processor:
        return jsonify({
            'active': False,
            'url': None,
            'frame_count': 0,
            'fps': 0
        })
    
    return jsonify({
        'active': True,
        'url': stream_processor.youtube_url,
        'frame_count': stream_processor.frame_count,
        'fps': stream_processor.fps
    })

@app.route('/video_feed')
def video_feed():
    """Get current video frame as JPEG image."""
    global stream_processor
    
    if not stream_processor or not is_processing:
        return Response("No active stream", status=404)
    
    frame = stream_processor.get_latest_frame()
    if frame is not None:
        return Response(frame, mimetype='image/jpeg')
    else:
        return Response("No frame available", status=503)

@app.route('/video_feed_mjpeg')
def video_feed_mjpeg():
    """Enhanced MJPEG streaming with adaptive frame delivery and buffer optimization.
    增强的MJPEG流媒体，支持自适应帧传输和缓冲区优化。
    """
    global stream_processor

    if not stream_processor or not is_processing:
        return Response("No active stream", status=404)

    def generate():
        boundary = "frame"
        last_frame_sent = None

        while is_processing and stream_processor:
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

def stop_stream_processing():
    """Helper function to stop stream processing."""
    global stream_processor, processing_thread, is_processing
    
    is_processing = False
    if stream_processor:
        stream_processor.stop()
        stream_processor = None
    
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=2)
