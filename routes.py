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
        last_frame_id = -1
        frame_skip_count = 0
        max_skip_frames = 2  # Skip at most 2 frames to maintain smoothness
        
        while is_processing and stream_processor:
            try:
                # Use buffered frame with age limit for smoother delivery
                # 使用带年龄限制的缓冲帧以实现更流畅的传输
                frame = stream_processor.get_buffered_frame(max_age_ms=50)
                current_frame_id = stream_processor.frame_count if stream_processor else last_frame_id
                
                # Implement adaptive frame skipping to maintain target frame rate
                # 实现自适应跳帧以保持目标帧率
                if frame is not None and current_frame_id != last_frame_id:
                    # Check if we should skip this frame to maintain smooth playback
                    # 检查是否应跳过此帧以保持流畅播放
                    frame_gap = current_frame_id - last_frame_id
                    if frame_gap > 1 and frame_skip_count < max_skip_frames:
                        frame_skip_count += 1
                        # Remove blocking sleep to prevent worker timeout
                        # 移除阻塞性sleep以防止worker超时
                        continue
                    
                    frame_skip_count = 0
                    last_frame_id = current_frame_id
                    
                    # Enhanced MJPEG headers with performance optimizations
                    # 增强的MJPEG头部，包含性能优化
                    timestamp = str(int(time.time() * 1000))
                    yield (b"--" + boundary.encode() + b"\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                           b"Pragma: no-cache\r\n"
                           b"Expires: 0\r\n"
                           b"X-Timestamp: " + timestamp.encode() + b"\r\n"
                           b"X-Frame-ID: " + str(current_frame_id).encode() + b"\r\n"
                           b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                else:
                    # Non-blocking frame waiting - yield control instead of sleep
                    # 非阻塞帧等待 - 让出控制权而非休眠
                    # Use a very short yield to prevent busy waiting while avoiding worker timeout
                    # 使用极短的让出以防止忙等待，同时避免worker超时
                    import threading
                    threading.Event().wait(0.001)  # Non-blocking minimal wait
                    
            except GeneratorExit:
                break
            except Exception as e:
                logging.error(f"Error in enhanced MJPEG generator: {e}")
                # Use non-blocking recovery pause to prevent worker timeout
                # 使用非阻塞恢复暂停以防止worker超时
                import threading
                threading.Event().wait(0.005)  # Brief non-blocking recovery pause

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
