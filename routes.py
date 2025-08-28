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
is_processing = False  # Legacy flag; active state should prefer stream_processor.is_running

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
    
    if not stream_processor or not getattr(stream_processor, 'is_running', False):
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
    
    if not stream_processor or not getattr(stream_processor, 'is_running', False):
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

    if not stream_processor or not getattr(stream_processor, 'is_running', False):
        return Response("No active stream", status=404)

    def generate():
        boundary = "frame"
        last_frame_id = -1
        frame_skip_count = 0
        max_skip_frames = 2  # Skip at most 2 frames to maintain smoothness
        
        last_send_ts = time.time()
        last_frame_send_ts = 0.0  # Control max MJPEG send rate
        last_progress_ts = time.time()  # Track progress to detect stalls
        stall_timeout = 4.0  # seconds without new frames triggers reconnect
        heartbeat_interval = 2.0  # seconds; send a keepalive chunk to prevent worker timeout
        min_frame_interval = 1.0 / 15.0  # Pace to ~15 FPS
        while stream_processor and getattr(stream_processor, 'is_running', False):
            try:
                # Use buffered frame with age limit for smoother delivery
                # 使用带年龄限制的缓冲帧以实现更流畅的传输
                frame = stream_processor.get_buffered_frame(max_age_ms=2200)
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
                    last_progress_ts = time.time()  # Progress made
                    
                    # Enhanced MJPEG headers with performance optimizations
                    # 增强的MJPEG头部，包含性能优化
                    # Pace output to ~15 FPS
                    now_send = time.time()
                    if last_frame_send_ts and (now_send - last_frame_send_ts) < min_frame_interval:
                        # sleep the remaining time
                        time.sleep(max(0.0, min_frame_interval - (now_send - last_frame_send_ts)))
                    last_frame_send_ts = time.time()
                    timestamp = str(int(last_frame_send_ts * 1000))
                    yield (b"--" + boundary.encode() + b"\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                           b"Pragma: no-cache\r\n"
                           b"Expires: 0\r\n"
                           b"X-Timestamp: " + timestamp.encode() + b"\r\n"
                           b"X-Frame-ID: " + str(current_frame_id).encode() + b"\r\n"
                           b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                else:
                    # If no progress for too long, break to trigger client reconnection
                    # 若长时间无新帧，断开以触发客户端重连
                    if time.time() - last_progress_ts > stall_timeout:
                        logging.warning("MJPEG stall detected on server (>4s no new frames). Closing stream to trigger reconnect.")
                        break
                    # Heartbeat: resend the latest JPEG frame as keepalive to prevent worker timeout
                    # 心跳：重发最近的JPEG帧作为保活，避免 Gunicorn 超时且不破坏浏览器解码
                    now_ts = time.time()
                    if now_ts - last_send_ts >= heartbeat_interval:
                        last_send_ts = now_ts
                        latest = stream_processor.get_latest_frame() if stream_processor else None
                        if latest:
                            timestamp = str(int(time.time() * 1000))
                            yield (b"--" + boundary.encode() + b"\r\n"
                                   b"Content-Type: image/jpeg\r\n"
                                   b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                                   b"Pragma: no-cache\r\n"
                                   b"Expires: 0\r\n"
                                   b"X-Timestamp: " + timestamp.encode() + b"\r\n"
                                   b"X-Frame-ID: " + str(current_frame_id).encode() + b"\r\n"
                                   b"Content-Length: " + str(len(latest)).encode() + b"\r\n\r\n" + latest + b"\r\n")
                        else:
                            time.sleep(0.05)
                    else:
                        # Brief sleep to avoid busy loop
                        time.sleep(0.05)
                    
            except GeneratorExit:
                break
            except Exception as e:
                logging.error(f"Error in enhanced MJPEG generator: {e}")
                # Brief pause to avoid tight loop
                time.sleep(0.05)

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
