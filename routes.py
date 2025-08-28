import logging
import json
from flask import render_template, request, jsonify, Response
from app import app
from stream_processor import StreamProcessor
import threading
import time
from urllib.parse import urlparse

# Minimal 1x1 black JPEG for heartbeat fallback when no frame exists yet
# 极简1x1黑色JPEG占位图：在首帧尚未产生时用于心跳包，防止超时
SMALL_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\x09\x09\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c !,'\x1f\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    b"\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xd2\xcf\xff\xd9"
)

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
        stall_timeout = 6.0  # seconds without new frames triggers reconnect / 超过6秒无新帧则断开重连（降低误判）
        heartbeat_interval = 0.2  # seconds; frequent keepalives to prevent worker timeout / 更频繁心跳
        min_frame_interval = 1.0 / 15.0  # Pace to ~15 FPS
        last_sent_frame = None  # Cache last successfully sent JPEG for heartbeat
        # 说明：缓存上一帧，在无新帧或latest为空时用于心跳保活，避免Gunicorn超时
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
                    # Pace output to ~15 FPS without long blocking sleeps (avoids Gunicorn timeout)
                    # 以非阻塞方式限速到约15FPS，避免长时间sleep触发Gunicorn超时
                    now_send = time.time()
                    if last_frame_send_ts and (now_send - last_frame_send_ts) < min_frame_interval:
                        # Not yet time to send next frame. Push heartbeat if due; otherwise brief yieldless wait.
                        # 发送时机未到：若到心跳周期则发送心跳，否则短暂等待
                        if now_send - last_send_ts >= heartbeat_interval:
                            last_send_ts = now_send
                            hb = frame or last_sent_frame or (stream_processor.get_latest_frame() if stream_processor else None) or SMALL_JPEG
                            if hb:
                                ts_hb = str(int(now_send * 1000))
                                yield (b"--" + boundary.encode() + b"\r\n"
                                       b"Content-Type: image/jpeg\r\n"
                                       b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                                       b"Pragma: no-cache\r\n"
                                       b"Expires: 0\r\n"
                                       b"X-Timestamp: " + ts_hb.encode() + b"\r\n"
                                       b"X-Frame-ID: " + str(current_frame_id).encode() + b"\r\n"
                                       b"Content-Length: " + str(len(hb)).encode() + b"\r\n\r\n" + hb + b"\r\n")
                        # Very short sleep to avoid busy loop, but not long enough to risk timeout
                        # 极短休眠避免空转，不至于引发超时
                        time.sleep(0.005)
                        continue
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
                    # Cache last successfully sent frame for future heartbeats
                    # 缓存最近成功发送的帧，用于后续心跳保活
                    last_sent_frame = frame
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
                        # Prefer latest; fallback to last_sent_frame if latest is unavailable
                        # 优先使用最新帧；若不可用则回退至最近发送的帧
                        heartbeat_frame = None
                        if stream_processor:
                            heartbeat_frame = stream_processor.get_latest_frame()
                        if not heartbeat_frame:
                            heartbeat_frame = last_sent_frame or SMALL_JPEG
                        if heartbeat_frame:
                            ts_hb = str(int(time.time() * 1000))
                            yield (b"--" + boundary.encode() + b"\r\n"
                                   b"Content-Type: image/jpeg\r\n"
                                   b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                                   b"Pragma: no-cache\r\n"
                                   b"Expires: 0\r\n"
                                   b"X-Timestamp: " + ts_hb.encode() + b"\r\n"
                                   b"X-Frame-ID: " + str(current_frame_id).encode() + b"\r\n"
                                   b"Content-Length: " + str(len(heartbeat_frame)).encode() + b"\r\n\r\n" + heartbeat_frame + b"\r\n")
                        else:
                            # No frame at all (rare). Yield minimal wait and try again very soon.
                            # 完全无帧（少见）：极短休眠后立即重试
                            time.sleep(0.005)
                    else:
                        # Always keep loop responsive; very short sleep avoids busy-loop yet yields often
                        # 保持循环高响应：极短休眠避免空转，并频繁机会发心跳
                        time.sleep(0.005)
                    
            except GeneratorExit:
                break
            except Exception as e:
                logging.error(f"Error in enhanced MJPEG generator: {e}")
                # Brief pause to avoid tight loop (micro-sleep to ensure worker never blocks long)
                # 极短休眠以避免空转，同时保证worker不长时间阻塞
                time.sleep(0.005)

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
