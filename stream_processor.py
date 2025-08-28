import cv2
import logging
import threading
import time
import subprocess
import json
import requests
from urllib.parse import urlparse, parse_qs
import os
from collections import deque  # For frame buffer implementation
# 用于帧缓冲区实现

class StreamProcessor:
    def __init__(self, youtube_url):
        self.youtube_url = youtube_url
        self.cap = None
        self.is_running = True  # Set to True initially
        self.latest_frame = None
        self.frame_count = 0
        self.fps = 0
        # FPS measurement window state
        # FPS 计算使用滑动窗口统计
        self._fps_window_start = time.time()
        self._fps_window_count = 0
        self.lock = threading.Lock()
        
        # Frame buffer for smoother streaming
        # 帧缓冲区以实现更流畅的流媒体传输
        self.frame_buffer = deque(maxlen=3)  # Small buffer to reduce latency
        self.buffer_lock = threading.Lock()
        
        # Performance monitoring
        # 性能监控
        self.processing_times = deque(maxlen=30)  # Track processing performance
        self.dropped_frames = 0

    def validate_stream(self):
        """Validate if the YouTube URL is actually retrievable for playback.
        通过尝试获取直链来验证能否实际播放。
        """
        try:
            # Prefer testing real playback capability via get_stream_url
            # 优先通过获取直链验证实际播放能力
            stream_url = self.get_stream_url()
            if stream_url:
                return True, "Stream is valid."
            else:
                return False, "Unable to retrieve direct stream URL via yt-dlp."
            
        except subprocess.TimeoutExpired:
            logging.error("Timeout while validating stream")
            return False, "Timeout while validating stream with yt-dlp."
        except Exception as e:
            logging.error(f"Error validating stream: {str(e)}")
            return False, f"An exception occurred: {str(e)}"
    
    def get_stream_url(self):
        """Get the direct stream URL using yt-dlp."""
        try:
            base_cmd = ['yt-dlp', '--get-url', '--format', 'best[height<=720]/best', self.youtube_url]
            attempts = []
            # 1) With cookies if available
            if os.path.exists('cookies.txt'):
                attempts.append(base_cmd + ['--cookies', 'cookies.txt'])
                logging.info("Using local cookies.txt file for getting stream URL.")
            # 2) Try different player clients (some bypass additional checks)
            attempts.append(base_cmd + ['--extractor-args', 'youtube:player_client=android'])
            attempts.append(base_cmd + ['--extractor-args', 'youtube:player_client=ios'])
            # 3) Last fallback: no cookies, default client
            attempts.append(base_cmd)

            for idx, cmd in enumerate(attempts, 1):
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0 and result.stdout.strip():
                        stream_url = result.stdout.strip()
                        logging.info(f"Got stream URL (attempt {idx}): {stream_url[:100]}...")
                        return stream_url
                    else:
                        err = (result.stderr or "").strip()
                        if err:
                            logging.error(f"yt-dlp error getting URL (attempt {idx}): {err}")
                        # Detect bot-check/auth requirement and stop early to avoid endless retries
                        if 'Sign in to confirm you’re not a bot' in err or 'confirm you\u2019re not a bot' in err:
                            logging.error("YouTube requires authentication or bot confirmation. Provide cookies.txt or use authenticated environment.")
                            return None
                except subprocess.TimeoutExpired:
                    logging.error(f"yt-dlp timeout getting URL (attempt {idx})")
                except Exception as inner:
                    logging.error(f"Error running yt-dlp (attempt {idx}): {inner}")
            return None
        
        except Exception as e:
            logging.error(f"Error getting stream URL: {str(e)}")
            return None
    
    def start_processing(self):
        """Start processing the video stream."""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries and self.is_running:
            stream_url = None  # Reset stream_url
            try:
                # Get direct stream URL inside the loop for fresh URL on each retry
                stream_url = self.get_stream_url()
                if not stream_url:
                    logging.error("Failed to get stream URL")
                    retry_count += 1
                    time.sleep(5)
                    continue

                # Open video capture
                self.cap = cv2.VideoCapture(stream_url)
                
                if not self.cap.isOpened():
                    logging.error("Failed to open video stream")
                    retry_count += 1
                    time.sleep(5)
                    continue
                
                # Optimize buffer settings for smoother streaming
        # 优化缓冲区设置以提升流畅度
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Slightly larger buffer to handle jitter
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)   # Reduced timeout for faster recovery
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # Faster timeout to detect issues
        
                # Enable hardware acceleration if available
                # 启用硬件加速（如果可用）
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                except:
                    pass  # Fallback to default codec
                
                self.is_running = True
                logging.info("Started video processing")
                
                # Reset FPS window when (re)starting
                self._fps_window_start = time.time()
                self._fps_window_count = 0
                consecutive_failures = 0
                max_consecutive_failures = 10
                
                while self.is_running:
                    try:
                        ret, frame = self.cap.read()
                        
                        if not ret:
                            consecutive_failures += 1
                            logging.warning(f"Failed to read frame ({consecutive_failures}/{max_consecutive_failures})")
                            
                            if consecutive_failures >= max_consecutive_failures:
                                logging.error("Too many consecutive frame read failures, reconnecting...")
                                break
                            
                            time.sleep(0.1)
                            continue
                        
                        # Reset failure counter on successful read
                        consecutive_failures = 0
                        
                        # Validate frame
                        if frame is None or frame.size == 0:
                            logging.warning("Received empty frame")
                            continue
                        
                        # Update frame counters and FPS (1-second window)
                        # 更新帧计数与 FPS（1 秒窗口）
                        self.frame_count += 1
                        self._fps_window_count += 1
                        now = time.time()
                        elapsed = now - self._fps_window_start
                        if elapsed >= 1.0:
                            # Compute FPS for the past window and reset window counters
                            self.fps = self._fps_window_count / elapsed
                            self._fps_window_start = now
                            self._fps_window_count = 0
                        
                        # Resize frame for processing
                        height, width = frame.shape[:2]
                        if width > 1280:
                            scale = 1280 / width
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Encode frame as JPEG with optimized settings
                        # 使用优化设置编码JPEG帧
                        encode_params = [
                            cv2.IMWRITE_JPEG_QUALITY, 75,  # Reduced quality for better performance
                            cv2.IMWRITE_JPEG_OPTIMIZE, 1,  # Enable JPEG optimization
                            cv2.IMWRITE_JPEG_PROGRESSIVE, 1  # Progressive JPEG for better streaming
                        ]
                        success, buffer = cv2.imencode('.jpg', frame, encode_params)
                        
                        if not success:
                            logging.warning("Failed to encode frame as JPEG")
                            continue
                        
                        # Store frame in buffer for smoother delivery
                        # 将帧存储在缓冲区中以实现更流畅的传输
                        frame_data = buffer.tobytes()
                        with self.buffer_lock:
                            self.frame_buffer.append({
                                'data': frame_data,
                                'timestamp': time.time(),
                                'frame_id': self.frame_count
                            })
                        
                        # Also update latest frame for compatibility
                        # 同时更新最新帧以保持兼容性
                        with self.lock:
                            self.latest_frame = frame_data
                        
                        # Track processing performance
                        # 跟踪处理性能
                        processing_end = time.time()
                        self.processing_times.append(processing_end - now)
                        
                        # Adaptive frame rate control for smoother streaming
                        # 自适应帧率控制以提升流畅度
                        target_fps = 30  # Reduced from 60 for better stability
                        frame_time = 1.0 / target_fps
                        processing_time = time.time() - now
                        sleep_time = max(0.001, frame_time - processing_time)  # Minimum 1ms sleep
                        time.sleep(sleep_time)
                        
                    except Exception as frame_error:
                        logging.error(f"Error processing frame: {str(frame_error)}")
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            break
                        time.sleep(0.1)
                
                # If we exit the frame loop, try to reconnect
                if self.is_running:
                    logging.info("Attempting to reconnect to stream...")
                    self.cleanup()
                    retry_count += 1
                    time.sleep(5)
                else:
                    break
                    
            except Exception as e:
                logging.error(f"Error in stream processing: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    logging.info(f"Retrying stream processing ({retry_count}/{max_retries})...")
                    time.sleep(5)
                else:
                    logging.error("Max retries reached, stopping stream processing")
                    break
            finally:
                if self.cap:
                    self.cap.release()
                    self.cap = None
        
        # Mark processor as no longer running before cleanup so routes see the correct state
        # 在清理之前将处理器标记为未运行，以便路由能正确感知状态
        self.is_running = False
        self.cleanup()
        logging.info("Stream processing stopped")
    
    def get_latest_frame(self):
        """Get the latest processed frame with buffer optimization.
        获取最新处理的帧，使用缓冲区优化。
        """
        with self.buffer_lock:
            if self.frame_buffer:
                # Return the most recent frame from buffer
                # 从缓冲区返回最新帧
                return self.frame_buffer[-1]['data']
        
        # Fallback to direct frame access
        # 回退到直接帧访问
        with self.lock:
            return self.latest_frame
    
    def get_buffered_frame(self, max_age_ms=100):
        """Get a frame from buffer that's not too old.
        从缓冲区获取不太旧的帧。
        """
        current_time = time.time()
        with self.buffer_lock:
            # Find the newest frame that's not too old
            # 找到不太旧的最新帧
            for frame_info in reversed(self.frame_buffer):
                age_ms = (current_time - frame_info['timestamp']) * 1000
                if age_ms <= max_age_ms:
                    return frame_info['data']
        
        # If no recent frame, return the latest available
        # 如果没有最近的帧，返回最新可用的
        return self.get_latest_frame()
    
    def get_performance_stats(self):
        """Get performance statistics.
        获取性能统计信息。
        """
        if not self.processing_times:
            return {'avg_processing_time': 0, 'dropped_frames': self.dropped_frames}
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        return {
            'avg_processing_time': avg_time * 1000,  # Convert to ms
            'dropped_frames': self.dropped_frames,
            'buffer_size': len(self.frame_buffer)
        }
    
    def stop(self):
        """Stop the stream processing."""
        self.is_running = False
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources including frame buffer.
        清理资源包括帧缓冲区。
        """
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear frame buffer
        # 清空帧缓冲区
        with self.buffer_lock:
            self.frame_buffer.clear()
        
        logging.info("Stream processor cleaned up")
