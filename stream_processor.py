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
        self.frame_count = 0
        self.fps = 0
        self._fps_window_start = time.time()
        self._fps_window_count = 0

        # Dedicated, low-contention frame slot for MJPEG streaming
        # 用于 MJPEG 流的专用、低竞争的帧槽
        self._mjpeg_frame = None
        self._mjpeg_lock = threading.Lock()
        
        # Performance monitoring
        # 性能监控
        self.processing_times = deque(maxlen=30)  # Track processing performance
        self.dropped_frames = 0

    def validate_stream(self):
        """Validate if the YouTube URL is accessible and is a live stream."""
        try:
            # Use yt-dlp to get stream info
            cmd = ['yt-dlp', '--dump-json', '--no-download', self.youtube_url]
            if os.path.exists('cookies.txt'):
                cmd.extend(['--cookies', 'cookies.txt'])
                logging.info("Using local cookies.txt file for validation.")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                error_output = result.stderr.strip()
                logging.error(f"yt-dlp error: {error_output}")
                return False, error_output or "yt-dlp returned a non-zero exit code."
            
            info = json.loads(result.stdout)
            
            # Check if it's a live stream
            is_live = info.get('is_live', False)
            if not is_live:
                logging.warning("URL may not be a live stream, but attempting to process anyway")
            
            return True, "Stream is valid."
            
        except subprocess.TimeoutExpired:
            logging.error("Timeout while validating stream")
            return False, "Timeout while validating stream with yt-dlp."
        except Exception as e:
            logging.error(f"Error validating stream: {str(e)}")
            return False, f"An exception occurred: {str(e)}"
    
    def get_stream_url(self):
        """Get the direct stream URL using yt-dlp."""
        try:
            cmd = ['yt-dlp', '--get-url', '--format', 'best[height<=720]', self.youtube_url]
            if os.path.exists('cookies.txt'):
                cmd.extend(['--cookies', 'cookies.txt'])
                logging.info("Using local cookies.txt file for getting stream URL.")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logging.error(f"yt-dlp error getting URL: {result.stderr}")
                return None
            
            stream_url = result.stdout.strip()
            logging.info(f"Got stream URL: {stream_url[:100]}...")
            return stream_url
            
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
                        
                        # Put the latest encoded frame in the dedicated MJPEG slot
                        # 将最新编码的帧放入专用的 MJPEG 槽中
                        with self._mjpeg_lock:
                            self._mjpeg_frame = buffer.tobytes()
                        
                        # Track processing performance
                        # 跟踪处理性能
                        processing_end = time.time()
                        self.processing_times.append(processing_end - now)
                        
                        # Adaptive frame rate control for smoother streaming
                        # 自适应帧率控制以提升流畅度
                        target_fps = 24  # Adjusted from 30 for user request
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
        
        self.cleanup()
        logging.info("Stream processing stopped")
    
    def get_latest_frame(self):
        """Get the latest JPEG frame for MJPEG streaming.
        获取用于 MJPEG 流的最新 JPEG 帧。
        """
        with self._mjpeg_lock:
            return self._mjpeg_frame
    
    
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
            'buffer_size': 1 if self._mjpeg_frame else 0
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
        
        
        logging.info("Stream processor cleaned up")
