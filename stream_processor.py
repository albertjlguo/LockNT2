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
import yt_dlp  # For YouTube-DL functionality
# 用于帧缓冲区实现

class StreamProcessor:
    def __init__(self):
        self.youtube_url = None
        self.stream_url = None
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
    
    def _get_stream_url(self):
        if not self.youtube_url:
            logging.error("YouTube URL is not set.")
            return None

        self._status = "Getting stream URL..."
        logging.info(f"Attempting to get stream URL for {self.youtube_url}")
        try:
            ydl_opts = {
                'format': 'best[height<=720]',
                'quiet': True,
                'no_warnings': True,
                'cookiefile': 'cookies.txt'
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(self.youtube_url, download=False)
                stream_url = info_dict.get('url')
                if stream_url:
                    logging.info(f"Got new stream URL: {stream_url[:100]}...")
                    self._status = "Got stream URL. Initializing video capture..."
                    return stream_url
                else:
                    self._status = "Failed to get stream URL."
                    logging.error("Failed to get stream URL.")
                    return None
        except Exception as e:
            self._status = f"Error getting stream URL: {e}"
            logging.error(f"Error getting stream URL: {e}")
            return None
    
    def _reconnect(self):
        self._status = "Reconnecting..."
        if self.cap:
            self.cap.release()
            self.cap = None

        self.stream_url = self._get_stream_url()
        if self.stream_url:
            self.cap = cv2.VideoCapture(self.stream_url)
            if self.cap.isOpened():
                self._status = "Reconnected successfully. Resuming stream."
                logging.info("Reconnected and resumed video capture.")
                return True
        
        self._status = "Failed to reconnect."
        logging.error("Failed to create VideoCapture from new stream URL.")
        return False

    def start_processing(self):
        """Start processing the video stream."""
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries and self.is_running:
            try:
                self.stream_url = self._get_stream_url()
                if not self.stream_url:
                    logging.error("Failed to get stream URL")
                    raise IOError("Failed to get stream URL")

                self.cap = cv2.VideoCapture(self.stream_url)
                if not self.cap.isOpened():
                    logging.error("Failed to open video stream")
                    raise IOError("Failed to open video stream")

                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

                logging.info("Started video processing")
                self._fps_window_start = time.time()
                self._fps_window_count = 0
                consecutive_failures = 0
                max_consecutive_failures = 10

                while self.is_running:
                    ret, frame = self.cap.read()
                    if not ret:
                        consecutive_failures += 1
                        logging.warning(f"Failed to read frame ({consecutive_failures}/{max_consecutive_failures})")
                        if consecutive_failures >= max_consecutive_failures:
                            logging.error("Max consecutive read failures reached. Breaking to reconnect.")
                            break  # Break inner loop to trigger reconnection logic
                        time.sleep(0.1)
                        continue
                    
                    consecutive_failures = 0

                    if frame is None or frame.size == 0:
                        logging.warning("Received empty frame")
                        continue

                    self.frame_count += 1
                    self._fps_window_count += 1
                    now = time.time()
                    elapsed = now - self._fps_window_start
                    if elapsed >= 1.0:
                        self.fps = self._fps_window_count / elapsed
                        self._fps_window_start = now
                        self._fps_window_count = 0

                    height, width = frame.shape[:2]
                    if width > 1280:
                        scale = 1280 / width
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))

                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
                    success, buffer = cv2.imencode('.jpg', frame, encode_params)
                    if not success:
                        logging.warning("Failed to encode frame as JPEG")
                        continue

                    with self._mjpeg_lock:
                        self._mjpeg_frame = buffer.tobytes()

                    target_fps = 24
                    frame_time = 1.0 / target_fps
                    processing_time = time.time() - now
                    sleep_time = max(0.001, frame_time - processing_time)
                    time.sleep(sleep_time)

                # If inner loop breaks (e.g., due to read failures), we'll try to reconnect.
                if not self.is_running:
                    break # Exit outer loop if stop() was called

            except Exception as e:
                logging.error(f"Error in stream processing loop: {str(e)}")
            finally:
                if self.cap:
                    self.cap.release()
                    self.cap = None

            # Reconnection logic
            if self.is_running:
                retry_count += 1
                logging.info(f"Attempting to reconnect... ({retry_count}/{max_retries})")
                time.sleep(5)
            else:
                break

        self.cleanup()
        logging.info("Stream processing stopped after max retries or manual stop.")

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
