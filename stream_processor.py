import cv2
import logging
import threading
import time
import subprocess
import json
import requests
from urllib.parse import urlparse, parse_qs
import os

class StreamProcessor:
    def __init__(self, youtube_url):
        self.youtube_url = youtube_url
        self.cap = None
        self.is_running = False
        self.latest_frame = None
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.lock = threading.Lock()
        
    def validate_stream(self):
        """Validate if the YouTube URL is accessible and is a live stream."""
        try:
            # Use yt-dlp to get stream info
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-download',
                self.youtube_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logging.error(f"yt-dlp error: {result.stderr}")
                return False
            
            info = json.loads(result.stdout)
            
            # Check if it's a live stream
            is_live = info.get('is_live', False)
            if not is_live:
                # Try to get direct stream URL anyway for non-live videos
                logging.warning("URL may not be a live stream, but attempting to process anyway")
            
            return True
            
        except subprocess.TimeoutExpired:
            logging.error("Timeout while validating stream")
            return False
        except Exception as e:
            logging.error(f"Error validating stream: {str(e)}")
            return False
    
    def get_stream_url(self):
        """Get the direct stream URL using yt-dlp."""
        try:
            cmd = [
                'yt-dlp',
                '--get-url',
                '--format', 'worst[height>=240][height<=360]/best[height<=360]',
                '--no-warnings',
                '--hls-prefer-native',
                '--hls-use-mpegts',
                self.youtube_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
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
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get direct stream URL
                stream_url = self.get_stream_url()
                if not stream_url:
                    if attempt < max_retries - 1:
                        logging.warning(f"Failed to get stream URL, attempt {attempt + 1}/{max_retries}")
                        time.sleep(2)
                        continue
                    else:
                        logging.error("Failed to get stream URL after all attempts")
                        return
            
                # Open video capture with optimized settings
                self.cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                
                if not self.cap.isOpened():
                    if attempt < max_retries - 1:
                        logging.warning(f"Failed to open video stream, attempt {attempt + 1}/{max_retries}")
                        time.sleep(2)
                        continue
                    else:
                        logging.error("Failed to open video stream after all attempts")
                        return
            
                # Optimize for low latency streaming
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 10)  # Even lower FPS for stability
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                
                # Connection successful, break retry loop
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Connection attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(2)
                    continue
                else:
                    logging.error(f"All connection attempts failed: {str(e)}")
                    return
        
        try:
            
            self.is_running = True
            self.failed_attempts = 0
            logging.info("Started video processing")
            
            frame_time_start = time.time()
            
            while self.is_running and self.cap is not None:
                ret, frame = self.cap.read()
                
                if not ret:
                    logging.warning("Failed to read frame, attempting to reconnect...")
                    time.sleep(0.5)
                    # Try to reconnect after 3 failed attempts
                    if hasattr(self, 'failed_attempts'):
                        self.failed_attempts += 1
                    else:
                        self.failed_attempts = 1
                    
                    if self.failed_attempts > 3:
                        logging.error("Too many failed attempts, stopping stream")
                        break
                    continue
                else:
                    # Reset failed attempts on successful read
                    self.failed_attempts = 0
                
                # Update frame count and FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - frame_time_start >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_time)
                    frame_time_start = current_time
                
                # Resize frame for processing - much smaller for faster processing
                height, width = frame.shape[:2]
                if width > 640:
                    scale = 640 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                
                # Encode frame as JPEG with lower quality for faster processing
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                
                # Store latest frame
                with self.lock:
                    self.latest_frame = buffer.tobytes()
                
                # Optimized delay for stability
                time.sleep(0.1)  # ~10 FPS - more stable processing
                
        except Exception as e:
            logging.error(f"Error in stream processing: {str(e)}")
        finally:
            self.cleanup()
    
    def get_latest_frame(self):
        """Get the latest processed frame."""
        with self.lock:
            return self.latest_frame
    
    def stop(self):
        """Stop the stream processing."""
        self.is_running = False
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
        logging.info("Stream processor cleaned up")
