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
        self.is_running = True  # Set to True initially
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
                '--format', 'best[height<=720]',
                self.youtube_url
            ]
            
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
            try:
                # Get direct stream URL
                stream_url = self.get_stream_url()
                if not stream_url:
                    logging.error("Failed to get stream URL")
                    retry_count += 1
                    time.sleep(5)
                    continue
                
                # Open video capture with retry logic
                self.cap = cv2.VideoCapture(stream_url)
                
                if not self.cap.isOpened():
                    logging.error("Failed to open video stream")
                    retry_count += 1
                    time.sleep(5)
                    continue
                
                # Set buffer size to reduce latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Set timeout for read operations
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
                
                self.is_running = True
                logging.info("Started video processing")
                
                frame_time_start = time.time()
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
                        
                        # Update frame count and FPS
                        self.frame_count += 1
                        current_time = time.time()
                        if current_time - frame_time_start >= 1.0:
                            self.fps = self.frame_count / (current_time - self.last_time)
                            frame_time_start = current_time
                        
                        # Resize frame for processing
                        height, width = frame.shape[:2]
                        if width > 1280:
                            scale = 1280 / width
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Encode frame as JPEG
                        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        
                        if not success:
                            logging.warning("Failed to encode frame as JPEG")
                            continue
                        
                        # Store latest frame
                        with self.lock:
                            self.latest_frame = buffer.tobytes()
                        
                        # Small delay to prevent overwhelming the system
                        time.sleep(0.016)  # ~60 FPS to match frontend
                        
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
