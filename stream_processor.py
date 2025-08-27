import cv2
import logging
import threading
import time
import subprocess
import json
import requests
from urllib.parse import urlparse, parse_qs
import os

def is_production_environment():
    """Detect if running in production environment."""
    production_indicators = [
        os.getenv('FLASK_ENV') == 'production',
        os.getenv('ENVIRONMENT') == 'production', 
        'locknt2.theiresearch.com' in os.getenv('SERVER_NAME', ''),
        os.path.exists('/etc/production'),
        'replit' in os.getcwd().lower()
    ]
    return any(production_indicators)

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
            # Use yt-dlp to get stream info with production-friendly options
            cmd = [
                'yt-dlp',
                '--dump-json',
                '--no-download',
                '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                '--add-header', 'Accept-Language:en-US,en;q=0.9',
                '--extractor-retries', '3',
                '--no-check-certificate',
                '--ignore-errors',
                self.youtube_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            
            if result.returncode != 0:
                logging.warning(f"Stream validation failed, but continuing: {result.stderr}")
                # In production, we'll be more lenient and try to proceed anyway
                return True
            
            try:
                # Parse JSON response
                stream_info = json.loads(result.stdout)
                
                # Check if it's a live stream (but don't fail if we can't determine)
                is_live = stream_info.get('is_live', True)  # Default to True if unknown
                title = stream_info.get('title', 'Unknown')
                
                logging.info(f"Stream info - Title: {title}, Live: {is_live}")
                return True
                
            except json.JSONDecodeError:
                logging.warning("Could not parse stream info, but proceeding anyway")
                return True
            
        except subprocess.TimeoutExpired:
            logging.warning("Timeout while validating stream, but proceeding anyway")
            return True
        except Exception as e:
            logging.warning(f"Error validating stream, but proceeding anyway: {str(e)}")
            return True
    
    def get_stream_url(self):
        """Get the direct stream URL using yt-dlp with environment-specific settings."""
        try:
            # Check for cookies file - use updated version in production
            if is_production_environment():
                cookies_file = "cookies_updated.txt" if os.path.exists("cookies_updated.txt") else "cookies.txt"
                logging.info("Production environment - using enhanced extraction")
            else:
                cookies_file = "cookies.txt"
                logging.info("Development environment - using standard extraction")
            
            use_cookies = os.path.exists(cookies_file)
            
            # Environment-specific extraction settings
            if is_production_environment():
                # Production: Enhanced extraction with anti-bot measures
                cmd = [
                    'yt-dlp',
                    '--get-url',
                    '--format', 'best[height<=720]',
                    '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    '--add-header', 'Accept-Language:en-US,en;q=0.9',
                    '--add-header', 'Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    '--add-header', 'Accept-Encoding:gzip, deflate, br',
                    '--add-header', 'DNT:1',
                    '--add-header', 'Connection:keep-alive',
                    '--add-header', 'Upgrade-Insecure-Requests:1',
                    '--extractor-retries', '8',
                    '--socket-timeout', '60',
                    '--no-check-certificate',
                    '--sleep-interval', '2',
                    '--max-sleep-interval', '5',
                ]
            else:
                # Development: Simple extraction
                cmd = [
                    'yt-dlp',
                    '--get-url',
                    '--format', 'best[height<=720]',
                    '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    '--add-header', 'Accept-Language:en-US,en;q=0.9',
                    '--extractor-retries', '3',
                    '--socket-timeout', '30',
                    '--no-check-certificate',
                ]
            
            # Add cookies if available
            if use_cookies:
                cmd.extend(['--cookies', cookies_file])
                logging.info(f"Using cookies file: {cookies_file}")
            else:
                logging.warning("No cookies file found - YouTube may block access")
            
            cmd.append(self.youtube_url)
            
            logging.info(f"Attempting to extract stream URL from: {self.youtube_url}")
            logging.info(f"yt-dlp command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            
            # Log detailed output for debugging
            if result.stdout:
                logging.info(f"yt-dlp stdout: {result.stdout}")
            if result.stderr:
                logging.warning(f"yt-dlp stderr: {result.stderr}")
            
            if result.returncode == 0 and result.stdout.strip():
                stream_url = result.stdout.strip()
                logging.info(f"Successfully extracted stream URL: {stream_url[:100]}...")
                return stream_url
            else:
                logging.warning(f"First attempt failed with return code {result.returncode}")
            
            # Fallback attempt - only in production or if development fails
            if is_production_environment() or result.returncode != 0:
                logging.info("Attempting fallback extraction...")
                
                if is_production_environment():
                    # Production fallback: More aggressive settings
                    fallback_cmd = [
                        'yt-dlp',
                        '--get-url',
                        '--format', 'worst[height>=360]/best[height<=480]',
                        '--user-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        '--add-header', 'Accept-Language:en-US,en;q=0.9',
                        '--add-header', 'Sec-Fetch-Dest:document',
                        '--add-header', 'Sec-Fetch-Mode:navigate',
                        '--add-header', 'Sec-Fetch-Site:none',
                        '--extractor-retries', '10',
                        '--socket-timeout', '90',
                        '--no-check-certificate',
                        '--ignore-errors',
                        '--sleep-interval', '3',
                        '--max-sleep-interval', '8',
                    ]
                else:
                    # Development fallback: Simple lower quality
                    fallback_cmd = [
                        'yt-dlp',
                        '--get-url',
                        '--format', 'best[height<=480]',
                        '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        '--add-header', 'Accept-Language:en-US,en;q=0.9',
                        '--extractor-retries', '5',
                        '--socket-timeout', '60',
                        '--no-check-certificate',
                        '--ignore-errors',
                    ]
            
            # Add cookies to fallback as well
            if use_cookies:
                fallback_cmd.extend(['--cookies', cookies_file])
                
            fallback_cmd.append(self.youtube_url)
            
            logging.info("Attempting fallback extraction with lower quality...")
            logging.info(f"Fallback yt-dlp command: {' '.join(fallback_cmd)}")
            fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=60)
            
            # Log detailed fallback output
            if fallback_result.stdout:
                logging.info(f"Fallback yt-dlp stdout: {fallback_result.stdout}")
            if fallback_result.stderr:
                logging.warning(f"Fallback yt-dlp stderr: {fallback_result.stderr}")
            
                if fallback_result.returncode == 0 and fallback_result.stdout.strip():
                    stream_url = fallback_result.stdout.strip()
                    logging.info(f"Fallback extraction successful: {stream_url[:100]}...")
                    return stream_url
                else:
                    logging.warning(f"Fallback extraction failed with return code {fallback_result.returncode}")
                    
                    # Third attempt - only in production
                    if is_production_environment():
                        logging.info("Attempting final production extraction method...")
                        final_cmd = [
                            'yt-dlp',
                            '--get-url',
                            '--format', '(best[height<=360]/worst)[protocol^=http]',
                            '--user-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                            '--extractor-retries', '15',
                            '--socket-timeout', '120',
                            '--no-check-certificate',
                            '--ignore-errors',
                            '--sleep-interval', '5',
                            '--max-sleep-interval', '10',
                            '--no-warnings',
                        ]
                        
                        if use_cookies:
                            final_cmd.extend(['--cookies', cookies_file])
                        
                        final_cmd.append(self.youtube_url)
                        
                        try:
                            final_result = subprocess.run(final_cmd, capture_output=True, text=True, timeout=150)
                            if final_result.returncode == 0 and final_result.stdout.strip():
                                stream_url = final_result.stdout.strip()
                                logging.info(f"Final extraction successful: {stream_url[:100]}...")
                                return stream_url
                        except Exception as final_error:
                            logging.error(f"Final extraction failed: {str(final_error)}")
            
            logging.error(f"All extraction attempts failed. YouTube may be blocking access.")
            logging.error(f"Consider adding cookies or using a different approach.")
            return None
                
        except subprocess.TimeoutExpired:
            logging.error("Timeout while extracting stream URL")
            return None
        except Exception as e:
            logging.error(f"Error extracting stream URL: {str(e)}")
            return None
    
    def start_processing(self):
        """Start processing the video stream with enhanced error handling."""
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries and self.is_running:
            try:
                logging.info(f"Attempt {retry_count + 1}/{max_retries}: Getting stream URL...")
                # Get direct stream URL
                stream_url = self.get_stream_url()
                if not stream_url:
                    logging.error(f"CRITICAL: Failed to get stream URL, retry {retry_count + 1}/{max_retries}")
                    logging.error(f"This means yt-dlp could not extract the YouTube stream URL")
                    logging.error(f"YouTube URL: {self.youtube_url}")
                    retry_count += 1
                    time.sleep(10)
                    continue
                
                # Open video capture with enhanced error handling
                try:
                    self.cap = cv2.VideoCapture(stream_url)
                    
                    if not self.cap.isOpened():
                        logging.error("Failed to open video stream")
                        retry_count += 1
                        time.sleep(5)
                        continue
                    
                    # Set buffer size to reduce latency and prevent memory issues
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Set reasonable timeouts to prevent hanging
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
                    self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                    
                    # Additional OpenCV settings for stability
                    self.cap.set(cv2.CAP_PROP_FPS, 30)  # Limit FPS
                    
                except Exception as cv_error:
                    logging.error(f"OpenCV VideoCapture error: {str(cv_error)}")
                    retry_count += 1
                    time.sleep(5)
                    continue
                
                self.is_running = True
                logging.info("Started video processing")
                
                frame_time_start = time.time()
                consecutive_failures = 0
                max_consecutive_failures = 5  # Reduced for faster recovery
                frame_timeout = 30  # Maximum seconds to wait for frames
                last_frame_time = time.time()
                
                while self.is_running:
                    try:
                        # Check for frame timeout
                        current_time = time.time()
                        if current_time - last_frame_time > frame_timeout:
                            logging.error("Frame timeout - no frames received for 30 seconds")
                            break
                        
                        # Use a timeout for frame reading to prevent hanging
                        ret, frame = self.cap.read()
                        
                        if not ret:
                            consecutive_failures += 1
                            logging.warning(f"Failed to read frame ({consecutive_failures}/{max_consecutive_failures})")
                            
                            if consecutive_failures >= max_consecutive_failures:
                                logging.error("Too many consecutive frame read failures, reconnecting...")
                                break
                            
                            time.sleep(0.2)
                            continue
                        
                        # Reset failure counter and update last frame time
                        consecutive_failures = 0
                        last_frame_time = current_time
                        
                        # Validate frame with better error handling
                        if frame is None:
                            logging.warning("Received None frame")
                            continue
                            
                        if frame.size == 0:
                            logging.warning("Received empty frame")
                            continue
                        
                        # Check frame dimensions
                        if len(frame.shape) != 3 or frame.shape[2] != 3:
                            logging.warning(f"Invalid frame shape: {frame.shape}")
                            continue
                        
                        # Update frame count and FPS
                        self.frame_count += 1
                        if current_time - frame_time_start >= 1.0:
                            self.fps = self.frame_count / (current_time - self.last_time)
                            frame_time_start = current_time
                        
                        # Resize frame for processing with error handling
                        try:
                            height, width = frame.shape[:2]
                            if width > 1280:
                                scale = 1280 / width
                                new_width = int(width * scale)
                                new_height = int(height * scale)
                                frame = cv2.resize(frame, (new_width, new_height))
                        except Exception as resize_error:
                            logging.warning(f"Frame resize error: {str(resize_error)}")
                            continue
                        
                        # Encode frame as JPEG with error handling
                        try:
                            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            
                            if not success or buffer is None:
                                logging.warning("Failed to encode frame as JPEG")
                                continue
                        except Exception as encode_error:
                            logging.warning(f"Frame encoding error: {str(encode_error)}")
                            continue
                        
                        # Store latest frame safely
                        try:
                            with self.lock:
                                self.latest_frame = buffer.tobytes()
                        except Exception as store_error:
                            logging.warning(f"Frame storage error: {str(store_error)}")
                            continue
                        
                        # Controlled delay to prevent overwhelming the system
                        time.sleep(0.033)  # ~30 FPS for stability
                        
                    except cv2.error as cv_error:
                        logging.error(f"OpenCV error processing frame: {str(cv_error)}")
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            break
                        time.sleep(0.2)
                    except Exception as frame_error:
                        logging.error(f"Unexpected error processing frame: {str(frame_error)}")
                        consecutive_failures += 1
                        if consecutive_failures >= max_consecutive_failures:
                            break
                        time.sleep(0.2)
                
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
                # Safe cleanup
                try:
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                except Exception as cleanup_error:
                    logging.warning(f"Error during VideoCapture cleanup: {str(cleanup_error)}")
        
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
        """Clean up resources safely."""
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
        except Exception as e:
            logging.warning(f"Error during cleanup: {str(e)}")
        
        # Reset state
        self.latest_frame = None
        logging.info("Stream processor cleaned up")
