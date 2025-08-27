"""
Fallback video sources for testing and backup when YouTube streams fail.
"""

import cv2
import numpy as np
import logging
import time
import threading

class FallbackVideoSource:
    """Provides fallback video sources when primary stream fails."""
    
    def __init__(self):
        self.current_source = None
        self.is_running = False
        self.latest_frame = None
        self.lock = threading.Lock()
        
    def create_test_pattern(self, width=640, height=480):
        """Create a test pattern with timestamp."""
        # Create a colorful test pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create gradient background
        for y in range(height):
            for x in range(width):
                frame[y, x] = [
                    int(255 * (x / width)),  # Red gradient
                    int(255 * (y / height)),  # Green gradient
                    128  # Blue constant
                ]
        
        # Add timestamp text
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cv2.putText(frame, f"TEST PATTERN - {timestamp}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "YouTube Stream Unavailable", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Using Fallback Source", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add moving element
        t = int(time.time() * 2) % width
        cv2.circle(frame, (t, height//2), 20, (255, 255, 255), -1)
        
        return frame
    
    def create_webcam_source(self, device_id=0):
        """Try to create a webcam source as fallback."""
        try:
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                logging.info(f"Webcam source {device_id} available as fallback")
                return cap
            else:
                cap.release()
                return None
        except Exception as e:
            logging.warning(f"Failed to open webcam {device_id}: {str(e)}")
            return None
    
    def start_test_pattern_stream(self):
        """Start a test pattern stream as fallback."""
        self.is_running = True
        logging.info("Starting test pattern fallback stream")
        
        def generate_frames():
            while self.is_running:
                try:
                    frame = self.create_test_pattern()
                    
                    # Encode as JPEG
                    success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if success:
                        with self.lock:
                            self.latest_frame = buffer.tobytes()
                    
                    time.sleep(0.033)  # ~30 FPS
                except Exception as e:
                    logging.error(f"Error generating test pattern: {str(e)}")
                    time.sleep(1)
        
        thread = threading.Thread(target=generate_frames)
        thread.daemon = True
        thread.start()
        return thread
    
    def start_webcam_stream(self, device_id=0):
        """Start webcam stream as fallback."""
        cap = self.create_webcam_source(device_id)
        if not cap:
            return None
            
        self.is_running = True
        self.current_source = cap
        logging.info(f"Starting webcam fallback stream from device {device_id}")
        
        def capture_frames():
            while self.is_running and cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Add fallback indicator
                        cv2.putText(frame, "FALLBACK: Webcam Feed", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Encode as JPEG
                        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if success:
                            with self.lock:
                                self.latest_frame = buffer.tobytes()
                    
                    time.sleep(0.033)  # ~30 FPS
                except Exception as e:
                    logging.error(f"Error capturing webcam frame: {str(e)}")
                    time.sleep(1)
            
            cap.release()
        
        thread = threading.Thread(target=capture_frames)
        thread.daemon = True
        thread.start()
        return thread
    
    def get_latest_frame(self):
        """Get the latest frame from fallback source."""
        with self.lock:
            return self.latest_frame
    
    def stop(self):
        """Stop the fallback source."""
        self.is_running = False
        if self.current_source:
            try:
                self.current_source.release()
            except:
                pass
            self.current_source = None
        logging.info("Fallback source stopped")

class EnhancedStreamProcessor:
    """Enhanced stream processor with fallback support."""
    
    def __init__(self, youtube_url, enable_fallback=True):
        self.youtube_url = youtube_url
        self.enable_fallback = enable_fallback
        self.primary_processor = None
        self.fallback_source = None
        self.is_using_fallback = False
        self.latest_frame = None
        self.lock = threading.Lock()
        
    def start_with_fallback(self):
        """Start processing with fallback support."""
        from stream_processor import StreamProcessor
        
        # Try primary YouTube stream first
        try:
            self.primary_processor = StreamProcessor(self.youtube_url)
            
            # Start in a separate thread
            def run_primary():
                try:
                    self.primary_processor.start_processing()
                except Exception as e:
                    logging.error(f"Primary stream failed: {str(e)}")
                    if self.enable_fallback:
                        self._start_fallback()
            
            primary_thread = threading.Thread(target=run_primary)
            primary_thread.daemon = True
            primary_thread.start()
            
            # Monitor primary stream health
            self._monitor_primary_stream()
            
        except Exception as e:
            logging.error(f"Failed to start primary stream: {str(e)}")
            if self.enable_fallback:
                self._start_fallback()
    
    def _monitor_primary_stream(self):
        """Monitor primary stream and switch to fallback if needed."""
        no_frame_count = 0
        max_no_frame_count = 10  # 10 seconds without frames
        
        while True:
            try:
                if self.primary_processor:
                    frame = self.primary_processor.get_latest_frame()
                    if frame:
                        no_frame_count = 0
                        with self.lock:
                            self.latest_frame = frame
                            self.is_using_fallback = False
                    else:
                        no_frame_count += 1
                        if no_frame_count >= max_no_frame_count and self.enable_fallback:
                            logging.warning("Primary stream appears dead, switching to fallback")
                            self._start_fallback()
                            break
                
                time.sleep(1)
            except Exception as e:
                logging.error(f"Error monitoring primary stream: {str(e)}")
                time.sleep(5)
    
    def _start_fallback(self):
        """Start fallback video source."""
        if self.is_using_fallback:
            return
            
        logging.info("Starting fallback video source")
        self.fallback_source = FallbackVideoSource()
        
        # Try webcam first, then test pattern
        fallback_thread = self.fallback_source.start_webcam_stream()
        if not fallback_thread:
            fallback_thread = self.fallback_source.start_test_pattern_stream()
        
        self.is_using_fallback = True
        
        # Monitor fallback source
        def monitor_fallback():
            while self.is_using_fallback and self.fallback_source:
                try:
                    frame = self.fallback_source.get_latest_frame()
                    if frame:
                        with self.lock:
                            self.latest_frame = frame
                    time.sleep(0.1)
                except Exception as e:
                    logging.error(f"Error monitoring fallback: {str(e)}")
                    time.sleep(1)
        
        monitor_thread = threading.Thread(target=monitor_fallback)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def get_latest_frame(self):
        """Get the latest frame from active source."""
        with self.lock:
            return self.latest_frame
    
    def is_fallback_active(self):
        """Check if fallback source is active."""
        return self.is_using_fallback
    
    def stop(self):
        """Stop all sources."""
        if self.primary_processor:
            self.primary_processor.stop()
        if self.fallback_source:
            self.fallback_source.stop()
        logging.info("Enhanced stream processor stopped")
