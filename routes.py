import logging
import json
import os
from flask import render_template, request, jsonify, Response
from app import app
from stream_processor import StreamProcessor
from fallback_sources import FallbackVideoSource
import threading
import time

# Global stream processor instance
stream_processor = None
fallback_source = None
processing_thread = None
is_processing = False
using_fallback = False
failed_frame_count = 0

# Environment detection
def is_production_environment():
    """Detect if running in production environment."""
    # Check for production indicators
    production_indicators = [
        os.getenv('FLASK_ENV') == 'production',
        os.getenv('ENVIRONMENT') == 'production', 
        'locknt2.theiresearch.com' in os.getenv('SERVER_NAME', ''),
        os.path.exists('/etc/production'),  # Custom production marker
        'replit' in os.getcwd().lower()  # Replit production deployment
    ]
    return any(production_indicators)

@app.route('/')
def index():
    """Main page with object detection interface."""
    return render_template('index.html')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Start processing a YouTube stream."""
    global stream_processor, processing_thread, is_processing
    
    try:
        data = request.get_json()
        youtube_url = data.get('url')
        
        if not youtube_url:
            return jsonify({'error': 'No URL provided'}), 400
        
        # Stop any existing stream
        stop_stream_processing()
        
        # Create new stream processor
        stream_processor = StreamProcessor(youtube_url)
        
        # Initialize fallback source only in production
        global fallback_source
        if is_production_environment():
            fallback_source = FallbackVideoSource()
            logging.info("Production environment detected - fallback source initialized")
        else:
            logging.info("Development environment detected - no fallback source")
        
        # In production, skip strict validation to avoid bot detection issues
        # Just log a warning if validation fails but continue anyway
        try:
            if not stream_processor.validate_stream():
                logging.warning("Stream validation failed, but continuing in production mode")
        except Exception as validation_error:
            logging.warning(f"Stream validation error (continuing anyway): {validation_error}")
        
        # Start processing in background thread
        processing_thread = threading.Thread(target=stream_processor.start_processing)
        processing_thread.daemon = True
        processing_thread.start()
        
        is_processing = True
        logging.info(f"Started processing stream: {youtube_url}")
        
        return jsonify({'message': 'Stream processing started successfully'})
        
    except Exception as e:
        logging.error(f"Error starting stream: {str(e)}")
        return jsonify({'error': f'Failed to start stream: {str(e)}'}), 500

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
    """Get current video frame as JPEG image with production fallback."""
    global stream_processor, fallback_source, using_fallback, failed_frame_count
    
    if not is_processing:
        logging.warning("Video feed requested but no active stream")
        return Response("No active stream", status=404)
    
    # Try to get frame from primary stream
    frame = None
    if stream_processor and not using_fallback:
        frame = stream_processor.get_latest_frame()
    
    # Production environment fallback logic
    if is_production_environment():
        if frame is None:
            failed_frame_count += 1
            logging.warning(f"No frame from primary stream (failure #{failed_frame_count})")
            
            # Switch to fallback after 5 consecutive failures
            if failed_frame_count >= 5 and fallback_source and not using_fallback:
                logging.info("Switching to fallback source due to repeated primary stream failures")
                fallback_source.start_test_pattern_stream()
                using_fallback = True
        else:
            # Reset failure count on successful frame
            failed_frame_count = 0
        
        # Get frame from fallback if using fallback
        if using_fallback and fallback_source:
            fallback_frame = fallback_source.get_latest_frame()
            if fallback_frame:
                frame = fallback_frame
    
    # Return frame or error
    if frame is not None:
        return Response(frame, mimetype='image/jpeg')
    else:
        if is_production_environment():
            # In production, start emergency fallback if all else fails
            if not using_fallback and fallback_source:
                logging.info("Starting emergency fallback source")
                fallback_source.start_test_pattern_stream()
                using_fallback = True
                emergency_frame = fallback_source.get_latest_frame()
                if emergency_frame:
                    return Response(emergency_frame, mimetype='image/jpeg')
        
        # Log the reason for 503 error
        env_type = "production" if is_production_environment() else "development"
        logging.error(f"No frame available in {env_type} environment - Stream URL extraction may have failed. Stream processor status: running={getattr(stream_processor, 'is_running', False)}")
        return Response("No frame available - stream processing failed", status=503)

def stop_stream_processing():
    """Helper function to stop stream processing."""
    global stream_processor, fallback_source, processing_thread, is_processing, using_fallback, failed_frame_count
    
    is_processing = False
    using_fallback = False
    failed_frame_count = 0
    
    if stream_processor:
        stream_processor.stop()
        stream_processor = None
    
    if fallback_source:
        fallback_source.stop()
        fallback_source = None
    
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=2)
