import logging
import json
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
        
        # Initialize fallback source
        global fallback_source
        fallback_source = FallbackVideoSource()
        
        # In production, skip strict validation to avoid bot detection issues
        # Just log a warning if validation fails but continue anyway
        try:
            if not stream_processor.validate_stream():
                logging.warning("Stream validation failed, but continuing in production mode")
        except Exception as validation_error:
            logging.warning(f"Stream validation error (continuing anyway): {validation_error}")
        
        # Start processing in background thread with fallback monitoring
        def start_with_fallback():
            global using_fallback
            try:
                # Try primary stream first
                stream_processor.start_processing()
            except Exception as e:
                logging.error(f"Primary stream failed, starting fallback: {str(e)}")
                # Start fallback if primary fails
                fallback_source.start_test_pattern_stream()
                using_fallback = True
        
        processing_thread = threading.Thread(target=start_with_fallback)
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
    """Get current video frame as JPEG image."""
    global stream_processor, fallback_source, using_fallback
    
    if not is_processing:
        logging.warning("Video feed requested but no active stream")
        return Response("No active stream", status=404)
    
    frame = None
    
    # Try to get frame from primary stream first
    if stream_processor and not using_fallback:
        frame = stream_processor.get_latest_frame()
        
        # If no frame from primary, check if we should switch to fallback
        if frame is None:
            logging.warning("No frame from primary stream, checking fallback...")
            if fallback_source and not using_fallback:
                logging.info("Starting fallback source due to primary stream failure")
                fallback_source.start_test_pattern_stream()
                using_fallback = True
    
    # Get frame from fallback if using fallback or primary failed
    if using_fallback and fallback_source:
        frame = fallback_source.get_latest_frame()
    
    if frame is not None:
        return Response(frame, mimetype='image/jpeg')
    else:
        # Start fallback as last resort
        if not using_fallback and fallback_source:
            logging.info("Starting emergency fallback source")
            fallback_source.start_test_pattern_stream()
            using_fallback = True
            frame = fallback_source.get_latest_frame()
            if frame:
                return Response(frame, mimetype='image/jpeg')
        
        logging.error(f"No frame available from any source")
        return Response("No frame available - all sources failed", status=503)

def stop_stream_processing():
    """Helper function to stop stream processing."""
    global stream_processor, fallback_source, processing_thread, is_processing, using_fallback
    
    is_processing = False
    using_fallback = False
    
    if stream_processor:
        stream_processor.stop()
        stream_processor = None
    
    if fallback_source:
        fallback_source.stop()
        fallback_source = None
    
    if processing_thread and processing_thread.is_alive():
        processing_thread.join(timeout=2)
