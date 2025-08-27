# YouTube Live Stream AI Object Detection

Real-time AI object detection system for YouTube live streams using Flask and TensorFlow.js.

## ✨ Features

- **Real-time Stream Processing**: Extract and process YouTube live streams using yt-dlp
- **Browser-based AI Detection**: COCO-SSD model running in TensorFlow.js
- **Live Visualization**: Canvas-based detection overlay with bounding boxes
- **Performance Monitoring**: Real-time FPS, frame count, and detection statistics
- **Modern UI**: Responsive Bootstrap 5 interface with dark mode support

## 🛠️ Tech Stack

**Backend**: Flask, OpenCV, yt-dlp, Threading  
**Frontend**: TensorFlow.js, COCO-SSD, Canvas API, Bootstrap 5  
**AI Model**: Pre-trained COCO-SSD object detection model

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Modern web browser with JavaScript enabled

### Installation & Setup
```bash
# Install dependencies (using uv - already configured in pyproject.toml)
uv sync

# Or install manually
pip install flask opencv-python yt-dlp requests werkzeug

# Start the application
python app.py
```

### Usage
1. Open `http://localhost:5000` in your browser
2. Enter a YouTube live stream URL
3. Click "Start Detection" to begin real-time analysis
4. View live object detection results with bounding boxes and confidence scores

## 📋 Project Structure

```
├── app.py                 # Flask application entry point
├── routes.py              # API endpoints and request handlers
├── stream_processor.py    # YouTube stream processing and OpenCV handling
├── templates/
│   └── index.html        # Main web interface
├── static/
│   ├── css/style.css     # Custom styles
│   └── js/
│       ├── stream.js     # Stream management and video display
│       └── detection.js  # AI model loading and object detection
└── pyproject.toml        # Python dependencies configuration
```

## 🔧 Architecture

**Data Flow**: YouTube Live Stream → yt-dlp → Stream URL → OpenCV → Video Frames → Flask API → Frontend Canvas → TensorFlow.js → AI Detection Results

### Core Components

- **StreamProcessor**: Handles YouTube URL extraction, OpenCV frame capture, and automatic reconnection
- **Flask Routes**: API endpoints for stream control (`/start_stream`, `/stop_stream`, `/video_feed`, `/stream_status`)
- **Frontend Managers**: 
  - `StreamManager`: Video display and frame fetching
  - `ObjectDetectionManager`: AI model loading and detection processing

## 🎯 Supported Streams

- ✅ Public live streams
- ✅ 24/7 continuous broadcasts  
- ✅ HD video streams (auto-scaled to 1280x720)
- ❌ Private or restricted streams

## 🛠️ Troubleshooting

### Common Issues

**Black screen or no video display**
- Check browser console for errors
- Verify YouTube URL is a valid live stream
- Ensure backend service is running on port 5000

**Video freezing on single frame**
- Browser caching issue - refresh the page
- Check network connection stability
- Verify stream is still live

**AI detection not working**
- Wait for model to fully load (check status indicator)
- Ensure TensorFlow.js is loaded properly
- Check browser compatibility (modern browsers required)

**Backend connection errors**
- Verify yt-dlp is installed and accessible
- Check if YouTube stream is publicly accessible
- Review server logs for detailed error messages

### Performance Optimization

- **Frame Rate**: ~60 FPS video display, 5 FPS AI detection
- **Resolution**: Auto-scaled to max 1280x720 for optimal performance
- **Reconnection**: Automatic retry with exponential backoff (max 3 attempts)

### Debug Commands

```bash
# Test stream processing
curl -X POST http://localhost:5000/start_stream \
  -H "Content-Type: application/json" \
  -d '{"url": "YOUR_YOUTUBE_URL"}'

# Check current stream status
curl http://localhost:5000/stream_status

# Get current video frame
curl http://localhost:5000/video_feed -o frame.jpg
```

## 📄 License

This project is licensed under the MIT License.

## 🔗 References

- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [COCO-SSD Model](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd)
- [yt-dlp Project](https://github.com/yt-dlp/yt-dlp)
- [OpenCV Python Documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
