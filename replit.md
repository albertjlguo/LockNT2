# Overview

A real-time object detection web application that analyzes YouTube live streams using AI-powered computer vision. The system processes video streams from YouTube URLs and performs object detection using TensorFlow.js with the COCO-SSD model, providing live visual feedback and analytics on detected objects.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Single Page Application (SPA)**: Built with vanilla HTML, CSS, and JavaScript for real-time interaction
- **Canvas-based Rendering**: Dual canvas system for video display and detection overlay visualization
- **Real-time UI Updates**: Live statistics and detection feedback using DOM manipulation
- **Responsive Design**: Bootstrap 5 framework with custom CSS variables for theming
- **Dark/Light Mode**: Theme switching capability with CSS custom properties

## Backend Architecture
- **Flask Web Framework**: Lightweight Python web server handling HTTP requests and API endpoints
- **Modular Route Structure**: Separated route handlers in dedicated files for maintainability
- **Background Processing**: Threading-based stream processing to prevent blocking main application
- **Stream Validation**: YouTube URL validation and stream accessibility checking before processing

## AI/ML Processing
- **Client-side Object Detection**: TensorFlow.js with COCO-SSD model running in browser
- **Real-time Analysis**: Continuous frame processing for object detection without server load
- **Detection Statistics**: Object counting, tracking, and analytics generation
- **Performance Optimization**: Efficient canvas rendering and detection intervals

## Video Stream Processing
- **YouTube Stream Integration**: yt-dlp tool for extracting and validating YouTube stream URLs
- **OpenCV Integration**: Computer vision processing for frame extraction and manipulation
- **Multi-threaded Processing**: Background threads for stream processing to maintain UI responsiveness
- **Stream State Management**: Global state tracking for active streams and processing status

## Session Management
- **Flask Sessions**: Server-side session handling with configurable secret keys
- **Environment Configuration**: Environment-based configuration for deployment flexibility
- **Proxy Support**: ProxyFix middleware for proper header handling in production environments

# External Dependencies

## Core Frameworks
- **Flask**: Python web framework for backend API and routing
- **Bootstrap 5**: Frontend CSS framework for responsive UI components
- **Font Awesome**: Icon library for user interface elements

## AI/ML Libraries
- **TensorFlow.js**: Browser-based machine learning framework
- **COCO-SSD Model**: Pre-trained object detection model for real-time inference

## Video Processing
- **OpenCV (cv2)**: Computer vision library for video frame processing
- **yt-dlp**: Command-line tool for YouTube video/stream URL extraction and validation

## Development Tools
- **Werkzeug**: WSGI utilities including ProxyFix middleware for Flask applications

## Browser APIs
- **Canvas API**: For video rendering and detection overlay visualization
- **Web Workers**: Potential for background processing optimization