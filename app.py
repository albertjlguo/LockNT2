import os
import logging
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
import requests
import threading
import time
from urllib.parse import urlparse, parse_qs
import re
from stream_processor import StreamProcessor
from yolo_detector import yolo_detector, initialize_yolo
import base64
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
# app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # Comment out if ProxyFix not available

# Import routes
from routes import *

# Initialize YOLOv11 detector on startup
# 启动时初始化YOLOv11检测器
print("Initializing YOLOv11 detector...")
if initialize_yolo():
    print("✓ YOLOv11 detector initialized successfully")
else:
    print("⚠ YOLOv11 detector initialization failed, will use COCO-SSD fallback")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
