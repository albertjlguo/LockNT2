#!/usr/bin/env python3
"""
YOLOv11 to TensorFlow.js Converter
YOLOv11 转 TensorFlow.js 转换器
Converts YOLOv11 PyTorch model to TensorFlow.js format for web deployment
将YOLOv11 PyTorch模型转换为TensorFlow.js格式以供Web部署
"""

import os
import sys

def convert_yolo11_to_tfjs():
    """
    Convert YOLOv11 model to TensorFlow.js format
    将YOLOv11模型转换为TensorFlow.js格式
    """
    print("=" * 60)
    print("YOLOv11 to TensorFlow.js Converter")
    print("YOLOv11 转 TensorFlow.js 转换器")
    print("=" * 60)
    
    try:
        # Check if ultralytics is installed
        # 检查是否安装了ultralytics
        try:
            from ultralytics import YOLO
            print("✓ Ultralytics YOLO library found / 找到Ultralytics YOLO库")
        except ImportError:
            print("Installing Ultralytics... / 正在安装Ultralytics...")
            os.system("pip install ultralytics")
            from ultralytics import YOLO
        
        # Create models directory if it doesn't exist
        # 如果模型目录不存在则创建
        models_dir = "static/models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"✓ Created models directory: {models_dir}")
        
        # Download and load YOLOv11n model (nano - best for web)
        # 下载并加载YOLOv11n模型（nano - 最适合Web）
        print("\nDownloading YOLOv11n model... / 正在下载YOLOv11n模型...")
        model = YOLO("yolo11n.pt")  # This will auto-download if not present / 如果不存在会自动下载
        print("✓ YOLOv11n model loaded / YOLOv11n模型已加载")
        
        # Export to TensorFlow.js format
        # 导出为TensorFlow.js格式
        print("\nExporting to TensorFlow.js format... / 正在导出为TensorFlow.js格式...")
        print("This may take a few minutes... / 这可能需要几分钟...")
        
        # Export with optimized settings for web
        # 使用Web优化设置导出
        export_path = model.export(
            format="tfjs",
            imgsz=640,  # Input size / 输入尺寸
            half=False,  # Full precision for better accuracy / 全精度以获得更好的准确性
            int8=False,  # No int8 quantization for better accuracy / 不使用int8量化以获得更好的准确性
            nms=True,  # Include NMS in model / 在模型中包含NMS
            batch=1,  # Batch size for web / Web批次大小
        )
        
        print(f"✓ Model exported to: {export_path}")
        
        # Move the exported model to static/models directory
        # 将导出的模型移动到static/models目录
        import shutil
        dest_path = os.path.join(models_dir, "yolo11n_web_model")
        
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        
        # The export creates a folder with the model name
        # 导出会创建一个带有模型名称的文件夹
        export_folder = "yolo11n_web_model"
        if os.path.exists(export_folder):
            shutil.move(export_folder, dest_path)
            print(f"✓ Model moved to: {dest_path}")
        
        print("\n" + "=" * 60)
        print("✅ Conversion Complete! / 转换完成！")
        print("=" * 60)
        print("\nModel Information / 模型信息:")
        print(f"- Model: YOLOv11n (Nano)")
        print(f"- Input Size: 640x640")
        print(f"- Output Format: TensorFlow.js")
        print(f"- Location: {dest_path}/model.json")
        print("\nNext Steps / 下一步:")
        print("1. The model is ready to use in your web application")
        print("   模型已准备好在您的Web应用程序中使用")
        print("2. Update your HTML to load the new YOLOv11 detection script")
        print("   更新您的HTML以加载新的YOLOv11检测脚本")
        print("3. The detection manager will automatically load this model")
        print("   检测管理器将自动加载此模型")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during conversion / 转换过程中出错: {str(e)}")
        print("\nTroubleshooting / 故障排除:")
        print("1. Make sure you have Python 3.8+ installed")
        print("   确保您安装了Python 3.8+")
        print("2. Try installing dependencies manually:")
        print("   尝试手动安装依赖项:")
        print("   pip install ultralytics tensorflow tensorflowjs")
        print("3. Check your internet connection for model download")
        print("   检查您的互联网连接以下载模型")
        return False

def create_fallback_config():
    """
    Create a fallback configuration file if conversion fails
    如果转换失败，创建回退配置文件
    """
    config = {
        "model": "coco-ssd",
        "fallback": True,
        "message": "Using COCO-SSD as fallback. YOLOv11 conversion required."
    }
    
    import json
    models_dir = "static/models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    config_path = os.path.join(models_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n📝 Created fallback configuration: {config_path}")
    print("The application will use COCO-SSD until YOLOv11 is available")
    print("应用程序将使用COCO-SSD，直到YOLOv11可用")

if __name__ == "__main__":
    success = convert_yolo11_to_tfjs()
    if not success:
        create_fallback_config()
    
    sys.exit(0 if success else 1)
