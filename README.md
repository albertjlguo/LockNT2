# YouTube直播流AI目标检测系统

一个基于Flask和TensorFlow.js的实时YouTube直播流AI目标检测Web应用程序。

## 🎯 功能特性

- **实时视频流处理**: 使用yt-dlp提取YouTube直播流URL
- **AI目标检测**: 基于TensorFlow.js COCO-SSD模型的浏览器端检测
- **实时可视化**: Canvas绘制检测框和标签
- **流状态监控**: 实时显示FPS、帧数和检测统计
- **响应式UI**: Bootstrap 5现代化界面设计
- **暗黑模式**: 支持明暗主题切换

## 🛠️ 技术栈

### 后端 (Backend)
- **Flask**: Python Web框架
- **OpenCV**: 视频帧捕获和处理
- **yt-dlp**: YouTube视频流URL提取
- **Threading**: 多线程流处理

### 前端 (Frontend)
- **TensorFlow.js**: 浏览器端AI模型推理
- **COCO-SSD**: 预训练目标检测模型
- **Canvas API**: 视频渲染和检测结果绘制
- **Bootstrap 5**: 响应式UI框架
- **FontAwesome**: 图标库

## 📦 安装和运行

### 1. 环境要求
```bash
Python 3.8+
pip (Python包管理器)
```

### 2. 安装依赖
```bash
pip install flask opencv-python yt-dlp requests
```

### 3. 启动应用
```bash
python app.py
```

### 4. 访问应用
打开浏览器访问: `http://localhost:5000`

## 🚀 使用方法

### 基本使用流程

1. **输入YouTube直播URL**
   - 在输入框中粘贴YouTube直播链接
   - 例如: `https://www.youtube.com/watch?v=VIDEO_ID`

2. **启动检测**
   - 点击"Start Detection"按钮
   - 等待AI模型加载完成

3. **观看实时检测**
   - 视频流将显示在canvas上
   - AI检测框会实时标注识别的物体
   - 右侧面板显示检测统计信息

4. **停止检测**
   - 点击"Stop"按钮停止流处理

### 支持的YouTube直播类型
- ✅ 公开直播流
- ✅ 24/7连续直播
- ✅ 高清视频流 (自动调整至1280x720)
- ❌ 私有或受限制的直播

## 🔧 系统架构

### 数据流程
```
YouTube直播 → yt-dlp → 直播流URL → OpenCV → 视频帧 → Flask API → 前端Canvas → TensorFlow.js → AI检测结果
```

### 核心组件

#### 1. StreamProcessor (stream_processor.py)
- 负责YouTube流URL提取
- OpenCV视频帧捕获
- 帧编码和缓存管理
- 自动重连和错误处理

#### 2. Flask路由 (routes.py)
- `/start_stream`: 启动流处理
- `/stop_stream`: 停止流处理  
- `/video_feed`: 提供视频帧数据
- `/stream_status`: 流状态查询

#### 3. 前端管理器
- **StreamManager**: 视频流显示和管理
- **ObjectDetectionManager**: AI模型加载和检测

## 🐛 常见问题和解决方案

### 问题1: 黑屏或无视频显示
**症状**: Canvas显示黑色，无视频内容

**原因分析**:
- 前端帧获取逻辑问题
- Canvas初始化失败
- 后端流处理未启动

**解决方案**:
```javascript
// 修复前端帧获取缓存问题
this.frameImage = new Image(); // 每次创建新对象
const frameUrl = `./video_feed?t=${timestamp}&r=${Math.random()}`; // 双重随机参数
```

### 问题2: 视频卡顿，只显示一帧
**症状**: 视频画面静止不动，帧计数器不更新

**原因分析**:
- 浏览器缓存导致重复加载同一帧
- 帧获取频率设置不当
- Image对象复用问题

**解决方案**:
```javascript
// 优化帧获取逻辑
fetchNextFrame() {
    // 每次创建新Image对象避免缓存
    this.frameImage = new Image();
    this.frameImage.crossOrigin = 'anonymous';
    
    // 添加随机参数防缓存
    const timestamp = Date.now() + Math.random() * 1000;
    const frameUrl = `./video_feed?t=${timestamp}&r=${Math.random()}`;
    this.frameImage.src = frameUrl;
}
```

### 问题3: AI检测不工作
**症状**: 视频正常但无检测框显示

**原因分析**:
- `isModelLoaded()`方法调用错误
- AI模型未正确加载
- 检测方法调用失败

**解决方案**:
```javascript
// 修复AI检测方法调用
if (window.detectionManager && window.detectionManager.isModelLoaded) {
    // isModelLoaded是属性，不是方法
    this.performDetection();
}
```

### 问题4: 后端流处理不稳定
**症状**: 频繁出现"Failed to fetch"错误，服务崩溃

**原因分析**:
- OpenCV视频捕获连接不稳定
- 缺乏重连机制
- 错误处理不完善

**解决方案**:
```python
# 添加智能重连机制
def start_processing(self):
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries and self.is_running:
        try:
            # 设置超时保护
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
            
            # 连续失败检测
            consecutive_failures = 0
            max_consecutive_failures = 10
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        break
                # 处理成功帧...
        except Exception as e:
            retry_count += 1
            time.sleep(5)  # 延迟重试
```

### 问题5: URL路径错误
**症状**: 控制台显示404错误，帧加载失败

**原因分析**:
- Replit等云环境下的路径解析问题
- 绝对路径vs相对路径问题

**解决方案**:
```javascript
// 使用相对路径
const frameUrl = `./video_feed?t=${timestamp}&r=${Math.random()}`;
// 而不是 `/video_feed?...`
```

### 问题6: 性能优化
**症状**: 系统响应慢，CPU占用高

**解决方案**:
- 调整帧率: 30FPS → 20FPS (50ms间隔)
- 后端处理: 0.033s → 0.1s延迟
- 帧尺寸限制: 最大1280px宽度

## 📊 性能参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 前端帧率 | 20 FPS | 50ms间隔，平衡流畅度和性能 |
| 后端处理 | 10 FPS | 0.1s延迟，减少系统负载 |
| 最大分辨率 | 1280x720 | 自动缩放，保持16:9比例 |
| 重连次数 | 3次 | 失败后最多重试3次 |
| 连续失败阈值 | 10次 | 超过10次连续失败则重连 |

## 🔍 调试和监控

### 浏览器控制台日志
```javascript
// 关键日志信息
"Canvas elements found: {videoCanvas: true, detectionCanvas: true}"
"COCO-SSD model loaded successfully"
"Frame loaded: 1280 x 720"
"Frame drawn to canvas successfully"
"Triggering AI detection..."
"Drawing detections: 3" // 检测到的物体数量
```

### 后端日志监控
```python
# 关键日志信息
"Started processing stream: https://www.youtube.com/..."
"Got stream URL: https://manifest.googlevideo.com/..."
"Started video processing"
"Stream processing stopped"
```

## 🚨 故障排除

### 快速诊断清单
1. ✅ 检查YouTube URL是否为有效直播
2. ✅ 确认浏览器控制台无错误信息
3. ✅ 验证后端服务正常运行 (端口5000)
4. ✅ 检查网络连接是否稳定
5. ✅ 确认AI模型已成功加载

### 常用调试命令
```bash
# 测试后端流处理
curl -X POST http://localhost:5000/start_stream \
  -H "Content-Type: application/json" \
  -d '{"url": "YOUR_YOUTUBE_URL"}'

# 检查视频帧
curl -s http://localhost:5000/video_feed -o test_frame.jpg

# 查看流状态
curl -s http://localhost:5000/stream_status
```

## 📝 开发日志

### 主要修复历程 (2025-08-27)

1. **初始问题**: 黑屏无显示
   - 修复Canvas初始化和上下文获取

2. **帧获取问题**: 只显示一帧静止画面
   - 重构帧获取逻辑，避免浏览器缓存
   - 每次创建新Image对象

3. **AI检测错误**: `isModelLoaded is not a function`
   - 修复方法调用：`isModelLoaded()` → `isModelLoaded`

4. **后端稳定性**: 频繁崩溃和重连
   - 添加智能重连机制和超时保护
   - 增强错误处理和资源清理

5. **性能优化**: 系统卡顿
   - 调整帧率和处理频率
   - 优化错误重试机制

## 🤝 贡献指南

1. Fork本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情

## 👨‍💻 作者

**Albert Guo** - [sub.jl@icloud.com](mailto:sub.jl@icloud.com)

---

## 🔗 相关链接

- [TensorFlow.js官方文档](https://www.tensorflow.org/js)
- [COCO-SSD模型文档](https://github.com/tensorflow/tfjs-models/tree/master/coco-ssd)
- [yt-dlp项目](https://github.com/yt-dlp/yt-dlp)
- [OpenCV Python文档](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
