# YouTube Live Stream AI Object Detection

一个基于Python Flask和TensorFlow.js的实时YouTube直播流AI目标检测系统。

## 功能特性

- **实时视频流处理**: 使用yt-dlp提取YouTube直播流URL，OpenCV处理视频帧
- **AI目标检测**: 前端使用TensorFlow.js的COCO-SSD模型进行实时目标检测
- **响应式界面**: Bootstrap 5构建的现代化深色主题UI
- **高性能渲染**: 60 FPS视频播放，5 FPS AI检测，平衡性能与准确性
- **错误恢复**: 完善的重连机制和错误处理，确保服务稳定性
- **生产环境支持**: 支持cookies认证绕过YouTube机器人验证

## 技术栈

**后端**:
- Python 3.11+
- Flask (Web框架)
- OpenCV (视频处理)
- yt-dlp (YouTube流提取)

**前端**:
- HTML5 Canvas (视频渲染)
- TensorFlow.js (AI模型)
- COCO-SSD (目标检测模型)
- Bootstrap 5 (UI框架)
- FontAwesome (图标)

## 快速开始

### 环境要求

- Python 3.11+
- yt-dlp CLI工具
- 现代浏览器(支持WebGL)

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd youtube-live-ai-detection
```

2. **安装Python依赖**
```bash
pip install -r requirements.txt
```

3. **安装yt-dlp**
```bash
# Ubuntu/Debian
sudo apt install yt-dlp

# macOS
brew install yt-dlp

# 或使用pip
pip install yt-dlp
```

4. **运行应用**
```bash
python app.py
```

5. **访问应用**
打开浏览器访问 `http://localhost:5000`

### 使用方法

1. 在输入框中粘贴YouTube直播流URL
2. 点击"开始检测"按钮
3. 等待AI模型加载完成
4. 观看实时视频流和AI检测结果

## 🍪 生产环境Cookies支持

### 为什么需要Cookies？

生产环境下，YouTube可能会显示"Sign in to confirm you're not a bot"错误，导致无法提取直播流URL。通过提供浏览器cookies可以绕过这个限制。

### 设置Cookies

1. **生成cookies模板**
```bash
python add_cookies_support.py
```

2. **获取YouTube Cookies**
   - 打开浏览器，访问 https://youtube.com
   - 完成任何必要的登录和验证
   - 打开开发者工具 (F12)
   - 切换到 Application/应用程序 标签
   - 左侧选择 Storage > Cookies > https://www.youtube.com
   - 复制以下重要cookies的值：
     - `VISITOR_INFO1_LIVE`
     - `YSC`
     - `PREF`
     - `CONSENT` (如果有)

3. **填写cookies.txt文件**
将cookies按以下格式填入`cookies.txt`：
```
.youtube.com	TRUE	/	TRUE	1234567890	VISITOR_INFO1_LIVE	你的值
.youtube.com	TRUE	/	TRUE	1234567890	YSC	你的值
.youtube.com	TRUE	/	TRUE	1234567890	PREF	你的值
```

**注意事项**：
- 使用TAB分隔符，不是空格
- 过期时间可以设置为未来的时间戳
- 保护好cookies文件，不要提交到版本控制
- 完成后重启应用即可使用cookies绕过验证

### Cookies工作原理

- **无cookies时**: 系统记录警告但继续尝试提取（可能失败）
- **有cookies时**: yt-dlp使用cookies进行认证，绕过机器人验证
- **自动检测**: 系统自动检测`cookies.txt`文件并应用到所有yt-dlp命令

## 系统架构

### 数据流程

```
YouTube直播 → yt-dlp(+cookies) → 直播流URL → OpenCV → 视频帧 → Flask API → 前端Canvas → TensorFlow.js → AI检测结果
```

### 核心组件

1. **StreamProcessor** (`stream_processor.py`)
   - YouTube流URL提取（支持cookies）
   - 视频帧捕获和处理
   - 多线程流处理
   - 双重提取策略（720p→480p）

2. **Flask Routes** (`routes.py`)
   - `/start_stream`: 启动流处理
   - `/stop_stream`: 停止流处理
   - `/video_feed`: 获取视频帧
   - `/stream_status`: 查询流状态

3. **StreamManager** (`static/js/stream.js`)
   - 前端流管理
   - 视频帧获取和渲染
   - 错误处理和重试机制

4. **ObjectDetectionManager** (`static/js/detection.js`)
   - TensorFlow.js模型加载
   - COCO-SSD目标检测
   - 检测结果渲染

5. **Cookies Support** (`add_cookies_support.py`)
   - Cookies模板生成
   - 格式验证
   - 设置指导

## 性能参数

- **视频帧率**: 60 FPS (可调节)
- **AI检测频率**: 5 FPS (平衡性能)
- **视频分辨率**: 最大1280px宽度
- **JPEG质量**: 85%
- **重连机制**: 最多3次重试，5-10秒间隔
- **提取策略**: 720p失败时自动降级到480p

## 常见问题与解决方案

### 问题1: 生产环境503错误（最常见）

**症状**: 前端显示黑屏，控制台显示503 Service Unavailable
**原因**: YouTube机器人验证阻止yt-dlp提取流URL
**解决方案**:
1. **使用Cookies（推荐）**:
   ```bash
   python add_cookies_support.py
   # 按照提示设置cookies.txt文件
   ```
2. **检查服务器日志**:
   - 查看详细的yt-dlp错误输出
   - 确认是否为"Sign in to confirm you're not a bot"错误
3. **验证URL**:
   - 确保YouTube URL是有效的直播流
   - 测试URL在浏览器中是否可访问

### 问题2: Cookies设置错误

**症状**: 设置cookies后仍然出现503错误
**原因**: Cookies格式错误或已过期
**解决方案**:
1. **验证格式**:
   ```bash
   python add_cookies_support.py  # 检查格式验证结果
   ```
2. **重新获取Cookies**:
   - 清除浏览器缓存
   - 重新访问YouTube并获取新的cookies
   - 确保使用TAB分隔符而非空格
3. **检查权限**:
   - 确保cookies.txt文件可读
   - 检查文件路径是否正确

### 问题3: AI检测不工作

**症状**: 视频正常播放但无检测框显示
**原因**: TensorFlow.js模型加载失败或检测逻辑错误
**解决方案**:
1. 检查浏览器控制台是否有JavaScript错误
2. 确认网络可以访问TensorFlow.js CDN
3. 刷新页面重新加载模型
4. 检查WebGL支持情况

### 问题4: 视频播放卡顿

**症状**: 视频帧率低或播放不流畅
**原因**: 网络带宽不足或系统性能限制
**解决方案**:
1. 降低视频质量设置（系统会自动降级到480p）
2. 调整帧率参数（修改setTimeout间隔）
3. 检查系统CPU和内存使用情况
4. 优化网络连接

### 问题5: yt-dlp版本兼容性

**症状**: 开发环境正常，生产环境失败
**原因**: yt-dlp版本差异或系统环境不同
**解决方案**:
1. 统一yt-dlp版本
2. 检查系统依赖
3. 使用Docker确保环境一致性
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
