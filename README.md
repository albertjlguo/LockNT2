# YouTube 直播实时目标检测应用

这是一个基于 Flask 和 TensorFlow.js 的 Web 应用，能够实时捕获 YouTube 直播视频流，并在前端通过 COCO-SSD 模型进行目标检测，最终将结果可视化显示。

## 🚨 重要：关于 YouTube 身份验证

在云端服务器（如 Replit）上运行时，YouTube 可能会将请求识别为自动化机器人，并返回 `Sign in to confirm you’re not a bot` 错误，导致无法获取视频流。

**解决方案是使用 Cookies 进行身份验证。**

### 如何配置 Cookies

1.  **安装浏览器扩展**：在您的 Chrome 或 Firefox 浏览器上安装一个可以导出 Cookies 的扩展，例如 **"Get cookies.txt"** 或 **"Cookie-Editor"**。
2.  **登录并导出**：
    *   在浏览器中登录您的 Google/YouTube 账号。
    *   访问 `https://www.youtube.com`。
    *   点击扩展图标，选择导出 `cookies.txt` 格式（Netscape format）。
3.  **创建 `cookies.txt` 文件**：
    *   在本项目根目录下，创建一个名为 `cookies.txt` 的文件。
    *   将刚才复制的 Cookies 内容**完整地**粘贴到此文件中并保存。

程序会自动检测并使用此文件进行身份验证。由于 `.gitignore` 文件已将 `cookies.txt` 排除，您的机密信息不会被提交到 Git 仓库。

**注意：Cookies 可能会过期，如果再次遇到此错误，请重复上述步骤更新 `cookies.txt` 文件。**

## 🎯 功能特性

- **后端视频流代理**：通过 Flask 和 `yt-dlp`，将 YouTube 直播流转换为可供前端消费的 MJPEG 流。
- **AI 目标检测**：在浏览器端使用 TensorFlow.js 和 COCO-SSD 模型进行实时检测。
- **Cookies 认证**：支持使用 `cookies.txt` 文件绕过 YouTube 的机器人检测。
- **健壮的流处理**：具备自动重连和超时机制，确保长时间稳定运行。
- **实时可视化**：在 Canvas 上绘制检测边界框、标签和置信度。
- **状态监控**：实时显示 FPS、检测到的对象列表和历史记录。
- **响应式 UI**：基于 Bootstrap 5 构建，适配不同设备。

## ✨ 交互式多目标追踪 (Interactive Multi-Object Tracking)

本应用现已升级为支持交互式**多目标**追踪。

- **添加追踪目标**: 在视频画面上**点击**任何一个**未被追踪**的物体，系统会为其分配一个唯一的ID和颜色，并开始持续追踪。
- **移除单个目标**: **再次点击**一个**已被追踪**的物体，即可将其从追踪列表中移除。
- **清空所有目标**: 点击视频画面中的任意**空白区域**，可以一次性清空所有正在追踪的目标。
- **视觉反馈**:
  - 每个被追踪的目标都会有一个**专属颜色**的边界框和运动轨迹，方便区分。
  - 当目标被短暂遮挡时，系统会尝试预测其位置，此时其边界框会变为**橙色**。

## 🛠️ 技术栈

- **后端**: Flask, OpenCV, yt-dlp
- **前端**: JavaScript, TensorFlow.js, COCO-SSD, Bootstrap 5, Canvas API

## 📦 安装与运行

1.  **克隆仓库**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    # 如果没有 requirements.txt，请手动安装:
    # pip install flask opencv-python yt-dlp requests
    ```
3.  **配置 Cookies** (请参考上文的“重要”部分)
    - 创建并配置 `cookies.txt` 文件。

4.  **启动应用**
    ```bash
    python app.py
    ```
5.  **访问应用**
    - 打开浏览器并访问 `http://127.0.0.1:5000`。

## 🤝 贡献

欢迎通过 Pull Request 提交改进和修复。

## 👨‍💻 作者

**Albert Guo** - [sub.jl@icloud.com](mailto:sub.jl@icloud.com)
