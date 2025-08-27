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

- **后端视频流代理**：通过 Flask 和 `yt-dlp` 将 YouTube 直播流转换为可供前端消费的 MJPEG 流。
- **AI 目标检测**：在浏览器端使用 TensorFlow.js 与 COCO-SSD 模型进行实时检测。
- **Cookies 认证**：通过 `cookies.txt` 绕过 YouTube 反爬验证（见上文）。
- **健壮的流处理**：自动重连与超时控制，适配长时间运行。
- **最小化前端可视化**：仅在用户点击后显示已锁定目标的追踪框与 ID 标签；不自动显示检测框；隐藏轨迹折线。
- **全新侧边栏**：新增 "Object Tracking" 面板，仅展示已锁定目标列表（`templates/index.html` 中的 `#trackingList`）。
- **交互快捷操作**：支持点击锁定/解锁、右键清空、键盘快捷键；状态面板显示 FPS 与帧数。
- **响应式 UI**：基于 Bootstrap 5 构建，适配不同设备。

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

## 🧭 使用说明（前端交互）

- **开始检测**：在输入框粘贴 YouTube 直播地址，点击 “Start Detection”。
- **点击上框（锁定）**：首次点击画面中的目标位置，系统会执行一次性检测并为最近目标创建追踪框；之后仅显示已锁定目标的框与 ID。
- **切换锁定/解锁**：再次点击已存在的追踪框可在锁定与解锁之间切换。
- **右键清空**：在画布上右键可清空所有目标与叠加层（`static/js/stream.js` 中绑定 `contextmenu`）。
- **键盘快捷键**（窗口激活时有效）：
  - `L`：解锁全部已追踪目标（保留轨迹实体，但取消锁定）。
  - `C`：清空全部目标（移除所有追踪并清空叠加层）。
  - `A`：切换 Auto-create（默认关闭）。关闭时仅在手动点击后创建目标。
- **轨迹显示**：轨迹折线已隐藏，仅显示边框及标签，保持界面简洁。
- **侧边栏**：右侧 "Object Tracking" 列表仅展示已锁定目标的 ID 与状态，无自动检测结果与历史记录面板。

提示：模型加载与检测在前端进行；为提升性能，检测存在节流机制，间隔帧将只做轨迹预测而不绘制检测框（保持“仅点击后显示”原则）。

## 🤝 贡献

欢迎通过 Pull Request 提交改进和修复。

## 👨‍💻 作者

**Albert Guo** - [sub.jl@icloud.com](mailto:sub.jl@icloud.com)
