# PhotoPlay — 扫照片，播视频

以照片为 key、视频为 value 的 AR 识别播放系统（个人验证项目）。

手机浏览器打开页面即进入相机画面，**无需任何操作**：系统自动识别画面中的注册照片，命中后关联视频以透视变换精确叠加在照片上播放，跟随照片移动——如同照片"活了过来"。照片移出画面或视频播完，自动回到识别状态。

## 效果演示

对准一张照片（屏幕显示或实体打印件均可）：

```
[ 相机画面 ]  →  [ 自动识别（~300ms) ]  →  [ 视频叠加在照片位置播放 ]
                                            移动手机，视频实时跟随
```

## 工作原理

```
启动：三路并行 —— 摄像头授权 / 特征库下载 / Worker 内加载 OpenCV.js (WASM)
      → GET /api/features 拉取二进制特征库（注册时离线预计算，ETag 协商缓存）
      → Worker 内直接反序列化构造合并特征索引

识别：Web Worker 后台线程每 300ms 静默截帧（限宽 640px）
      → ORB 特征提取 → 对全库合并描述子一次 knnMatch + Lowe 比率过滤
      → 按归属照片分票，findHomography(RANSAC) 几何校验，内点 ≥ 10 判定命中

追踪：命中后 Worker 内 LK 光流逐帧追踪
      → 每帧由匹配点增量更新单应性矩阵 H_new = T · H_old（归一化防漂移）
      → 连续 6 帧追踪失败或视频播完，回到识别状态

渲染：主线程 WebGL 透视绘制视频帧到照片四角位置（无滤波，粘贴式跟随）
      视频真实首帧呈现前不绘制，避免加载期黑块
```

后端为纯静态服务：二进制特征库下发、视频流（Range + 缓存头），不参与实时识别——识别与追踪全部在前端本地完成。

## 快速开始

```powershell
# 1. 安装依赖（Python 3.10+）
pip install -r requirements.txt

# 2. 下载 OpenCV.js（保存到 web/vendor/opencv.js，并预压缩 .gz 可显著加速公网加载）
curl -L -o web/vendor/opencv.js https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js

# 3. 放入素材（同名关联：照片 A.jpg ↔ 视频 A.mp4）
#    data/photos/A.jpg    注册用高清原图
#    data/videos/A.mp4    关联视频

# 4. 注册建库（提取三尺度 ORB 特征；已装 ffmpeg 时自动转码 faststart 优化起播）
python tools/register.py

# 5. 启动服务
uvicorn server.main:app --host 0.0.0.0 --port 8000

# 6. 电脑浏览器验证
#    http://localhost:8000
```

注册新照片后无需重启服务：`POST /api/reload` 热加载。

## 手机访问（摄像头需 HTTPS）

`getUserMedia` 要求安全上下文，手机访问需内网穿透提供 HTTPS 域名：

```powershell
# 以 cpolar 为例（具体命令见 comand.md）
cpolar http 8000
# 得到 https://xxxx.cpolar.top 后，手机浏览器/微信内打开
```

加载完成提示"对准照片，自动识别"即可使用。

## 接口一览

| 接口 | 说明 |
|---|---|
| `GET /` | AR H5 页面（自动识别 + 透视叠加） |
| `GET /api/features` | 二进制识别特征库（内存缓存 + ETag + 预压缩 gzip） |
| `GET /videos/{id}` | 关联视频（Range 流式 + Cache-Control，iOS 兼容） |
| `POST /api/match` | 服务端识别接口（备用降级通道，前端默认不调用） |
| `POST /api/reload` | 热加载特征库与注册表 |
| `GET /api/health` | 健康检查 |

## 目录结构

```
server/main.py       FastAPI：特征库/视频静态下发 + 备用识别接口
server/matcher.py    ORB 匹配器（服务端识别用；特征存取供注册工具调用）
server/registry.py   registry.json 读写封装
tools/register.py    建库 CLI（三尺度特征提取 + ffmpeg 视频转码优化）
data/photos/         注册用高清原图（用户放入，git 忽略）
data/videos/         关联视频（用户放入，git 忽略；.optimized/ 为转码标记）
data/features/       ORB 特征 .npz（工具产出，git 忽略）
data/registry.json   照片-视频注册表
web/index.html       AR 页面骨架
web/app.js           主线程：采集 + 视频预热 + WebGL 渲染
web/cv-worker.js     Worker 线程：ORB 识别 + 合并索引匹配 + 光流追踪
web/style.css        样式
web/vendor/          OpenCV.js（手动下载，git 忽略；.gz 为预压缩版）
```

## 使用建议

- **注册图用视频原始帧**（无播放器 UI、无字幕遮挡的干净画面），叠加效果最佳
- 拍摄时让照片占画面一半以上、光线充足，识别率最高
- 低纹理照片（大片纯色/暗部）识别较弱，属已知边界
- 识别与阈值参数见 `web/cv-worker.js` 顶部常量区（`MIN_INLIERS`、`DETECT_INTERVAL` 等），后端备用参数见 `server/matcher.py` 顶部
- 前端代码修改后浏览器强刷（Ctrl+F5）即可，无需重启后端
- 安装 ffmpeg 后重新执行 `python tools/register.py` 可将存量视频转码为 faststart 格式（一次即可，`.optimized/` 有标记不重复转码）
