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
启动：浏览器加载本地 OpenCV.js (WASM)
      → GET /api/library 拉取注册表
      → 浏览器内对注册原图提取 ORB 特征（三尺度拼接：1x / 0.5x / 0.25x）

识别：每 300ms 静默截帧（限宽 800px）
      → ORB 特征提取 → 逐库 knnMatch + Lowe 比率过滤
      → findHomography(RANSAC) 几何校验，内点 ≥ 10 判定命中

追踪：命中后切换 LK 光流逐帧追踪（30-60fps）
      → 每帧由匹配点增量更新单应性矩阵 H_new = T · H_old
      → 连续 6 帧追踪失败或视频播完，回到识别状态

渲染：WebGL 透视绘制视频帧到照片四角位置
      （绕开 CSS 3D transform 在各手机浏览器上的兼容性差异）
```

后端为纯静态服务：注册表下发、注册原图、视频流（含 Range 支持），不参与实时识别——识别与追踪全部在前端本地完成，一次加载后离线可用。

## 快速开始

```powershell
# 1. 安装依赖（Python 3.10+）
pip install -r requirements.txt

# 2. 下载 OpenCV.js（约 10MB，保存到 web/vendor/opencv.js）
curl -L -o web/vendor/opencv.js https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js

# 3. 放入素材（同名关联：照片 A.jpg ↔ 视频 A.mp4）
#    data/photos/A.jpg    注册用高清原图
#    data/videos/A.mp4    关联视频

# 4. 注册建库
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
# 以 cpolar 为例
cpolar http 8000
# 得到 https://xxxx.cpolar.top 后，手机浏览器/微信内打开
```

首次打开需下载约 10MB 识别引擎（之后有浏览器缓存），加载完成提示"对准照片，自动识别"即可使用。

## 接口一览

| 接口 | 说明 |
|---|---|
| `GET /` | AR 版 H5 页面（自动识别 + 透视叠加） |
| `GET /classic.html` | 备用版页面（拍照按钮 + 后端识别 + 整页播放） |
| `GET /api/library` | 下发识别库（photo_id / 原图地址 / 视频地址） |
| `GET /photos/{id}` | 注册原图 |
| `GET /videos/{id}` | 关联视频（Range 流式，iOS 兼容） |
| `POST /api/match` | 后端识别接口（备用版页面使用） |
| `POST /api/reload` | 热加载特征库与注册表 |
| `GET /api/health` | 健康检查 |

## 目录结构

```
server/main.py       FastAPI：静态下发 + 备用识别接口 + Range 视频流
server/matcher.py    ORB 匹配器（备用版后端识别用；含单应性计算）
server/registry.py   registry.json 读写封装
tools/register.py    建库 CLI（增量 / --force 全量 / --photo+--video 单张指定）
data/photos/         注册用高清原图（用户放入，git 忽略）
data/videos/         关联视频（用户放入，git 忽略）
data/features/       ORB 特征 .npz（工具产出，git 忽略）
data/registry.json   照片-视频注册表
web/index.html       AR 版页面骨架
web/app.js           AR 核心：ORB 识别 + 光流追踪 + WebGL 渲染
web/classic.html     备用版页面（自包含）
web/style.css        样式
web/vendor/          OpenCV.js（手动下载，git 忽略）
```

## 使用建议

- **注册图用视频原始帧**（无播放器 UI、无字幕遮挡的干净画面），叠加效果最佳
- 拍摄时让照片占画面一半以上、光线充足，识别率最高
- 低纹理照片（大片纯色/暗部）识别较弱，属已知边界
- 识别与阈值参数见 `web/app.js` 顶部常量区（`MIN_INLIERS`、`DETECT_INTERVAL` 等），后端备用参数见 `server/matcher.py` 顶部
