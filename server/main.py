"""FastAPI 服务入口。

接口：
- POST /api/match          上传照片 -> 识别 -> 返回 video_url 或未命中
- GET  /videos/{photo_id}  Range 流式返回关联视频（iOS Safari/微信必需）
- POST /api/reload         热加载特征库与注册表（注册新照片后免重启）
- GET  /api/health         健康检查（特征库规模）
- GET  /                   托管 web/ 静态 H5 页面

运行（项目根目录）：
    uvicorn server.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import struct
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.matcher import OrbMatcher
from server.registry import Registry

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "features"
VIDEOS_DIR = ROOT / "data" / "videos"
REGISTRY_PATH = ROOT / "data" / "registry.json"
WEB_DIR = ROOT / "web"

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15MB 上传上限

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("photoplay")

app = FastAPI(title="PhotoPlay", docs_url=None, redoc_url=None)

matcher = OrbMatcher()
registry = Registry(REGISTRY_PATH)


@app.on_event("startup")
def startup() -> None:
    matcher.load(FEATURES_DIR)
    registry.reload()
    _invalidate_features_cache()


@app.middleware("http")
async def no_store_cache(request, call_next):
    """HTML 页面禁用缓存（避免微信缓存旧版页面）；JS/图片等大文件允许缓存。"""
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


@app.middleware("http")
async def precompressed_static(request: Request, call_next):
    """静态大文件（如 opencv.js）优先下发预压缩 .gz 版本，显著降低传输量。

    仅当：客户端声明支持 gzip（Accept-Encoding 含 gzip/br 等）且对应 .gz 文件存在。
    """
    if request.method != "GET":
        return await call_next(request)

    # 只处理静态资源，跳过 API 与视频流
    path = request.url.path
    if path.startswith("/api/") or path.startswith("/videos/") or path.startswith("/photos/"):
        return await call_next(request)

    # 解析出文件在 web/ 下的相对路径
    rel = path.lstrip("/")
    if not rel or rel.endswith("/"):
        return await call_next(request)

    plain = WEB_DIR / rel
    if not plain.is_file():
        return await call_next(request)

    accept_encoding = request.headers.get("accept-encoding", "")
    wants_gzip = "gzip" in accept_encoding or "br" in accept_encoding or "deflate" in accept_encoding
    gz_path = plain.with_name(plain.name + ".gz")
    if wants_gzip and gz_path.is_file():
        # 用原文件扩展名推断正确 Content-Type（.gz 后缀会让猜测失效）
        import mimetypes
        media_type = mimetypes.guess_type(plain.name)[0] or "application/octet-stream"
        return FileResponse(
            gz_path,
            media_type=media_type,
            headers={
                "Content-Encoding": "gzip",
                "Cache-Control": "public, max-age=86400",
                "Vary": "Accept-Encoding",
            },
        )
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # noqa: ANN001
    logger.exception("未处理异常: %s", exc)
    return JSONResponse(status_code=500,
                        content={"matched": False, "error": "服务器内部错误"})


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "library_size": matcher.library_size,
            "registry_size": len(registry)}


@app.post("/api/match")
async def match_photo(file: UploadFile) -> JSONResponse:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="空文件")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图片过大")

    try:
        result = matcher.match(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result is None:
        return JSONResponse({"matched": False})

    entry = registry.get(result.photo_id)
    if entry is None:
        logger.warning("命中照片 %s 但注册表无记录", result.photo_id)
        return JSONResponse({"matched": False, "error": "注册表缺少关联视频"})

    video_path = VIDEOS_DIR / entry["video_file"]
    if not video_path.exists():
        logger.warning("视频文件缺失: %s", video_path)
        return JSONResponse({"matched": False, "error": "视频文件不存在"})

    return JSONResponse({
        "matched": True,
        "photo_id": result.photo_id,
        "title": entry.get("title", result.photo_id),
        "inliers": result.inliers,
        "video_url": f"/videos/{result.photo_id}",
        "homography": result.homography,
        "lib_size": list(result.lib_size) if result.lib_size else None,
    })


@app.get("/videos/{photo_id}")
def get_video(photo_id: str) -> FileResponse:
    # 防路径穿越
    if "/" in photo_id or "\\" in photo_id or ".." in photo_id:
        raise HTTPException(status_code=400, detail="非法 id")
    entry = registry.get(photo_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="未登记的照片")
    video_path = VIDEOS_DIR / entry["video_file"]
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="视频文件不存在")
    # FileResponse 自动支持 Range，满足 iOS 播放要求；
    # Cache-Control 让前端预热/预加载的缓冲可被 arVideo 复用，命中后免重复下载
    return FileResponse(video_path,
                        headers={"Cache-Control": "public, max-age=86400"})


@app.post("/api/reload")
def reload_library() -> dict:
    matcher.load(FEATURES_DIR)
    registry.reload()
    _invalidate_features_cache()
    return {"status": "ok", "library_size": matcher.library_size,
            "registry_size": len(registry)}


# ---- 特征库二进制载荷缓存：启动/热加载时构建一次，请求时零序列化直发 ----
_features_lock = threading.Lock()
_features_cache: tuple[str, bytes, bytes] | None = None  # (etag, body, gzip_body)


def _build_features_cache() -> tuple[str, bytes, bytes]:
    """构建二进制特征库载荷（含预压缩 gzip 变体）。

    格式：[u32 LE json长度][json 元数据数组][拼接的 pts(f32) / des(u8) 二进制]
    元数据每项：{photo_id, title, video_url, n, pts_off, des_off}
    （off 为相对二进制区起点的字节偏移）
    """
    meta: list[dict] = []
    blob = bytearray()
    for photo_id in registry.all_ids():
        entry = registry.get(photo_id) or {}
        video_path = VIDEOS_DIR / entry.get("video_file", "")
        if not video_path.exists():
            logger.warning("特征条目视频缺失，跳过: %s", photo_id)
            continue
        npz = FEATURES_DIR / f"{photo_id}.npz"
        if not npz.exists():
            logger.warning("特征文件缺失，跳过: %s", photo_id)
            continue
        try:
            data = np.load(npz)
            pts, des = data["pts"], data["des"]
            if des.ndim != 2 or des.shape[0] == 0 or "dims" not in data:
                continue
            w, h = (int(v) for v in data["dims"])
            if w <= 0 or h <= 0:
                continue
            # 关键点归一化到 [0,1]^2（与图像绝对尺寸解耦），float32 紧凑传输
            norm = np.ascontiguousarray(pts.astype(np.float32) / np.array([w, h], np.float32))
            des_c = np.ascontiguousarray(des)
            pts_off = len(blob)
            blob += norm.tobytes()
            des_off = len(blob)
            blob += des_c.tobytes()
            meta.append({
                "photo_id": photo_id,
                "title": entry.get("title", photo_id),
                "video_url": f"/videos/{photo_id}",
                "n": int(des.shape[0]),
                "pts_off": pts_off,
                "des_off": des_off,
            })
        except Exception as exc:  # 单个文件损坏不拖垮整体下发
            logger.warning("特征读取失败 %s: %s", photo_id, exc)
    header = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    body = struct.pack("<I", len(header)) + header + bytes(blob)
    etag = hashlib.md5(body, usedforsecurity=False).hexdigest()
    gz = gzip.compress(body, compresslevel=6)
    logger.info("特征库载荷已构建: %d 张, %.1f KB (gzip %.1f KB)",
                len(meta), len(body) / 1024, len(gz) / 1024)
    return etag, body, gz


def _invalidate_features_cache() -> None:
    global _features_cache
    with _features_lock:
        _features_cache = _build_features_cache()


@app.get("/api/features")
def features(request: Request) -> Response:
    """下发前端 AR 识别特征库（二进制，内存缓存 + ETag 协商缓存 + 预压缩 gzip）。

    前端据此直接在 Worker 内构造特征库做本地匹配，无需下载注册原图、
    无需在浏览器里重复提取特征，显著加快启动加载与识别速度。
    """
    with _features_lock:
        cache = _features_cache
    if cache is None:
        _invalidate_features_cache()
        with _features_lock:
            cache = _features_cache
    etag, body, gz = cache
    headers = {"ETag": etag, "Cache-Control": "no-cache", "Vary": "Accept-Encoding"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    if "gzip" in request.headers.get("accept-encoding", ""):
        headers["Content-Encoding"] = "gzip"
        return Response(content=gz, media_type="application/octet-stream",
                        headers=headers)
    return Response(content=body, media_type="application/octet-stream",
                    headers=headers)


# 静态 H5 托管（放最后，避免覆盖 API 路由）
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
