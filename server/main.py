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

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.matcher import OrbMatcher
from server.registry import Registry

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "features"
PHOTOS_DIR = ROOT / "data" / "photos"
VIDEOS_DIR = ROOT / "data" / "videos"
REGISTRY_PATH = ROOT / "data" / "registry.json"
WEB_DIR = ROOT / "web"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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


@app.middleware("http")
async def no_store_cache(request, call_next):
    """HTML 页面禁用缓存（避免微信缓存旧版页面）；JS/图片等大文件允许缓存。"""
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


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
    # FileResponse 自动支持 Range，满足 iOS 播放要求
    return FileResponse(video_path)


@app.post("/api/reload")
def reload_library() -> dict:
    matcher.load(FEATURES_DIR)
    registry.reload()
    return {"status": "ok", "library_size": matcher.library_size,
            "registry_size": len(registry)}


def _find_photo(photo_id: str) -> Path | None:
    """按 photo_id 在照片目录中查找原图文件。"""
    for ext in IMAGE_EXTS:
        candidate = PHOTOS_DIR / f"{photo_id}{ext}"
        if candidate.exists():
            return candidate
    return None


@app.get("/api/library")
def library() -> list[dict]:
    """下发前端 AR 识别库：photo_id、标题、图片与视频地址。"""
    items = []
    for photo_id in registry.all_ids():
        entry = registry.get(photo_id) or {}
        video_path = VIDEOS_DIR / entry.get("video_file", "")
        if not video_path.exists():
            logger.warning("库条目视频缺失，跳过: %s", photo_id)
            continue
        if _find_photo(photo_id) is None:
            logger.warning("库条目原图缺失，跳过: %s", photo_id)
            continue
        items.append({
            "photo_id": photo_id,
            "title": entry.get("title", photo_id),
            "image_url": f"/photos/{photo_id}",
            "video_url": f"/videos/{photo_id}",
        })
    return items


@app.get("/photos/{photo_id}")
def get_photo(photo_id: str) -> FileResponse:
    if "/" in photo_id or "\\" in photo_id or ".." in photo_id:
        raise HTTPException(status_code=400, detail="非法 id")
    photo_path = _find_photo(photo_id)
    if photo_path is None:
        raise HTTPException(status_code=404, detail="照片不存在")
    return FileResponse(photo_path)


# 静态 H5 托管（放最后，避免覆盖 API 路由）
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
