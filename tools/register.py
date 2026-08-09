"""照片注册建库 CLI。

用法（在项目根目录执行）：
    python tools/register.py                # 增量注册：跳过已有特征的照片
    python tools/register.py --force        # 全量重建特征
    python tools/register.py --photo xxx.jpg --video xxx.mp4   # 显式指定关联

约定：
- 照片放 data/photos/，视频放 data/videos/；
- 默认同名关联：photos/A.jpg -> videos/A.mp4（或 A.mov / A.m4v / A.webm）；
- photo_id 取照片文件名去扩展名；
- 输出特征到 data/features/{photo_id}.npz，登记 data/registry.json。
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# 允许从项目根目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.matcher import OrbMatcher
from server.registry import Registry

ROOT = Path(__file__).resolve().parent.parent
PHOTOS_DIR = ROOT / "data" / "photos"
VIDEOS_DIR = ROOT / "data" / "videos"
FEATURES_DIR = ROOT / "data" / "features"
REGISTRY_PATH = ROOT / "data" / "registry.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = [".mp4", ".mov", ".m4v", ".webm"]

# 多尺度特征提取配置：与前端查询帧的 ORB 参数保持一致（描述子空间才兼容）
LIB_MAX_W = 1600                 # 注册图限宽（大图为注册质量，限宽控制特征规模）
SCALES = (1.0, 0.5, 0.25)        # 多尺度覆盖"大/中/小目标"
ORB_FEATURES_PER_SCALE = 1500    # 每尺度特征上限（规模与识别率的平衡）
ORB_N_LEVELS = 16
ORB_SCALE_FACTOR = 1.2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("register")

# 注册专用 ORB（特征数独立于服务端 matcher，避免库描述子过多拖慢前端匹配）
_orb = cv2.ORB_create(nfeatures=ORB_FEATURES_PER_SCALE,
                      nlevels=ORB_N_LEVELS, scaleFactor=ORB_SCALE_FACTOR)


def extract_multiscale(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对注册图做三尺度 ORB 提取，坐标统一映射回原图像素坐标系。

    返回 (原图像素坐标 pts[N,2], 合并描述子 des[N,32])。
    多尺度拼接可覆盖"近距离大目标 / 远距离小目标"，提升实拍召回。
    """
    h0, w0 = image.shape[:2]
    base = min(1.0, LIB_MAX_W / w0)  # 大图先整体限宽
    all_pts: list[np.ndarray] = []
    all_des: list[np.ndarray] = []
    for s in SCALES:
        scale = base * s
        w = max(1, int(round(w0 * scale)))
        h = max(1, int(round(h0 * scale)))
        resized = cv2.resize(image, (w, h)) if scale < 1.0 else image
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        kps, des = _orb.detectAndCompute(gray, None)
        if des is None or len(kps) == 0:
            continue
        pts = np.float32([kp.pt for kp in kps])
        pts /= scale  # 该尺度像素坐标 -> 原图像素坐标
        all_pts.append(pts)
        all_des.append(des)
    if not all_des:
        return (np.empty((0, 2), np.float32), np.empty((0, 32), np.uint8))
    return np.vstack(all_pts).astype(np.float32), np.vstack(all_des)


def find_video(photo_id: str) -> str | None:
    """按同名规则在视频目录中查找关联视频。"""
    for ext in VIDEO_EXTS:
        candidate = VIDEOS_DIR / f"{photo_id}{ext}"
        if candidate.exists():
            return candidate.name
    return None


def optimize_video(video_file: str, force: bool) -> None:
    """用 ffmpeg 将视频转码为流媒体友好格式，显著改善起播与移动端兼容性。

    - +faststart：moov 原子前置，支持边下边播（否则需下载大半才能起播）
    - H.264 + yuv420p + AAC：iOS Safari / 微信内置浏览器兼容
    - 限宽 1920 + CRF 26：控制码率，降低弱网卡顿
    转码成功的视频在 data/videos/.optimized/ 留标记，重复注册不重复转码。
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("未检测到 ffmpeg，跳过视频转码优化（安装后可显著提升起播速度）")
        return
    src = VIDEOS_DIR / video_file
    mark_dir = VIDEOS_DIR / ".optimized"
    mark_dir.mkdir(exist_ok=True)
    marker = mark_dir / video_file
    if marker.exists() and not force:
        return
    tmp = src.with_name(src.stem + ".opt.mp4")
    cmd = [ffmpeg, "-y", "-i", str(src),
           "-vf", "scale='min(1920,iw)':-2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
           "-movflags", "+faststart", str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"")[-200:].decode("utf-8", "ignore")
        logger.warning("视频转码失败 %s: %s", video_file, tail)
        tmp.unlink(missing_ok=True)
        return
    old_kb = src.stat().st_size // 1024
    tmp.replace(src)
    new_kb = src.stat().st_size // 1024
    marker.touch()
    logger.info("视频已优化: %s (%d KB -> %d KB)", video_file, old_kb, new_kb)


def register_one(matcher: OrbMatcher, registry: Registry,
                 photo_path: Path, video_file: str | None, force: bool) -> bool:
    photo_id = photo_path.stem
    feature_path = matcher._feature_path(FEATURES_DIR, photo_id)

    if feature_path.exists() and not force:
        logger.info("跳过（特征已存在）: %s", photo_path.name)
    else:
        image = cv2.imread(str(photo_path))
        if image is None:
            logger.error("无法读取图片，跳过: %s", photo_path.name)
            return False
        pts, des = extract_multiscale(image)
        if des.shape[0] == 0:
            logger.error("未提取到特征点（图片纹理过少？），跳过: %s", photo_path.name)
            return False
        dims = (int(image.shape[1]), int(image.shape[0]))  # (宽, 高)
        matcher.save_features(FEATURES_DIR, photo_id, pts, des, dims=dims)
        logger.info("特征已生成: %s (%d 个特征点, 三尺度)", photo_id, des.shape[0])

    if video_file is None:
        video_file = find_video(photo_id)
    if video_file is None:
        logger.warning("未找到关联视频: %s（请放入 data/videos/%s.mp4 后用 --photo/--video 指定）",
                       photo_id, photo_id)
        return False

    if video_file.lower().endswith(".mp4"):
        optimize_video(video_file, force)
    else:
        logger.warning("非 mp4 视频跳过转码优化（建议转为 mp4）: %s", video_file)

    registry.set(photo_id, video_file, title=photo_id)
    logger.info("已登记: %s -> %s", photo_id, video_file)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="照片注册建库工具")
    parser.add_argument("--force", action="store_true", help="全量重建特征")
    parser.add_argument("--photo", help="单张注册：照片文件名（位于 data/photos/）")
    parser.add_argument("--video", help="单张注册：关联视频文件名（位于 data/videos/）")
    args = parser.parse_args()

    matcher = OrbMatcher()
    registry = Registry(REGISTRY_PATH)

    if args.photo:
        photo_path = PHOTOS_DIR / args.photo
        if not photo_path.exists():
            logger.error("照片不存在: %s", photo_path)
            sys.exit(1)
        ok = register_one(matcher, registry, photo_path, args.video, force=True)
        registry.save()
        sys.exit(0 if ok else 1)

    photos = sorted(p for p in PHOTOS_DIR.iterdir()
                    if p.suffix.lower() in IMAGE_EXTS) if PHOTOS_DIR.exists() else []
    if not photos:
        logger.warning("data/photos/ 中没有图片，请先放入注册用高清原图")
        sys.exit(1)

    success = sum(register_one(matcher, registry, p, None, args.force) for p in photos)
    registry.save()
    logger.info("注册完成：成功 %d / 共 %d，注册表总计 %d 条",
                success, len(photos), len(registry))


if __name__ == "__main__":
    main()
