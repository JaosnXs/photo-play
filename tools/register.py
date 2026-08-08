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
import sys
from pathlib import Path

import cv2

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("register")


def find_video(photo_id: str) -> str | None:
    """按同名规则在视频目录中查找关联视频。"""
    for ext in VIDEO_EXTS:
        candidate = VIDEOS_DIR / f"{photo_id}{ext}"
        if candidate.exists():
            return candidate.name
    return None


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
        pts, des = matcher.extract(image)
        if des.shape[0] == 0:
            logger.error("未提取到特征点（图片纹理过少？），跳过: %s", photo_path.name)
            return False
        dims = (int(image.shape[1]), int(image.shape[0]))  # (宽, 高)
        matcher.save_features(FEATURES_DIR, photo_id, pts, des, dims=dims)
        logger.info("特征已生成: %s (%d 个特征点)", photo_id, des.shape[0])

    if video_file is None:
        video_file = find_video(photo_id)
    if video_file is None:
        logger.warning("未找到关联视频: %s（请放入 data/videos/%s.mp4 后用 --photo/--video 指定）",
                       photo_id, photo_id)
        return False

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
