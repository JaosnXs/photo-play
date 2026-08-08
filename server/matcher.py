"""图像匹配器：以照片为 key 的相似检索。

设计要点：
- Matcher 抽象基类定义 load/match 接口，业务层只依赖接口。
  当前实现为 OrbMatcher（内存 HashMap 暴力遍历，百张规模 < 100ms），
  未来库容增大时可替换为 FAISS 向量索引实现，业务层零改动。
- 判定采用双层结构：Lowe 比率过滤出候选，再做单应性 RANSAC 几何校验，
  以"宁缺毋滥"避免误识别误播。
- 命中时返回单应性矩阵（归一化注册图坐标 -> 查询图像素坐标），
  供前端将视频透视叠加到实拍照片上（AR"照片变活"效果）。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---- 可调阈值 ----
ORB_N_FEATURES = 3000       # 每张图提取的最大特征点数（实拍目标占比小，需足量特征）
ORB_N_LEVELS = 16           # 金字塔层数：覆盖注册高清图与翻拍小图之间的大尺度差
ORB_SCALE_FACTOR = 1.2
LOWE_RATIO = 0.75           # Lowe 比率过滤阈值
MIN_GOOD_MATCHES = 15       # 进入几何校验所需的最少匹配点数
MIN_INLIERS = 10            # 单应性内点判定阈值
RANSAC_REPROJ_THRESHOLD = 5.0


@dataclass
class MatchResult:
    photo_id: str
    inliers: int  # 单应性内点数，用于日志与阈值调参
    # 3x3 单应性（行优先展平为 9 个 float）：
    # 归一化注册图坐标 [0,1]^2 -> 查询图像素坐标
    homography: Optional[list[float]] = None
    lib_size: Optional[tuple[int, int]] = None  # 注册图 (宽, 高)


class Matcher(ABC):
    """匹配器抽象接口。"""

    @abstractmethod
    def load(self, features_dir: Path | str) -> None:
        """加载特征库到内存。"""

    @abstractmethod
    def match(self, image_bytes: bytes) -> Optional[MatchResult]:
        """对上传图片做识别，命中返回 MatchResult，未命中返回 None。"""


class OrbMatcher(Matcher):
    """ORB 特征点 + 单应性校验匹配器。

    特征库结构: dict[photo_id, (keypoints_xy, descriptors, dims)]
    描述符匹配走 HashMap 逐库遍历，photo_id 查找 O(1)。
    """

    def __init__(self) -> None:
        self._orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES,
                                   nlevels=ORB_N_LEVELS,
                                   scaleFactor=ORB_SCALE_FACTOR)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # photo_id -> (关键点坐标, 描述符, (宽, 高))
        self._library: dict[str, tuple[np.ndarray, np.ndarray, Optional[tuple[int, int]]]] = {}

    # ---------- 特征提取 ----------

    def extract(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """提取 ORB 特征，返回 (关键点坐标[N,2], 描述符[N,32])。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        kps, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(kps) == 0:
            return np.empty((0, 2), np.float32), np.empty((0, 32), np.uint8)
        pts = np.float32([kp.pt for kp in kps])
        return pts, des

    def extract_from_bytes(self, image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
        """从上传字节流解码并提取特征，避免落盘临时文件。"""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法解码上传的图片")
        return self.extract(image)

    # ---------- 特征库加载 ----------

    @staticmethod
    def _feature_path(features_dir: Path, photo_id: str) -> Path:
        return features_dir / f"{photo_id}.npz"

    def save_features(self, features_dir: Path | str, photo_id: str,
                      pts: np.ndarray, des: np.ndarray,
                      dims: Optional[tuple[int, int]] = None) -> None:
        """将提取的特征持久化（供注册工具调用）。dims 为注册图 (宽, 高)。"""
        features_dir = Path(features_dir)
        features_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._feature_path(features_dir, photo_id),
                            pts=pts, des=des,
                            dims=np.array(dims if dims else (0, 0), np.int64))

    def load(self, features_dir: Path | str) -> None:
        """启动时加载所有 .npz 特征到内存 HashMap。"""
        features_dir = Path(features_dir)
        self._library.clear()
        if not features_dir.exists():
            logger.warning("特征目录不存在: %s", features_dir)
            return
        for fp in sorted(features_dir.glob("*.npz")):
            try:
                data = np.load(fp)
                pts, des = data["pts"], data["des"]
                if des.ndim != 2 or des.shape[0] == 0:
                    logger.warning("特征为空，跳过: %s", fp.name)
                    continue
                dims: Optional[tuple[int, int]] = None
                if "dims" in data:
                    w, h = (int(v) for v in data["dims"])
                    dims = (w, h) if w > 0 and h > 0 else None
                self._library[fp.stem] = (pts, des, dims)
            except Exception as exc:  # 单个文件损坏不拖垮整体加载
                logger.error("加载特征失败: %s (%s)", fp.name, exc)
        logger.info("特征库已加载: %d 张照片", len(self._library))

    @property
    def library_size(self) -> int:
        return len(self._library)

    # ---------- 匹配 ----------

    def _score_against(self, query_pts: np.ndarray, query_des: np.ndarray,
                       lib_pts: np.ndarray, lib_des: np.ndarray,
                       ) -> tuple[int, Optional[np.ndarray]]:
        """对单个库条目打分。

        返回 (内点数, H_lib_to_query)。H 将注册图像素坐标映射到查询图像素坐标；
        未通过校验返回 (0, None)。
        """
        matches = self._bf.knnMatch(query_des, lib_des, k=2)
        good = [m for pair in matches if len(pair) == 2
                for m, n in [pair] if m.distance < LOWE_RATIO * n.distance]
        if len(good) < MIN_GOOD_MATCHES:
            return 0, None
        src = query_pts[[m.queryIdx for m in good]]
        dst = lib_pts[[m.trainIdx for m in good]]
        # M 将查询图坐标映射到注册图坐标
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
        if M is None or mask is None:
            return 0, None
        inliers = int(mask.sum())
        if inliers < MIN_INLIERS:
            return inliers, None
        # 反解得到 注册图 -> 查询图 的映射（AR 叠加所需方向）
        try:
            H = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return inliers, None
        if abs(H[2, 2]) < 1e-12:
            return inliers, None
        H = H / H[2, 2]
        return inliers, H

    def match(self, image_bytes: bytes) -> Optional[MatchResult]:
        """识别上传图片：逐库打分，取内点数最高且达标者为命中。"""
        if not self._library:
            logger.info("特征库为空，无法匹配")
            return None
        query_pts, query_des = self.extract_from_bytes(image_bytes)
        if query_des.shape[0] == 0:
            logger.info("上传图片未提取到特征点")
            return None

        best_id: Optional[str] = None
        best_inliers = 0
        best_H: Optional[np.ndarray] = None
        best_dims: Optional[tuple[int, int]] = None
        for photo_id, (lib_pts, lib_des, dims) in self._library.items():
            inliers, H = self._score_against(query_pts, query_des, lib_pts, lib_des)
            if H is not None and inliers > best_inliers:
                best_id, best_inliers, best_H, best_dims = photo_id, inliers, H, dims

        if best_id is not None and best_H is not None and best_inliers >= MIN_INLIERS:
            H_norm = best_H
            if best_dims is not None:
                # 归一化: 让矩阵接受 [0,1]^2 的注册图相对坐标
                w, h = best_dims
                H_norm = best_H @ np.diag([w, h, 1.0])
                H_norm = H_norm / H_norm[2, 2]
            logger.info("匹配命中: %s (inliers=%d)", best_id, best_inliers)
            return MatchResult(photo_id=best_id, inliers=best_inliers,
                               homography=[float(v) for v in H_norm.flatten()],
                               lib_size=best_dims)
        logger.info("未命中 (最佳 inliers=%d, 阈值=%d)", best_inliers, MIN_INLIERS)
        return None
