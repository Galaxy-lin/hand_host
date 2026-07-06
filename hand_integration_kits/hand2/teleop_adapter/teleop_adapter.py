"""hand2 遥操作适配器（接入通用上位机版本）。

输入：上位机相机识别到的人手数据
        - 默认 21x3 关键点数组（MediaPipe 顺序）
        - 也接受 {"points": [...], "features": {...}, "landmarks": [...]} 字典形式
输出：list[float]，长度 = profile.num_joints，单位“度”，顺序与关节表一致：
        index(MCP/Rot/PIP/DIP) → middle → ring → thumb(CMC/Axl/MCP/IP)

接入向导第 6 步建议填写：
  * 遥操作方式：       Python 遥操作适配器
  * 平滑系数：         0.55         （EMA alpha；越大越跟手，越小越稳）
  * 最大角速度：       8.00 rad/s   （命令变化的速率限幅）
  * 样本匹配邻居数：   5            （PairedSampleRetargeter kNN k）
  * 样本匹配宽度：     0.45         （PairedSampleRetargeter kNN sigma）
  * open/fist 标定文件： 建议使用 hand2/teleop_config.json
  * 配对样本标定文件：   建议使用 hand2/teleop_pair_calibration.json
  * 电机-仿真标定文件： 建议使用 hand2/motor_to_sim_calibration.json（驱动侧用，遥操作不依赖）
  * 指尖 link：        可留空；hand2 走 paired_semantic 模式不需要
  * 腕部 link：        base_link

工作流程
--------
1. extract_features:   21x3 点 → curl/spread/pinch/palm_size 等语义特征
2. Calibration:        用 open / fist 基线把 curl 归一化到 [0,1]
3. 重定向：
     * 优先 PairedSampleRetargeter（依赖 mujoco + teleop_pair_calibration.json，
       使用语义锚点 + kNN 融合，质量最高）
     * 否则 HeuristicRetargeter（纯 numpy，按 open↔closed 姿态用 curl 系数混合）
4. CommandSmoother:    EMA + rate-limit 平滑
5. 弧度 → 度
"""

from __future__ import annotations

import json
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import mujoco  # type: ignore
    HAS_MUJOCO = True
except Exception:
    mujoco = None  # type: ignore
    HAS_MUJOCO = False


_HAND2_HELPER_MODULE = None


def _load_hand2_helper():
    global _HAND2_HELPER_MODULE
    if _HAND2_HELPER_MODULE is not None:
        return _HAND2_HELPER_MODULE
    helper_path = Path(__file__).resolve().with_name("hand2_teleop.py")
    if not helper_path.exists():
        raise RuntimeError(f"缺少 hand2_teleop.py: {helper_path}")
    module_name = f"dexhand_hand2_teleop_helper_{abs(hash(helper_path))}"
    spec = importlib.util.spec_from_file_location(module_name, helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 hand2_teleop.py: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _HAND2_HELPER_MODULE = module
    return module


# -----------------------------------------------------------------------------
# 关节顺序（必须与 hand2 profile / hand2_teleop.EXPECTED_JOINTS 一致）
# -----------------------------------------------------------------------------

EXPECTED_JOINT_NAMES = (
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
)

FINGER_LANDMARKS: Dict[str, Tuple[int, int, int, int]] = {
    "thumb":  (1, 2, 3, 4),
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}

DEFAULT_CURL_RANGES: Dict[str, Tuple[float, float]] = {
    "curl_thumb":  (0.15, 1.25),
    "curl_index":  (0.10, 1.65),
    "curl_middle": (0.10, 1.65),
    "curl_ring":   (0.10, 1.65),
    "curl_pinky":  (0.10, 1.65),
}

FEATURE_VECTOR_ORDER = (
    "curl_thumb", "curl_index", "curl_middle", "curl_ring", "curl_pinky",
    "spread_thumb", "spread_index", "spread_middle", "spread_ring",
    "palm_size", "pinch_index", "pinch_middle",
)

# 向导默认值（与界面截图保持一致）
_DEFAULT_SMOOTH_ALPHA = 0.55
_DEFAULT_RATE_LIMIT = 8.0       # rad/s
_DEFAULT_PAIRED_K = 5
_DEFAULT_PAIRED_SIGMA = 0.45
_DEFAULT_PAIRED_MODE = "semantic"
_DEFAULT_PAIRED_KNN_BLEND = 0.25
_DEFAULT_PAIRED_SNAP_RADIUS = 1.10


# -----------------------------------------------------------------------------
# 几何 / 特征工具（与 hand2_teleop.py 保持一致）
# -----------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def _smoothstep(value: float) -> float:
    x = _clamp(value, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros_like(vec, dtype=float)
    return vec / norm


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    ua = _unit(a)
    ub = _unit(b)
    if not np.any(ua) or not np.any(ub):
        return 0.0
    return float(math.acos(_clamp(float(np.dot(ua, ub)), -1.0, 1.0)))


def _flexion(points: np.ndarray, a: int, b: int, c: int) -> float:
    interior = _angle_between(points[a] - points[b], points[c] - points[b])
    return _clamp(math.pi - interior, 0.0, math.pi)


def _distance(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a] - points[b]))


def _normalized_distance(points: np.ndarray, a: int, b: int, palm_size: float) -> float:
    return _distance(points, a, b) / max(palm_size, 1e-6)


def _finger_curl(points: np.ndarray, name: str) -> float:
    if name == "thumb":
        cmc, mcp, ip, tip = FINGER_LANDMARKS[name]
        return 0.55 * _flexion(points, cmc, mcp, ip) + 0.45 * _flexion(points, mcp, ip, tip)
    mcp, pip, dip, tip = FINGER_LANDMARKS[name]
    return (
        0.20 * _flexion(points, 0, mcp, pip)
        + 0.45 * _flexion(points, mcp, pip, dip)
        + 0.35 * _flexion(points, pip, dip, tip)
    )


def _palm_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    palm_x = _unit(points[5] - points[17])
    palm_y = _unit(points[9] - points[0])
    if not np.any(palm_x):
        palm_x = np.array([1.0, 0.0, 0.0], dtype=float)
    if not np.any(palm_y):
        palm_y = np.array([0.0, 1.0, 0.0], dtype=float)
    return palm_x, palm_y


def _spread_angle(
    points: np.ndarray, finger: str, palm_x: np.ndarray, palm_y: np.ndarray
) -> float:
    mcp, _pip, _dip, tip = FINGER_LANDMARKS[finger]
    ray = _unit(points[tip] - points[mcp])
    return float(math.atan2(float(np.dot(ray, palm_x)), float(np.dot(ray, palm_y))))


def extract_features(points: np.ndarray) -> Dict[str, float]:
    """从 21x3 点提取 curl/spread/pinch/palm_size 特征。"""

    palm_size = float(np.mean([_distance(points, 0, j) for j in (5, 9, 13, 17)]))
    palm_size = max(palm_size, 1e-6)
    palm_x, palm_y = _palm_frame(points)

    features: Dict[str, float] = {"palm_size": palm_size}
    for finger in FINGER_LANDMARKS:
        features[f"curl_{finger}"] = _finger_curl(points, finger)
    for finger in ("index", "middle", "ring", "thumb"):
        features[f"spread_{finger}"] = _spread_angle(points, finger, palm_x, palm_y)
    features["pinch_index"] = _normalized_distance(points, 4, 8, palm_size)
    features["pinch_middle"] = _normalized_distance(points, 4, 12, palm_size)
    return {k: float(v) for k, v in features.items() if np.isfinite(v)}


# -----------------------------------------------------------------------------
# 校准（open / fist 基线）
# -----------------------------------------------------------------------------

class _Calibration:
    """与 hand2_teleop.Calibration 一致的最小实现。"""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else None
        self.open_features: Dict[str, float] = {}
        self.fist_features: Dict[str, float] = {}
        self.loaded = False
        self.load()

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.open_features = self._clean(payload.get("open_features", {}))
        self.fist_features = self._clean(payload.get("fist_features", {}))
        self.loaded = True

    def save(self) -> None:
        if not self.path:
            return
        payload = {
            "version": 1,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "open_features": self.open_features,
            "fist_features": self.fist_features,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def capture_open(self, features: Dict[str, float]) -> None:
        self.open_features = self._clean(features)

    def capture_fist(self, features: Dict[str, float]) -> None:
        self.fist_features = self._clean(features)

    def normalized_curls(self, features: Dict[str, float]) -> Dict[str, float]:
        return {
            f: self.normalize_feature(f"curl_{f}", features.get(f"curl_{f}", 0.0))
            for f in FINGER_LANDMARKS
        }

    def normalize_feature(self, name: str, value: float) -> float:
        default_open, default_fist = DEFAULT_CURL_RANGES.get(name, (0.0, 1.0))
        open_v = self.open_features.get(name, default_open)
        fist_v = self.fist_features.get(name, default_fist)
        denom = fist_v - open_v
        if abs(denom) < 1e-5 or not np.isfinite(value):
            denom = default_fist - default_open
            open_v = default_open
        return _clamp((float(value) - open_v) / denom, 0.0, 1.0)

    def spread_baseline(self, name: str) -> float:
        return float(self.open_features.get(f"spread_{name}", 0.0))

    @staticmethod
    def _clean(features: Any) -> Dict[str, float]:
        if not isinstance(features, dict):
            return {}
        out: Dict[str, float] = {}
        for k, v in features.items():
            try:
                num = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(num):
                out[str(k)] = num
        return out


# -----------------------------------------------------------------------------
# 启发式 retargeter（无 mujoco 依赖）
# -----------------------------------------------------------------------------

_DEFAULT_OPEN_RAD = np.array([
    0.02, 0.00, 0.00, 0.00,
    0.02, 0.00, 0.00, 0.00,
    0.02, 0.00, 0.00, 0.00,
    0.10, -0.25, 0.02, 0.02,
], dtype=float)

_DEFAULT_CLOSED_RAD = np.array([
    1.35, 0.00, 1.45, 1.25,
    1.35, 0.00, 1.45, 1.25,
    1.35, 0.00, 1.45, 1.25,
    1.30, -1.35, 1.45, 1.20,
], dtype=float)


class _HeuristicRetargeter:
    """从 hand2_teleop.HandRetargeter 抽出的纯 numpy 版本。"""

    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        open_pose: np.ndarray,
        closed_pose: np.ndarray,
    ) -> None:
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.open_pose = np.clip(open_pose, self.low, self.high)
        self.closed_pose = np.clip(closed_pose, self.low, self.high)

    def open_command(self) -> np.ndarray:
        return self.open_pose.copy()

    def command(
        self,
        features: Dict[str, float],
        curls: Dict[str, float],
        calib: _Calibration,
    ) -> np.ndarray:
        cmd = self.open_pose.copy()
        self._set_long_finger(cmd, 0, curls.get("index", 0.0))
        self._set_long_finger(cmd, 4, curls.get("middle", 0.0))
        self._set_long_finger(cmd, 8, curls.get("ring", 0.0))
        self._set_spread(cmd, features, calib)
        self._set_thumb(cmd, features, curls.get("thumb", 0.0))
        return np.clip(cmd, self.low, self.high)

    def _set_long_finger(self, cmd: np.ndarray, offset: int, curl: float) -> None:
        flex = _smoothstep(curl)
        mcp = _smoothstep(_clamp(curl * 0.95 + 0.04, 0.0, 1.0))
        pip = _smoothstep(_clamp(curl * 1.08, 0.0, 1.0))
        dip = _smoothstep(_clamp(curl * 1.02, 0.0, 1.0))
        cmd[offset + 0] = self._lerp(offset + 0, mcp)
        cmd[offset + 2] = self._lerp(offset + 2, pip)
        cmd[offset + 3] = self._lerp(offset + 3, 0.85 * dip + 0.15 * flex)

    def _set_spread(
        self, cmd: np.ndarray, features: Dict[str, float], calib: _Calibration
    ) -> None:
        middle_now = float(features.get("spread_middle", 0.0))
        middle_base = calib.spread_baseline("middle")
        index_delta = (float(features.get("spread_index", 0.0)) - middle_now) - (
            calib.spread_baseline("index") - middle_base
        )
        middle_delta = middle_now - middle_base
        ring_delta = (float(features.get("spread_ring", 0.0)) - middle_now) - (
            calib.spread_baseline("ring") - middle_base
        )
        cmd[1] = self._clip_one(1, 1.15 * _clamp(index_delta, -0.75, 0.75))
        cmd[5] = self._clip_one(5, 0.45 * _clamp(middle_delta, -0.75, 0.75))
        cmd[9] = self._clip_one(9, 1.15 * _clamp(ring_delta, -0.75, 0.75))

    def _set_thumb(
        self, cmd: np.ndarray, features: Dict[str, float], thumb_curl: float
    ) -> None:
        pinch = float(features.get("pinch_index", 1.0))
        pinch_close = _clamp((0.70 - pinch) / (0.70 - 0.18), 0.0, 1.0)
        opposition = max(0.65 * thumb_curl, pinch_close)
        cmd[12] = self._lerp(12, _smoothstep(0.35 * thumb_curl + 0.65 * opposition))
        cmd[13] = self._lerp(13, _smoothstep(0.30 * thumb_curl + 0.70 * opposition))
        cmd[14] = self._lerp(14, _smoothstep(thumb_curl))
        cmd[15] = self._lerp(15, _smoothstep(_clamp(thumb_curl * 1.05, 0.0, 1.0)))

    def _lerp(self, idx: int, t: float) -> float:
        return float((1.0 - t) * self.open_pose[idx] + t * self.closed_pose[idx])

    def _clip_one(self, idx: int, value: float) -> float:
        return _clamp(value, float(self.low[idx]), float(self.high[idx]))


# -----------------------------------------------------------------------------
# 平滑器
# -----------------------------------------------------------------------------

class _CommandSmoother:
    def __init__(
        self,
        initial: np.ndarray,
        low: np.ndarray,
        high: np.ndarray,
        alpha: float,
        rate_limit: float,
    ) -> None:
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.alpha = _clamp(alpha, 0.0, 1.0)
        self.rate_limit = float(rate_limit)
        self.current = np.clip(np.asarray(initial, dtype=float), self.low, self.high)
        self._last_step = time.monotonic()

    def step(self, target: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        target = np.clip(
            np.nan_to_num(np.asarray(target, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
            self.low,
            self.high,
        )
        now = time.monotonic()
        if dt is None:
            dt = max(now - self._last_step, 1e-3)
        self._last_step = now
        blended = self.current + self.alpha * (target - self.current)
        max_step = max(self.rate_limit * max(float(dt), 1e-3), 1e-4)
        delta = np.clip(blended - self.current, -max_step, max_step)
        self.current = np.clip(self.current + delta, self.low, self.high)
        return self.current.copy()

    def reset(self, command: np.ndarray) -> None:
        self.current = np.clip(np.asarray(command, dtype=float), self.low, self.high)
        self._last_step = time.monotonic()


# -----------------------------------------------------------------------------
# TeleopAdapter 主类
# -----------------------------------------------------------------------------

class TeleopAdapter:
    """hand2 遥操作统一适配器。"""

    def __init__(self, profile):
        self.profile = profile
        self.config: Dict[str, Any] = (
            getattr(profile, "teleop_adapter_config", {}) or {}
        )
        self.num_joints = int(getattr(profile, "num_joints", 16))

        # ---- 向导可调参数 ----
        self.smooth_alpha = float(
            self.config.get(
                "smooth_alpha",
                self.config.get("smoothing_alpha", _DEFAULT_SMOOTH_ALPHA),
            )
        )
        self.rate_limit_rad_s = float(
            self.config.get(
                "rate_limit_rad_s",
                self.config.get("rate_limit_rad_per_s", _DEFAULT_RATE_LIMIT),
            )
        )
        self.paired_k = int(self.config.get("paired_k", _DEFAULT_PAIRED_K))
        self.paired_sigma = float(self.config.get("paired_sigma", _DEFAULT_PAIRED_SIGMA))
        self.paired_mode = str(self.config.get("paired_mode", _DEFAULT_PAIRED_MODE))
        self.paired_knn_blend = float(
            self.config.get("paired_knn_blend", _DEFAULT_PAIRED_KNN_BLEND)
        )
        self.paired_snap_radius = float(
            self.config.get("paired_snap_radius", _DEFAULT_PAIRED_SNAP_RADIUS)
        )

        # 仅记录方便调试：上位机不强求适配器去使用这两个 link 名，
        # 但留出来便于自定义重定向器接管。
        self.fingertip_links: Tuple[str, ...] = self._coerce_link_list(
            self.config.get("fingertip_links")
        )
        self.wrist_link: str = str(self.config.get("wrist_link", "base_link"))

        # ---- ctrl 范围 / open / closed ----
        self.low_rad, self.high_rad = self._build_ctrl_range()
        self.open_rad, self.closed_rad = self._build_open_closed()

        # ---- 校准 + 重定向 + 平滑 ----
        self._calibration = _Calibration(self._calibration_path())
        self._heuristic = _HeuristicRetargeter(
            self.low_rad, self.high_rad, self.open_rad, self.closed_rad
        )
        self._smoother = _CommandSmoother(
            initial=self.open_rad,
            low=self.low_rad,
            high=self.high_rad,
            alpha=self.smooth_alpha,
            rate_limit=self.rate_limit_rad_s,
        )

        # ---- 可选：mujoco + paired 重定向 ----
        self._mujoco_model = self._try_load_mujoco_model()
        self._paired = self._try_load_paired_retargeter()
        self._tracker = None

    # -------------------------------------------------------------------------
    # 公开接口（上位机调用）
    # -------------------------------------------------------------------------

    def retarget(self, hand_payload) -> List[float]:
        """把 21x3 关键点（或携带特征的字典）转成 num_joints 个关节角度（度）。"""

        features, _ = self._coerce_payload(hand_payload)
        if features is None:
            # 输入无效：保持平滑器当前状态，返回 profile 默认角度
            return self._default_angles_deg()

        ctrl_rad = self._compute_ctrl_rad(features)
        smoothed = self._smoother.step(ctrl_rad)
        return [math.degrees(float(v)) for v in smoothed]

    def reset(self) -> None:
        """切换姿态时清空平滑器，把当前命令拉回 open 姿态。"""

        self._smoother.reset(self.open_rad)

    def process_frame(self, frame):
        """使用旧版 hand2 的 HandTracker 处理摄像头帧，返回带标注画面和语义特征。"""

        h2 = _load_hand2_helper()
        if self._tracker is None:
            self._tracker = h2.HandTracker(preferred_hand="Right", accept_any=True)
        selected, _results = self._tracker.process(frame)
        self._tracker.draw_landmarks(frame, selected)
        if selected is None:
            return frame, None
        features = h2.extract_features(selected.points)
        return frame, {"features": features}

    def close(self) -> None:
        tracker = self._tracker
        self._tracker = None
        if tracker is not None and hasattr(tracker, "close"):
            try:
                tracker.close()
            except Exception:
                pass

    def calibrate_open(self, hand_payload) -> bool:
        features, _ = self._coerce_payload(hand_payload)
        if features is None:
            return False
        self._calibration.capture_open(features)
        return True

    def calibrate_fist(self, hand_payload) -> bool:
        features, _ = self._coerce_payload(hand_payload)
        if features is None:
            return False
        self._calibration.capture_fist(features)
        return True

    def save_calibration(self) -> None:
        self._calibration.save()

    @property
    def has_paired_retargeter(self) -> bool:
        return self._paired is not None

    @property
    def has_calibration(self) -> bool:
        return bool(
            self._calibration.open_features and self._calibration.fist_features
        )

    # -------------------------------------------------------------------------
    # 内部：计算 ctrl
    # -------------------------------------------------------------------------

    def _compute_ctrl_rad(self, features: Dict[str, float]) -> np.ndarray:
        if self._paired is not None:
            try:
                return self._paired.command(
                    features,
                    k=self.paired_k,
                    sigma=self.paired_sigma,
                    mode=self.paired_mode,
                    knn_blend=self.paired_knn_blend,
                    snap_radius=self.paired_snap_radius,
                )
            except Exception:
                # 配对模式失败时退化为启发式
                pass
        curls = self._calibration.normalized_curls(features)
        return self._heuristic.command(features, curls, self._calibration)

    # -------------------------------------------------------------------------
    # 输入归一化
    # -------------------------------------------------------------------------

    def _coerce_payload(
        self, hand_payload: Any
    ) -> Tuple[Optional[Dict[str, float]], Optional[np.ndarray]]:
        if hand_payload is None:
            return None, None

        if isinstance(hand_payload, dict):
            feats = hand_payload.get("features")
            if isinstance(feats, dict):
                clean = {
                    str(k): float(v)
                    for k, v in feats.items()
                    if isinstance(v, (int, float)) and np.isfinite(v)
                }
                if clean:
                    return clean, None
            raw_points = (
                hand_payload.get("points")
                or hand_payload.get("landmarks")
                or hand_payload.get("keypoints")
            )
            if raw_points is not None:
                points = self._coerce_points(raw_points)
                if points is not None:
                    return extract_features(points), points
            return None, None

        points = self._coerce_points(hand_payload)
        if points is None:
            return None, None
        return extract_features(points), points

    @staticmethod
    def _coerce_points(payload: Any) -> Optional[np.ndarray]:
        try:
            arr = np.asarray(payload, dtype=float)
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[0] != 21 or arr.shape[1] != 3:
            return None
        if not np.all(np.isfinite(arr)):
            return None
        return arr

    @staticmethod
    def _coerce_link_list(value: Any) -> Tuple[str, ...]:
        if not value:
            return ()
        if isinstance(value, str):
            parts = [s.strip() for s in value.replace(";", ",").split(",")]
            return tuple(p for p in parts if p)
        if isinstance(value, (list, tuple)):
            return tuple(str(p).strip() for p in value if str(p).strip())
        return ()

    # -------------------------------------------------------------------------
    # 默认值
    # -------------------------------------------------------------------------

    def _default_angles_deg(self) -> List[float]:
        return [
            joint.default_angle
            for finger in self.profile.finger_configs
            for joint in finger.joints
        ]

    # -------------------------------------------------------------------------
    # 配置加载
    # -------------------------------------------------------------------------

    def _calibration_path(self) -> Optional[Path]:
        path = self.config.get("calibration_path") or self.config.get(
            "open_fist_calibration_path"
        )
        if not path:
            path = getattr(self.profile, "teleop_config_path", None)
        if not path:
            local_path = Path(__file__).resolve().with_name("teleop_config.json")
            path = local_path if local_path.exists() else None
        return Path(path) if path else None

    def _paired_calibration_path(self) -> Optional[Path]:
        path = self.config.get("paired_calibration_path") or self.config.get(
            "pair_calibration_path"
        )
        if not path:
            path = getattr(self.profile, "teleop_pair_calibration_path", None)
        if not path:
            local_path = Path(__file__).resolve().with_name("teleop_pair_calibration.json")
            path = local_path if local_path.exists() else None
        return Path(path) if path else None

    def _model_path(self) -> Optional[Path]:
        path = self.config.get("model_path")
        if not path:
            path = getattr(self.profile, "model_path", None)
        return Path(path) if path else None

    def _build_ctrl_range(self) -> Tuple[np.ndarray, np.ndarray]:
        joints = [j for f in self.profile.finger_configs for j in f.joints]
        if len(joints) != self.num_joints:
            return _DEFAULT_OPEN_RAD - 1.0, _DEFAULT_CLOSED_RAD + 1.0
        low_rad = np.array([math.radians(float(j.min_angle)) for j in joints], dtype=float)
        high_rad = np.array([math.radians(float(j.max_angle)) for j in joints], dtype=float)
        swap_mask = low_rad > high_rad
        if np.any(swap_mask):
            low_rad[swap_mask], high_rad[swap_mask] = (
                high_rad[swap_mask].copy(),
                low_rad[swap_mask].copy(),
            )
        return low_rad, high_rad

    def _build_open_closed(self) -> Tuple[np.ndarray, np.ndarray]:
        gestures = getattr(self.profile, "default_gestures", {}) or {}
        open_deg = gestures.get("张开")
        closed_deg = gestures.get("握拳")
        if (
            isinstance(open_deg, (list, tuple))
            and isinstance(closed_deg, (list, tuple))
            and len(open_deg) == self.num_joints
            and len(closed_deg) == self.num_joints
        ):
            open_rad = np.radians(np.array(open_deg, dtype=float))
            closed_rad = np.radians(np.array(closed_deg, dtype=float))
            return (
                np.clip(open_rad, self.low_rad, self.high_rad),
                np.clip(closed_rad, self.low_rad, self.high_rad),
            )
        return (
            np.clip(_DEFAULT_OPEN_RAD, self.low_rad, self.high_rad),
            np.clip(_DEFAULT_CLOSED_RAD, self.low_rad, self.high_rad),
        )

    def _try_load_mujoco_model(self):
        if not HAS_MUJOCO:
            return None
        path = self._model_path()
        if not path:
            return None
        try:
            if not path.exists():
                return None
            return mujoco.MjModel.from_xml_path(str(path.resolve()))
        except Exception:
            return None

    def _try_load_paired_retargeter(self):
        if self._mujoco_model is None:
            return None
        path = self._paired_calibration_path()
        if not path or not path.exists():
            return None
        try:
            PairedSampleRetargeter = _load_hand2_helper().PairedSampleRetargeter
            paired = PairedSampleRetargeter(path, self._mujoco_model)
            return paired if paired.ready else None
        except Exception:
            return None


def create_teleop_adapter(profile) -> TeleopAdapter:
    """通用上位机统一入口：根据 profile 实例化遥操作适配器。"""

    return TeleopAdapter(profile)
