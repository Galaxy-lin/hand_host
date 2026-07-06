"""hand2 MediaPipe-to-MuJoCo retargeting runtime used by the GUI."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import mujoco
import numpy as np


def _require_mediapipe_solutions() -> None:
    if hasattr(mp, "solutions"):
        return
    version = getattr(mp, "__version__", "unknown")
    raise RuntimeError(
        f"当前 mediapipe {version} 不包含旧版 mp.solutions API；"
        "请安装兼容版本: python3 -m pip install --force-reinstall 'mediapipe<0.10.22'"
    )


EXPECTED_JOINTS = (
    "if_mcp",
    "if_rot",
    "if_pip",
    "if_dip",
    "mf_mcp",
    "mf_rot",
    "mf_pip",
    "mf_dip",
    "rf_mcp",
    "rf_rot",
    "rf_pip",
    "rf_dip",
    "th_cmc",
    "th_axl",
    "th_mcp",
    "th_ipl",
)

FINGER_LANDMARKS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

DEFAULT_CURL_RANGES = {
    "curl_thumb": (0.15, 1.25),
    "curl_index": (0.10, 1.65),
    "curl_middle": (0.10, 1.65),
    "curl_ring": (0.10, 1.65),
    "curl_pinky": (0.10, 1.65),
}

FEATURE_VECTOR_ORDER = (
    "curl_thumb",
    "curl_index",
    "curl_middle",
    "curl_ring",
    "curl_pinky",
    "spread_thumb",
    "spread_index",
    "spread_middle",
    "spread_ring",
    "palm_size",
    "pinch_index",
    "pinch_middle",
)

def clamp(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def smoothstep(value: float) -> float:
    x = clamp(value, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.zeros_like(vec, dtype=float)
    return vec / norm


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    ua = unit(a)
    ub = unit(b)
    if not np.any(ua) or not np.any(ub):
        return 0.0
    return float(math.acos(clamp(float(np.dot(ua, ub)), -1.0, 1.0)))


def flexion_angle(points: np.ndarray, a: int, b: int, c: int) -> float:
    interior = angle_between(points[a] - points[b], points[c] - points[b])
    return clamp(math.pi - interior, 0.0, math.pi)


def distance(points: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a] - points[b]))


def normalized_distance(points: np.ndarray, a: int, b: int, palm_size: float) -> float:
    return distance(points, a, b) / max(palm_size, 1e-6)


def finger_curl(points: np.ndarray, name: str) -> float:
    if name == "thumb":
        cmc, mcp, ip, tip = FINGER_LANDMARKS[name]
        mcp_flex = flexion_angle(points, cmc, mcp, ip)
        ip_flex = flexion_angle(points, mcp, ip, tip)
        return 0.55 * mcp_flex + 0.45 * ip_flex

    mcp, pip, dip, tip = FINGER_LANDMARKS[name]
    mcp_flex = flexion_angle(points, 0, mcp, pip)
    pip_flex = flexion_angle(points, mcp, pip, dip)
    dip_flex = flexion_angle(points, pip, dip, tip)
    return 0.20 * mcp_flex + 0.45 * pip_flex + 0.35 * dip_flex


def palm_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    palm_x = unit(points[5] - points[17])
    palm_y = unit(points[9] - points[0])
    if not np.any(palm_x):
        palm_x = np.array([1.0, 0.0, 0.0], dtype=float)
    if not np.any(palm_y):
        palm_y = np.array([0.0, 1.0, 0.0], dtype=float)
    return palm_x, palm_y


def spread_angle(points: np.ndarray, finger: str, palm_x: np.ndarray, palm_y: np.ndarray) -> float:
    mcp, _pip, _dip, tip = FINGER_LANDMARKS[finger]
    ray = unit(points[tip] - points[mcp])
    return float(math.atan2(float(np.dot(ray, palm_x)), float(np.dot(ray, palm_y))))


def extract_features(points: np.ndarray) -> dict[str, float]:
    palm_size = float(
        np.mean(
            [
                distance(points, 0, 5),
                distance(points, 0, 9),
                distance(points, 0, 13),
                distance(points, 0, 17),
            ]
        )
    )
    palm_size = max(palm_size, 1e-6)
    palm_x, palm_y = palm_frame(points)

    features: dict[str, float] = {"palm_size": palm_size}
    for finger in FINGER_LANDMARKS:
        features[f"curl_{finger}"] = finger_curl(points, finger)

    for finger in ("index", "middle", "ring", "thumb"):
        features[f"spread_{finger}"] = spread_angle(points, finger, palm_x, palm_y)

    features["pinch_index"] = normalized_distance(points, 4, 8, palm_size)
    features["pinch_middle"] = normalized_distance(points, 4, 12, palm_size)
    return {k: float(v) for k, v in features.items() if np.isfinite(v)}


class Calibration:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.open_features: dict[str, float] = {}
        self.fist_features: dict[str, float] = {}
        self.loaded = False

    def load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.open_features = self._clean_features(payload.get("open_features", {}))
        self.fist_features = self._clean_features(payload.get("fist_features", {}))
        self.loaded = True

    def save(self) -> None:
        payload = {
            "version": 1,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "open_features": self.open_features,
            "fist_features": self.fist_features,
        }
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def reset(self) -> None:
        self.open_features = {}
        self.fist_features = {}
        self.loaded = False

    def capture_open(self, features: dict[str, float]) -> None:
        self.open_features = self._clean_features(features)

    def capture_fist(self, features: dict[str, float]) -> None:
        self.fist_features = self._clean_features(features)

    def normalized_curls(self, features: dict[str, float]) -> dict[str, float]:
        return {
            finger: self.normalized_feature(f"curl_{finger}", features.get(f"curl_{finger}", 0.0))
            for finger in FINGER_LANDMARKS
        }

    def normalized_feature(self, name: str, value: float) -> float:
        default_open, default_fist = DEFAULT_CURL_RANGES.get(name, (0.0, 1.0))
        open_value = self.open_features.get(name, default_open)
        fist_value = self.fist_features.get(name, default_fist)
        denom = fist_value - open_value
        if abs(denom) < 1e-5 or not np.isfinite(value):
            denom = default_fist - default_open
            open_value = default_open
        return clamp((value - open_value) / denom, 0.0, 1.0)

    def spread_baseline(self, name: str) -> float:
        return self.open_features.get(f"spread_{name}", 0.0)

    def status(self) -> str:
        open_state = "yes" if self.open_features else "no"
        fist_state = "yes" if self.fist_features else "no"
        loaded = " loaded" if self.loaded else ""
        return f"open={open_state} fist={fist_state}{loaded}"

    @staticmethod
    def _clean_features(features: dict[str, Any]) -> dict[str, float]:
        clean: dict[str, float] = {}
        for key, value in features.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                clean[str(key)] = numeric
        return clean


class HandRetargeter:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.ctrl_range = np.array(model.actuator_ctrlrange, dtype=float)
        self.low = self.ctrl_range[:, 0]
        self.high = self.ctrl_range[:, 1]
        self._validate_model()

        self.open_pose = self._clipped(
            np.array(
                [
                    0.02,
                    0.00,
                    0.00,
                    0.00,
                    0.02,
                    0.00,
                    0.00,
                    0.00,
                    0.02,
                    0.00,
                    0.00,
                    0.00,
                    0.10,
                    -0.25,
                    0.02,
                    0.02,
                ],
                dtype=float,
            )
        )
        self.closed_pose = self._clipped(
            np.array(
                [
                    1.35,
                    0.00,
                    1.45,
                    1.25,
                    1.35,
                    0.00,
                    1.45,
                    1.25,
                    1.35,
                    0.00,
                    1.45,
                    1.25,
                    1.30,
                    -1.35,
                    1.45,
                    1.20,
                ],
                dtype=float,
            )
        )

    def command(self, features: dict[str, float], curls: dict[str, float], calib: Calibration) -> np.ndarray:
        cmd = self.open_pose.copy()
        self._set_long_finger(cmd, 0, curls["index"])
        self._set_long_finger(cmd, 4, curls["middle"])
        self._set_long_finger(cmd, 8, curls["ring"])
        self._set_spread(cmd, features, calib)
        self._set_thumb(cmd, features, curls["thumb"])
        return self._clipped(cmd)

    def open_command(self) -> np.ndarray:
        return self.open_pose.copy()

    def _set_long_finger(self, cmd: np.ndarray, offset: int, curl: float) -> None:
        flex = smoothstep(curl)
        mcp = smoothstep(clamp(curl * 0.95 + 0.04, 0.0, 1.0))
        pip = smoothstep(clamp(curl * 1.08, 0.0, 1.0))
        dip = smoothstep(clamp(curl * 1.02, 0.0, 1.0))
        cmd[offset + 0] = self._lerp_pose(offset + 0, mcp)
        cmd[offset + 2] = self._lerp_pose(offset + 2, pip)
        cmd[offset + 3] = self._lerp_pose(offset + 3, 0.85 * dip + 0.15 * flex)

    def _set_spread(self, cmd: np.ndarray, features: dict[str, float], calib: Calibration) -> None:
        middle_now = features.get("spread_middle", 0.0)
        middle_base = calib.spread_baseline("middle")

        index_delta = (features.get("spread_index", 0.0) - middle_now) - (
            calib.spread_baseline("index") - middle_base
        )
        middle_delta = middle_now - middle_base
        ring_delta = (features.get("spread_ring", 0.0) - middle_now) - (
            calib.spread_baseline("ring") - middle_base
        )

        cmd[1] = self._clip_one(1, 1.15 * clamp(index_delta, -0.75, 0.75))
        cmd[5] = self._clip_one(5, 0.45 * clamp(middle_delta, -0.75, 0.75))
        cmd[9] = self._clip_one(9, 1.15 * clamp(ring_delta, -0.75, 0.75))

    def _set_thumb(self, cmd: np.ndarray, features: dict[str, float], thumb_curl: float) -> None:
        pinch = features.get("pinch_index", 1.0)
        pinch_close = clamp((0.70 - pinch) / (0.70 - 0.18), 0.0, 1.0)
        opposition = max(0.65 * thumb_curl, pinch_close)

        cmd[12] = self._lerp_pose(12, smoothstep(0.35 * thumb_curl + 0.65 * opposition))
        cmd[13] = self._lerp_pose(13, smoothstep(0.30 * thumb_curl + 0.70 * opposition))
        cmd[14] = self._lerp_pose(14, smoothstep(thumb_curl))
        cmd[15] = self._lerp_pose(15, smoothstep(clamp(thumb_curl * 1.05, 0.0, 1.0)))

    def _lerp_pose(self, index: int, t: float) -> float:
        return float((1.0 - t) * self.open_pose[index] + t * self.closed_pose[index])

    def _clip_one(self, index: int, value: float) -> float:
        return clamp(value, float(self.low[index]), float(self.high[index]))

    def _clipped(self, command: np.ndarray) -> np.ndarray:
        return np.clip(command, self.low, self.high)

    def _validate_model(self) -> None:
        if self.model.nu != len(EXPECTED_JOINTS):
            raise ValueError(f"Expected 16 actuators, got {self.model.nu}")
        actual = []
        for i in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[i, 0])
            actual.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        if tuple(actual) != EXPECTED_JOINTS:
            expected = ", ".join(EXPECTED_JOINTS)
            got = ", ".join(str(x) for x in actual)
            raise ValueError(f"Unexpected actuator joint order.\nExpected: {expected}\nGot: {got}")


class PairedSampleRetargeter:
    def __init__(self, path: Path, model: mujoco.MjModel) -> None:
        self.path = path
        self.model = model
        self.feature_order: tuple[str, ...] = FEATURE_VECTOR_ORDER
        self.sample_names: list[str] = []
        self.sample_feature_dicts: list[dict[str, float]] = []
        self.sample_features = np.zeros((0, len(self.feature_order)), dtype=float)
        self.sample_ctrl = np.zeros((0, model.nu), dtype=float)
        self.records_features: dict[str, dict[str, float]] = {}
        self.records_ctrl: dict[str, np.ndarray] = {}
        self.center = np.zeros(len(self.feature_order), dtype=float)
        self.scale = np.ones(len(self.feature_order), dtype=float)
        self.low = np.array(model.actuator_ctrlrange[:, 0], dtype=float)
        self.high = np.array(model.actuator_ctrlrange[:, 1], dtype=float)
        self._load()

    @property
    def ready(self) -> bool:
        return len(self.sample_names) > 0

    def describe(self) -> str:
        if not self.ready:
            return "paired calibration has no valid samples"
        mode = "semantic anchors" if "open_hand" in self.records_ctrl else "nearest samples"
        return f"paired calibration loaded: {len(self.sample_names)} samples, mode={mode}"

    def command(
        self,
        features: dict[str, float],
        k: int,
        sigma: float,
        mode: str = "semantic",
        knn_blend: float = 0.20,
        snap_radius: float = 1.15,
    ) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("paired calibration has no valid samples")
        if mode == "knn":
            return self._knn_command(features, k, sigma)
        semantic = self._semantic_command(features, snap_radius)
        if semantic is None:
            return self._knn_command(features, k, sigma)
        if mode == "hybrid":
            blend = clamp(knn_blend, 0.0, 1.0)
            knn = self._knn_command(features, k, sigma)
            return np.clip((1.0 - blend) * semantic + blend * knn, self.low, self.high)
        return semantic

    def _knn_command(self, features: dict[str, float], k: int, sigma: float) -> np.ndarray:
        distances = self._distances(features)

        count = max(1, min(int(k), len(distances)))
        nearest = np.argsort(distances)[:count]
        local_distances = distances[nearest]
        if sigma <= 1e-6:
            weights = 1.0 / np.maximum(local_distances, 1e-6)
        else:
            weights = np.exp(-0.5 * (local_distances / sigma) ** 2)
        if float(np.sum(weights)) <= 1e-9:
            weights = np.ones_like(weights)
        weights = weights / np.sum(weights)
        command = np.sum(self.sample_ctrl[nearest] * weights[:, None], axis=0)
        return np.clip(command, self.low, self.high)

    def classify(self, features: dict[str, float], radius: float) -> str:
        if not self.ready:
            return "UNKNOWN"
        distances = self._distances(features)
        nearest = int(np.argmin(distances))
        name = self.sample_names[nearest]
        distance = float(distances[nearest])
        if distance <= radius:
            return f"PAIR:{name}"
        return f"PAIR_NEAR:{name}"

    def _semantic_command(self, features: dict[str, float], snap_radius: float) -> np.ndarray | None:
        open_ctrl = self.records_ctrl.get("open_hand")
        if open_ctrl is None:
            return None

        cmd = open_ctrl.copy()
        index_alpha = self._curl_alpha(features, "index", "index_down")
        middle_alpha = self._curl_alpha(features, "middle", "middle_down")
        ring_alpha = self._curl_alpha(features, "ring", "ring_down")
        thumb_alpha = self._curl_alpha(features, "thumb", "thumb_down")
        pinky_alpha = self._curl_alpha(features, "pinky", "peace", fallback_gesture="fist")

        self._blend_anchor(cmd, open_ctrl, "index_down", index_alpha, [0, 1, 2, 3])
        self._blend_anchor(cmd, open_ctrl, "middle_down", middle_alpha, [4, 5, 6, 7])
        self._blend_anchor(cmd, open_ctrl, "ring_down", ring_alpha, [8, 9, 10, 11])
        self._blend_anchor(cmd, open_ctrl, "thumb_down", thumb_alpha, [12, 13, 14, 15])

        pinch_alpha = self._pinch_alpha(features, "ok")
        ok_alpha = pinch_alpha * (1.0 - middle_alpha) * (1.0 - ring_alpha)
        self._blend_to_pose(cmd, "ok", smoothstep(ok_alpha), [0, 1, 2, 3, 12, 13, 14, 15])

        peace_alpha = ring_alpha * pinky_alpha * (1.0 - index_alpha) * (1.0 - middle_alpha)
        self._blend_to_pose(cmd, "peace", smoothstep(peace_alpha), list(range(self.model.nu)))

        middle_thumb_alpha = middle_alpha * thumb_alpha * (1.0 - index_alpha) * (1.0 - ring_alpha)
        self._blend_to_pose(
            cmd,
            "middle_thumb_down",
            smoothstep(middle_thumb_alpha),
            [4, 5, 6, 7, 12, 13, 14, 15],
        )

        fist_alpha = min(index_alpha, middle_alpha, ring_alpha)
        fist_alpha *= max(thumb_alpha, self._curl_alpha(features, "thumb", "fist", fallback_gesture="thumb_down"))
        self._blend_to_pose(cmd, "fist", smoothstep(fist_alpha), list(range(self.model.nu)))
        snap_name, snap_alpha = self._nearest_snap(features, snap_radius)
        if snap_name is not None:
            self._blend_to_pose(cmd, snap_name, snap_alpha, list(range(self.model.nu)))
        return np.clip(cmd, self.low, self.high)

    def _blend_anchor(
        self,
        cmd: np.ndarray,
        open_ctrl: np.ndarray,
        gesture: str,
        alpha: float,
        indices: list[int],
    ) -> None:
        pose = self.records_ctrl.get(gesture)
        if pose is None:
            pose = self.records_ctrl.get("fist")
        if pose is None:
            return
        t = smoothstep(alpha)
        for idx in indices:
            cmd[idx] = (1.0 - t) * open_ctrl[idx] + t * pose[idx]

    def _blend_to_pose(self, cmd: np.ndarray, gesture: str, alpha: float, indices: list[int]) -> None:
        pose = self.records_ctrl.get(gesture)
        if pose is None:
            return
        t = clamp(alpha, 0.0, 1.0)
        for idx in indices:
            cmd[idx] = (1.0 - t) * cmd[idx] + t * pose[idx]

    def _curl_alpha(
        self,
        features: dict[str, float],
        finger: str,
        closed_gesture: str,
        fallback_gesture: str = "fist",
    ) -> float:
        key = f"curl_{finger}"
        open_value = self._feature_value("open_hand", key, DEFAULT_CURL_RANGES.get(key, (0.0, 1.0))[0])
        closed_value = self._feature_value(
            closed_gesture,
            key,
            self._feature_value(fallback_gesture, key, DEFAULT_CURL_RANGES.get(key, (0.0, 1.0))[1]),
        )
        current = float(features.get(key, open_value))
        denom = closed_value - open_value
        if abs(denom) < 1e-5:
            default_open, default_closed = DEFAULT_CURL_RANGES.get(key, (0.0, 1.0))
            open_value = default_open
            denom = default_closed - default_open
        return clamp((current - open_value) / denom, 0.0, 1.0)

    def _pinch_alpha(self, features: dict[str, float], gesture: str) -> float:
        key = "pinch_index"
        open_value = self._feature_value("open_hand", key, 1.0)
        closed_value = self._feature_value(gesture, key, 0.35)
        current = float(features.get(key, open_value))
        denom = open_value - closed_value
        if abs(denom) < 1e-5:
            return 0.0
        return clamp((open_value - current) / denom, 0.0, 1.0)

    def _feature_value(self, gesture: str, key: str, default: float) -> float:
        value = self.records_features.get(gesture, {}).get(key, default)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(numeric):
            return float(default)
        return numeric

    def _nearest_snap(self, features: dict[str, float], radius: float) -> tuple[str | None, float]:
        if radius <= 1e-6 or not self.ready:
            return None, 0.0
        distances = self._distances(features)
        nearest = int(np.argmin(distances))
        distance = float(distances[nearest])
        alpha = smoothstep(clamp((radius - distance) / radius, 0.0, 1.0))
        if alpha <= 0.0:
            return None, 0.0
        return self.sample_names[nearest], alpha

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with self.path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        order = payload.get("feature_vector_order")
        if isinstance(order, list) and all(isinstance(x, str) for x in order):
            self.feature_order = tuple(order)

        names: list[str] = []
        feature_dicts: list[dict[str, float]] = []
        feature_rows: list[list[float]] = []
        ctrl_rows: list[list[float]] = []
        gestures = payload.get("gestures", {})
        if not isinstance(gestures, dict):
            return

        for name, record in gestures.items():
            if not isinstance(record, dict):
                continue
            mp_record = record.get("mediapipe", {})
            mujoco_record = record.get("mujoco", {})
            if not isinstance(mp_record, dict) or not isinstance(mujoco_record, dict):
                continue

            vector = mp_record.get("feature_vector")
            feature_dict = self._clean_feature_dict(mp_record.get("features", {}))
            if not self._valid_vector(vector, len(self.feature_order)):
                vector = [float(feature_dict.get(key, 0.0)) for key in self.feature_order]
            ctrl = mujoco_record.get("ctrl")
            if not self._valid_vector(vector, len(self.feature_order)):
                continue
            if not self._valid_vector(ctrl, self.model.nu):
                continue

            gesture_name = str(name)
            ctrl_array = np.clip(np.array([float(v) for v in ctrl], dtype=float), self.low, self.high)
            names.append(gesture_name)
            feature_dicts.append(feature_dict)
            feature_rows.append([float(v) for v in vector])
            ctrl_rows.append(ctrl_array.tolist())
            self.records_features[gesture_name] = feature_dict
            self.records_ctrl[gesture_name] = ctrl_array

        if not names:
            return
        self.sample_names = names
        self.sample_feature_dicts = feature_dicts
        self.sample_features = np.array(feature_rows, dtype=float)
        self.sample_ctrl = np.clip(np.array(ctrl_rows, dtype=float), self.low, self.high)
        self.center = np.mean(self.sample_features, axis=0)
        self.scale = np.ptp(self.sample_features, axis=0)
        for idx, name in enumerate(self.feature_order):
            if name.startswith("spread_"):
                self.scale[idx] = max(float(np.std(self.sample_features[:, idx])), 0.35)
        self.scale[self.scale < 1e-4] = 1.0

    def _vector_from_features(self, features: dict[str, float]) -> np.ndarray:
        values = [float(features.get(key, 0.0)) for key in self.feature_order]
        return np.nan_to_num(np.array(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def _distances(self, features: dict[str, float]) -> np.ndarray:
        query = self._vector_from_features(features)
        diffs = self.sample_features - query[None, :]
        for idx, name in enumerate(self.feature_order):
            if name.startswith("spread_"):
                diffs[:, idx] = self._angle_diff(self.sample_features[:, idx], query[idx])
        return np.linalg.norm(diffs / self.scale[None, :], axis=1)

    @staticmethod
    def _angle_diff(values: np.ndarray, reference: float) -> np.ndarray:
        return np.arctan2(np.sin(values - reference), np.cos(values - reference))

    @staticmethod
    def _clean_feature_dict(value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            return {}
        clean: dict[str, float] = {}
        for key, raw in value.items():
            try:
                numeric = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                clean[str(key)] = numeric
        return clean

    @staticmethod
    def _valid_vector(value: Any, length: int) -> bool:
        if not isinstance(value, list) or len(value) != length:
            return False
        try:
            numeric = np.array([float(v) for v in value], dtype=float)
        except (TypeError, ValueError):
            return False
        return bool(np.all(np.isfinite(numeric)))


class CommandSmoother:
    def __init__(self, initial: np.ndarray, low: np.ndarray, high: np.ndarray, alpha: float, rate_limit: float) -> None:
        self.current = np.array(initial, dtype=float)
        self.low = np.array(low, dtype=float)
        self.high = np.array(high, dtype=float)
        self.alpha = clamp(alpha, 0.0, 1.0)
        self.rate_limit = float(rate_limit)

    def step(self, target: np.ndarray, dt: float) -> np.ndarray:
        target = np.clip(np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0), self.low, self.high)
        blended = self.current + self.alpha * (target - self.current)
        max_step = max(self.rate_limit * max(dt, 1e-3), 1e-4)
        delta = np.clip(blended - self.current, -max_step, max_step)
        self.current = np.clip(self.current + delta, self.low, self.high)
        return self.current.copy()

    def reset(self, command: np.ndarray) -> None:
        self.current = np.clip(command, self.low, self.high)


@dataclass
class SelectedHand:
    image_landmarks: Any
    points: np.ndarray
    label: str
    score: float


class HandTracker:
    def __init__(self, preferred_hand: str, accept_any: bool) -> None:
        _require_mediapipe_solutions()
        self.preferred_hand = preferred_hand
        self.accept_any = accept_any
        self.mp_hands = mp.solutions.hands
        self.drawer = mp.solutions.drawing_utils
        self.drawer_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )

    def close(self) -> None:
        self.hands.close()

    def process(self, bgr_frame: np.ndarray) -> tuple[SelectedHand | None, Any]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True
        if not results.multi_hand_landmarks:
            return None, results

        candidates: list[SelectedHand] = []
        for idx, image_landmarks in enumerate(results.multi_hand_landmarks):
            label = "Unknown"
            score = 0.0
            if results.multi_handedness and idx < len(results.multi_handedness):
                classification = results.multi_handedness[idx].classification[0]
                label = classification.label
                score = float(classification.score)

            landmarks = image_landmarks
            if results.multi_hand_world_landmarks and idx < len(results.multi_hand_world_landmarks):
                landmarks = results.multi_hand_world_landmarks[idx]
            points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=float)
            candidates.append(SelectedHand(image_landmarks, points, label, score))

        preferred = [c for c in candidates if c.label.lower() == self.preferred_hand.lower()]
        if preferred:
            return max(preferred, key=lambda c: c.score), results
        if self.accept_any:
            return max(candidates, key=lambda c: c.score), results
        return None, results

    def draw_landmarks(self, frame: np.ndarray, selected: SelectedHand | None) -> None:
        if selected is None:
            return
        self.drawer.draw_landmarks(
            frame,
            selected.image_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.drawer_styles.get_default_hand_landmarks_style(),
            self.drawer_styles.get_default_hand_connections_style(),
        )


def load_model(path: Path) -> mujoco.MjModel:
    if not path.exists():
        raise FileNotFoundError(path)
    return mujoco.MjModel.from_xml_path(str(path.resolve()))
