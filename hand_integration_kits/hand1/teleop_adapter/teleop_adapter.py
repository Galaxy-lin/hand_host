"""DexPilot teleoperation adapter packaged with the hand.

This file intentionally contains the hand-specific teleoperation setup that used
to live in the host UI. The host only imports TeleopAdapter and calls retarget().
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _retargeting_dir() -> Path:
    try:
        from app.hands.base import APP_DIR

        return APP_DIR / "services" / "retargeting"
    except Exception:
        return Path(__file__).resolve().parents[4] / "services" / "retargeting"


def _ensure_retargeting_imports():
    retarget_dir = str(_retargeting_dir())
    if retarget_dir not in sys.path:
        sys.path.insert(0, retarget_dir)
    from dex_retargeting.mujoco_robot_wrapper import MujocoRobotWrapper
    from dex_retargeting.mujoco_vector_optimizer import MujocoDexPilotOptimizer
    from dex_retargeting.optimizer_utils import LPFilter

    return MujocoRobotWrapper, MujocoDexPilotOptimizer, LPFilter


class MujocoSeqRetargeting:
    def __init__(self, optimizer, has_joint_limits=True, lp_filter=None):
        self.optimizer = optimizer
        robot = self.optimizer.robot

        self.has_joint_limits = has_joint_limits
        joint_limits = np.ones_like(robot.joint_limits)
        joint_limits[:, 0] = -1e4
        joint_limits[:, 1] = 1e4
        if has_joint_limits:
            joint_limits[:] = robot.joint_limits[:]
            self.optimizer.set_joint_limit(joint_limits[self.optimizer.idx_pin2target])
        self.joint_limits = joint_limits[self.optimizer.idx_pin2target]

        self.last_qpos = joint_limits.mean(1)[self.optimizer.idx_pin2target].astype(np.float32)
        self.accumulated_time = 0
        self.num_retargeting = 0
        self.filter = lp_filter

    def retarget(self, ref_value, fixed_qpos=np.array([])):
        tic = time.perf_counter()
        if fixed_qpos.size == 0 and len(self.optimizer.idx_pin2fixed) > 0:
            fixed_qpos = self.optimizer.robot.q0[self.optimizer.idx_pin2fixed]
        qpos = self.optimizer.retarget(
            ref_value=ref_value.astype(np.float64),
            fixed_qpos=fixed_qpos.astype(np.float64),
            last_qpos=np.clip(self.last_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1]),
        )
        self.accumulated_time += time.perf_counter() - tic
        self.num_retargeting += 1
        self.last_qpos = qpos

        robot_qpos = np.zeros(self.optimizer.robot.dof)
        robot_qpos[self.optimizer.idx_pin2fixed] = fixed_qpos
        robot_qpos[self.optimizer.idx_pin2target] = qpos
        if self.filter is not None:
            robot_qpos = self.filter.next(robot_qpos)
        return robot_qpos

    def reset(self):
        self.last_qpos = self.joint_limits.mean(1).astype(np.float32)
        self.num_retargeting = 0
        self.accumulated_time = 0

    @property
    def joint_names(self):
        return self.optimizer.robot.dof_joint_names


def _coerce_points(payload: Any):
    if isinstance(payload, dict):
        payload = payload.get("joint_pos_3d") or payload.get("points") or payload.get("landmarks")
    if payload is None:
        return None
    arr = np.asarray(payload, dtype=np.float32)
    if arr.shape[0] < 21 or arr.shape[-1] < 3:
        return None
    return arr[:21, :3]


class TeleopAdapter:
    _DEFAULT_TIP_LINKS = (
        "Thumb_Visual_link",
        "Index_Visual_link",
        "Middle_Visual_link",
        "Ring_Visual_link",
    )
    _VEC_INDICES = np.array(
        [
            [8, 12, 16, 12, 16, 16, 0, 0, 0, 0],
            [4, 4, 4, 8, 8, 12, 4, 8, 12, 16],
        ]
    )

    def __init__(self, profile):
        self.profile = profile
        config = getattr(profile, "teleop_adapter_config", {}) or {}
        MujocoRobotWrapper, MujocoDexPilotOptimizer, LPFilter = _ensure_retargeting_imports()

        target_joint_names = list(
            config.get("target_joint_names")
            or getattr(profile, "teleop_joint_names", None)
            or profile.joint_names
        )
        output_joint_names = list(
            config.get("output_joint_names")
            or getattr(profile, "teleop_joint_names", None)
            or profile.joint_names
        )
        tip_link_names = list(
            config.get("tip_link_names")
            or getattr(profile, "teleop_tip_link_names", None)
            or self._DEFAULT_TIP_LINKS
        )
        wrist_link = str(
            config.get("wrist_link_name")
            or getattr(profile, "teleop_wrist_link_name", "base_link")
        )
        scaling = float(config.get("scaling", 1.2))
        filter_alpha = float(config.get("filter_alpha", 0.5))

        self.robot = MujocoRobotWrapper(str(profile.model_path), from_string=False)
        optimizer = MujocoDexPilotOptimizer(
            self.robot,
            target_joint_names,
            finger_tip_link_names=tip_link_names,
            wrist_link_name=wrist_link,
            scaling=scaling,
        )
        self.retargeting = MujocoSeqRetargeting(
            optimizer,
            has_joint_limits=True,
            lp_filter=LPFilter(filter_alpha),
        )
        robot_joint_names = list(self.retargeting.joint_names)
        missing = [name for name in output_joint_names if name not in robot_joint_names]
        if missing:
            raise RuntimeError("DexPilot output joints missing in model: " + ", ".join(missing[:8]))
        self._output_indices = [robot_joint_names.index(name) for name in output_joint_names]

    def retarget(self, hand_payload) -> list[float]:
        points = _coerce_points(hand_payload)
        if points is None:
            return [
                joint.default_angle
                for finger in self.profile.finger_configs
                for joint in finger.joints
            ]
        points = points.copy()
        points[:, 0] = -points[:, 0]
        ref_value = points[self._VEC_INDICES[1], :] - points[self._VEC_INDICES[0], :]
        qpos = self.retargeting.retarget(ref_value)
        angles_deg = np.degrees(qpos).tolist()
        return [float(angles_deg[i]) for i in self._output_indices][: self.profile.num_joints]

    def reset(self):
        self.retargeting.reset()


def create_teleop_adapter(profile):
    return TeleopAdapter(profile)
