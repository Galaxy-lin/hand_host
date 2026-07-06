"""Motor adapter for hand1/hand3 legacy 16-motor protocol.

The host calls get_latest_telemetry() from a Qt timer on the UI thread, so this
adapter keeps hardware status polling in a background thread and exposes only a
cached telemetry snapshot to the host.
"""

from __future__ import annotations

import threading
import time
from typing import Dict

try:
    from legacy_motor import Motor

    HAS_MOTOR = True
except Exception:  # pragma: no cover - depends on runtime environment
    Motor = None
    HAS_MOTOR = False


class HandDriver:
    _FULL_SCAN_EVERY = 5
    _RESPONSE_DEADLINE_S = 0.06

    def __init__(self, profile):
        self.profile = profile
        self.config = getattr(profile, "motor_adapter_config", {}) or {}
        self.motor = None
        self.connected = False
        self.enabled = False
        self.angles = {
            joint.motor_id: joint.default_angle
            for finger in profile.finger_configs
            for joint in finger.joints
        }

        self._motor_ids = tuple(sorted(self.angles))
        self._status_polling_enabled = bool(
            self.config.get("status_polling_enabled", True)
        )
        self._poll_interval_s = max(
            0.05,
            float(self.config.get("telemetry_period_ms", 150)) / 1000.0,
        )
        self._poll_stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._telemetry_lock = threading.RLock()
        self._latest_telemetry: Dict[int, Dict] = {}
        self._status_cache: Dict[int, Dict] = {}
        self._prev_angles: Dict[int, float] = {}
        self._prev_time = 0.0
        self._poll_cycle = 0

    def connect(self, port: str, baudrate: int, protocol: str | None = None) -> bool:
        if not HAS_MOTOR:
            raise RuntimeError("legacy_motor.Motor is unavailable")
        self.disconnect()
        self.motor = Motor(port=port, baudrate=baudrate)
        self.connected = bool(self.motor.connect())
        if self.connected:
            self._start_polling()
        return self.connected

    def disconnect(self) -> None:
        self._stop_polling()
        if self.motor is not None:
            try:
                self.motor.disconnect()
            except Exception:
                pass
        self.motor = None
        self.connected = False
        self.enabled = False

    def enable_all(self, enable: bool = True) -> None:
        if self.motor is None or not self.connected:
            return
        self.motor.enable_all_motors_broadcast(bool(enable))
        self.enabled = bool(enable)
        self._refresh_cached_enabled_state()

    def set_motor_angle(self, motor_id: int, angle_deg: float, speed: float = 0.5) -> None:
        motor_id = int(motor_id)
        angle_deg = float(angle_deg)
        self.angles[motor_id] = angle_deg
        self._update_cached_target_angles({motor_id: angle_deg})
        if self.motor is not None and self.connected:
            self.motor.set_motor_angle(motor_id, angle_deg, speed=speed)

    def set_angles(self, angles_dict: dict[int, float]) -> None:
        normalized = {int(k): float(v) for k, v in angles_dict.items()}
        self.angles.update(normalized)
        self._update_cached_target_angles(normalized)
        if self.motor is not None and self.connected:
            self.motor.set_motor_angles_broadcast(normalized)

    def set_all_joint_angles(self, angles_list: list[float]) -> None:
        if len(angles_list) != self.profile.num_joints:
            raise ValueError(
                f"need {self.profile.num_joints} angles, got {len(angles_list)}"
            )
        self.set_angles(
            {
                self.profile.joint_index_to_motor_id[index]: float(angle)
                for index, angle in enumerate(angles_list)
            }
        )

    def home_all_motors(
        self,
        target_angles: dict[int, float] | None = None,
        speed: float = 0.3,
    ) -> None:
        if target_angles is None:
            target_angles = {
                joint.motor_id: joint.default_angle
                for finger in self.profile.finger_configs
                for joint in finger.joints
            }
        self.enable_all(True)
        self.set_angles(target_angles)

    def get_latest_telemetry(self) -> dict[int, dict]:
        with self._telemetry_lock:
            if self._latest_telemetry:
                return {mid: dict(data) for mid, data in self._latest_telemetry.items()}
        return self._fallback_telemetry()

    def get_status(self, motor_id: int) -> dict:
        return self.get_latest_telemetry().get(int(motor_id), {"online": False})

    def set_zero_point(self, motor_id: int) -> None:
        if self.motor is None or not self.connected:
            return
        can_data = bytearray(8)
        can_data[0] = int(motor_id) & 0xFF
        can_data[1] = 3
        frame = self.motor._build_frame(0xA0, bytes(can_data))
        self.motor._send_raw(frame)

    def set_max_current(self, motor_id: int, value_q15: int) -> None:
        if self.motor is None or not self.connected:
            return
        can_data = bytearray(8)
        can_data[0] = int(motor_id) & 0xFF
        can_data[1] = 18
        can_data[4] = int(value_q15) & 0xFF
        can_data[5] = (int(value_q15) >> 8) & 0xFF
        frame = self.motor._build_frame(0xA0, bytes(can_data))
        self.motor._send_raw(frame)

    def cleanup(self) -> None:
        self.disconnect()

    def _start_polling(self) -> None:
        if not self._status_polling_enabled or self.motor is None:
            with self._telemetry_lock:
                self._latest_telemetry = self._fallback_telemetry()
            return
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            name=f"{getattr(self.profile, 'hand_id', 'hand')}-motor-poll",
            daemon=True,
        )
        self._poll_thread.start()

    def _stop_polling(self) -> None:
        self._poll_stop.set()
        thread = self._poll_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._poll_thread = None

    def _poll_loop(self) -> None:
        self._prev_time = time.monotonic()
        while not self._poll_stop.is_set():
            start = time.monotonic()
            try:
                self._poll_once()
            except Exception:
                pass
            elapsed = time.monotonic() - start
            delay = max(0.02, self._poll_interval_s - elapsed)
            self._poll_stop.wait(delay)

    def _poll_once(self) -> None:
        motor = self.motor
        if motor is None or not self.connected or not getattr(motor, "connected", False):
            return

        ids_to_poll = self._ids_to_poll()
        if not ids_to_poll:
            return

        try:
            motor.clear_response_queue()
        except Exception:
            pass

        status_pull_id = getattr(motor, "STATUS_PULL_ID", 0xA0)
        for motor_id in ids_to_poll:
            if self._poll_stop.is_set():
                return
            try:
                frame1 = motor._build_frame(
                    status_pull_id, bytes([motor_id, 0x01]) + b"\x00" * 6
                )
                frame2 = motor._build_frame(
                    status_pull_id, bytes([motor_id, 0x02]) + b"\x00" * 6
                )
                motor._send_raw(frame1)
                motor._send_raw(frame2)
            except Exception:
                continue

        time.sleep(max(len(ids_to_poll) * 0.004, 0.02))

        responded: set[int] = set()
        ext_data: Dict[int, Dict] = {}
        deadline = time.monotonic() + self._RESPONSE_DEADLINE_S
        while not self._poll_stop.is_set() and time.monotonic() < deadline:
            try:
                response = motor.get_response(timeout=0.005)
            except Exception:
                break
            if response is None:
                continue
            data = getattr(response, "data", None)
            if not isinstance(data, dict) or "motor_id" not in data:
                continue
            motor_id = int(data["motor_id"])
            if motor_id not in self.angles:
                continue
            if data.get("status_index") == 0x02:
                ext_data[motor_id] = data
            else:
                self._status_cache[motor_id] = dict(data)
                responded.add(motor_id)

        for motor_id in ids_to_poll:
            if motor_id in responded:
                continue
            previous = self._status_cache.get(motor_id, {})
            self._status_cache[motor_id] = {
                "motor_id": motor_id,
                "online": False,
                "angle": previous.get("angle", self.angles.get(motor_id, 0.0)),
                "fsm_state": 0,
                "error_code": 0,
                "temp": previous.get("temp", 0.0),
                "bus_voltage": previous.get("bus_voltage", 0.0),
            }

        telemetry = self._build_telemetry(ext_data)
        with self._telemetry_lock:
            self._latest_telemetry = telemetry

    def _ids_to_poll(self) -> tuple[int, ...]:
        self._poll_cycle += 1
        online_ids = [
            mid
            for mid, status in self._status_cache.items()
            if mid in self.angles and status.get("online", False)
        ]
        if self._poll_cycle % self._FULL_SCAN_EVERY == 0 or not online_ids:
            return self._motor_ids

        offline = [mid for mid in self._motor_ids if mid not in online_ids]
        if offline:
            return tuple(online_ids + [offline[self._poll_cycle % len(offline)]])
        return tuple(online_ids)

    def _build_telemetry(self, ext_data: Dict[int, Dict]) -> dict[int, dict]:
        now = time.time()
        monotonic_now = time.monotonic()
        dt = max(monotonic_now - self._prev_time, 0.001)
        self._prev_time = monotonic_now

        telemetry: dict[int, dict] = {}
        for motor_id in self._motor_ids:
            status = self._status_cache.get(motor_id, {})
            ext = ext_data.get(motor_id, {})
            position = float(status.get("angle", self.angles.get(motor_id, 0.0)))
            previous = self._prev_angles.get(motor_id, position)
            estimated_velocity = (position - previous) / dt if status.get("online") else 0.0
            self._prev_angles[motor_id] = position
            telemetry[motor_id] = {
                "position": position,
                "velocity": float(ext.get("velocity", estimated_velocity)),
                "current": float(ext.get("q_current", 0.0)),
                "torque": 0.0,
                "temperature": float(status.get("temp", 0.0)),
                "bus_voltage": float(status.get("bus_voltage", 0.0)),
                "online": bool(status.get("online", False)),
                "error_code": int(status.get("error_code", 0)),
                "fsm_state": int(status.get("fsm_state", 1 if self.enabled else 0)),
                "timestamp": now,
            }
        return telemetry

    def _fallback_telemetry(self) -> dict[int, dict]:
        now = time.time()
        return {
            motor_id: {
                "position": float(angle),
                "velocity": 0.0,
                "current": 0.0,
                "torque": 0.0,
                "temperature": 0.0,
                "bus_voltage": 0.0,
                "online": bool(self.connected),
                "error_code": 0,
                "fsm_state": 1 if self.enabled else 0,
                "timestamp": now,
            }
            for motor_id, angle in self.angles.items()
        }

    def _update_cached_target_angles(self, targets: dict[int, float]) -> None:
        with self._telemetry_lock:
            if not self._latest_telemetry:
                return
            for motor_id, angle in targets.items():
                telemetry = self._latest_telemetry.get(int(motor_id))
                if telemetry is not None:
                    telemetry["position"] = float(angle)
                    telemetry["timestamp"] = time.time()

    def _refresh_cached_enabled_state(self) -> None:
        with self._telemetry_lock:
            for telemetry in self._latest_telemetry.values():
                telemetry["fsm_state"] = 1 if self.enabled else 0


def create_driver(profile):
    return HandDriver(profile)
