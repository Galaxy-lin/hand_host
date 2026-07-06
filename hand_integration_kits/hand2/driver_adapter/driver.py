"""hand2 总线舵机驱动适配器（接入通用上位机版本）。

把 hand2 的 Lobot 风格总线舵机协议封装成通用上位机要求的 HandDriver 接口。
你只需要改这个文件，不需要改上位机源码。

接入向导第 4 步“电机通信适配”里建议填写：
  * 协议名 / 默认波特率：可只填 controller / 9600 一行；也可同时加 direct / 115200。
  * 默认协议：与电脑实际接线方式一致（controller 接控制器板；direct 直挂舵机总线）。
  * 力控能力：保持未勾选（hand2 协议没有按电流/力矩做力控的能力）。
  * 滑条/遥操作合并下发间隔：默认 30 ms 即可。
  * 遥测刷新间隔：默认 100 ms 即可。

约定：
  1. 上位机下发的所有角度都是“度”，顺序与关节表一致。
  2. motor_id 必须与关节表里填的电机 ID 一致（hand2 = 1..16）。
  3. connect 成功返回 True；失败返回 False，错误细节由抛出的异常承载。
  4. 遥测 position 字段返回“度”（已通过 motor_to_sim_calibration.json 反算）。
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import serial  # type: ignore
    HAS_SERIAL = True
except ImportError:
    serial = None  # type: ignore
    HAS_SERIAL = False


# -----------------------------------------------------------------------------
# 协议常量（Lobot 总线舵机）
# -----------------------------------------------------------------------------

LOBOT_CMD_SERVO_MOVE = 3
LOBOT_CMD_SERVO_POS_READ = 21

DIRECT_CMD_SERVO_MOVE_TIME_WRITE = 1
DIRECT_CMD_SERVO_POS_READ = 28

SERVO_MIN_POSITION = 200
SERVO_MAX_POSITION = 900
SERVO_ZERO_POSITION = 500

# 真实舵机 ID -> MuJoCo actuator 名称（与 hand2_motor_utils 保持一致）
SERVO_ID_TO_ACTUATOR: Dict[int, str] = {
    1: "th_cmc_act", 2: "th_axl_act", 3: "th_mcp_act", 4: "th_ipl_act",
    5: "if_mcp_act", 6: "if_rot_act", 7: "if_pip_act", 8: "if_dip_act",
    9: "mf_mcp_act", 10: "mf_rot_act", 11: "mf_pip_act", 12: "mf_dip_act",
    13: "rf_mcp_act", 14: "rf_rot_act", 15: "rf_pip_act", 16: "rf_dip_act",
}
SERVO_IDS: Tuple[int, ...] = tuple(SERVO_ID_TO_ACTUATOR.keys())

# 上位机协议下拉框的名字（接入向导可能直接填 "serial"），都映射到下面两套底层实现
PROTOCOL_CONTROLLER = "controller"
PROTOCOL_DIRECT = "direct"
PROTOCOL_ALIASES: Dict[str, str] = {
    "controller": PROTOCOL_CONTROLLER,
    "lobot": PROTOCOL_CONTROLLER,
    "lobot_controller": PROTOCOL_CONTROLLER,
    "serial": PROTOCOL_CONTROLLER,    # 向导只留 serial 时按控制器板对待
    "direct": PROTOCOL_DIRECT,
    "lobot_direct": PROTOCOL_DIRECT,
    "bus": PROTOCOL_DIRECT,
}

DEFAULT_BAUDRATES: Dict[str, int] = {
    PROTOCOL_CONTROLLER: 9600,
    PROTOCOL_DIRECT: 115200,
}

# 速度 -> 移动时间（毫秒）
_MOVE_TIME_MIN_MS = 80
_MOVE_TIME_MAX_MS = 1200
_HOME_MOVE_TIME_MS = 1500
_TELEOP_FRAME_PERIOD_S = 0.08
_TELEOP_MOVE_TIME_MS = max(20, int(_TELEOP_FRAME_PERIOD_S * 900.0))
_INTERACTIVE_MIN_MOVE_TIME_MS = 70
_INTERACTIVE_MAX_MOVE_TIME_MS = 800
_INTERACTIVE_DEG_PER_SEC = 220.0

# 接入向导默认值
_DEFAULT_MERGE_INTERVAL_MS = 30
_DEFAULT_TELEMETRY_INTERVAL_MS = 100


# -----------------------------------------------------------------------------
# 协议帧构造与解析
# -----------------------------------------------------------------------------

def _clamp_int(value: float, lo: int, hi: int) -> int:
    return max(lo, min(int(round(value)), hi))


def _move_time_from_speed(
    speed: float,
    minimum: int = _MOVE_TIME_MIN_MS,
    maximum: int = _MOVE_TIME_MAX_MS,
) -> int:
    speed = max(0.01, min(float(speed), 1.0))
    return int(round(maximum - speed * (maximum - minimum)))


def _build_controller_move_frame(
    servo_targets: Iterable[Tuple[int, int]],
    move_time_ms: int,
    min_position: int = SERVO_MIN_POSITION,
    max_position: int = SERVO_MAX_POSITION,
) -> bytearray:
    """控制器协议多舵机移动包：55 55 | len | 03 | n | t_lo t_hi | id pos_lo pos_hi ..."""

    targets = list(servo_targets)
    servo_count = max(1, min(len(targets), 254))
    move_time_ms = max(0, min(int(move_time_ms), 30000))

    frame = bytearray([0x55, 0x55])
    frame.append(servo_count * 3 + 5)
    frame.append(LOBOT_CMD_SERVO_MOVE)
    frame.append(servo_count)
    frame.extend(move_time_ms.to_bytes(2, "little", signed=False))
    for servo_id, position in targets[:servo_count]:
        frame.append(int(servo_id))
        clipped = _clamp_int(position, min_position, max_position)
        frame.extend(clipped.to_bytes(2, "little", signed=False))
    return frame


def _build_controller_read_frame(servo_ids: Sequence[int] = SERVO_IDS) -> bytearray:
    ids = list(servo_ids)
    frame = bytearray([0x55, 0x55])
    frame.append(len(ids) + 3)
    frame.append(LOBOT_CMD_SERVO_POS_READ)
    frame.append(len(ids))
    for servo_id in ids:
        frame.append(int(servo_id))
    return frame


def _direct_checksum(body: Sequence[int]) -> int:
    return (~sum(int(v) for v in body)) & 0xFF


def _build_direct_frame(servo_id: int, command: int, params: Sequence[int] = ()) -> bytearray:
    """单舵机直连协议：55 55 | id | len | cmd | params... | checksum"""

    length = len(params) + 3
    body = [int(servo_id), length, int(command), *[int(v) & 0xFF for v in params]]
    return bytearray([0x55, 0x55, *body, _direct_checksum(body)])


def _read_until_header(serial_handle, deadline: float) -> bool:
    last = None
    while time.monotonic() < deadline:
        current = serial_handle.read(1)
        if not current:
            continue
        if last == b"\x55" and current == b"\x55":
            return True
        last = current
    return False


def _read_controller_positions(
    serial_handle,
    servo_ids: Sequence[int] = SERVO_IDS,
    timeout_s: float = 2.0,
) -> Dict[int, int]:
    ids = list(servo_ids)
    serial_handle.reset_input_buffer()
    serial_handle.write(_build_controller_read_frame(ids))
    serial_handle.flush()

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _read_until_header(serial_handle, deadline):
            break
        length_byte = serial_handle.read(1)
        if not length_byte:
            continue
        payload = serial_handle.read(length_byte[0] - 1)
        if len(payload) != length_byte[0] - 1:
            continue
        packet = bytes([length_byte[0]]) + payload
        if len(packet) < 3 or packet[1] != LOBOT_CMD_SERVO_POS_READ:
            continue
        count = packet[2]
        expected_len = 3 + count * 3
        if len(packet) < expected_len:
            continue
        positions: Dict[int, int] = {}
        for i in range(count):
            base = 3 + i * 3
            sid = packet[base]
            pos = packet[base + 1] | (packet[base + 2] << 8)
            if sid in ids:
                positions[sid] = int(pos)
        if all(sid in positions for sid in ids):
            return positions

    raise TimeoutError("Timed out reading positions through the controller protocol.")


def _read_direct_packet(serial_handle, deadline: float) -> Tuple[int, int, List[int]]:
    while time.monotonic() < deadline:
        if not _read_until_header(serial_handle, deadline):
            break
        id_byte = serial_handle.read(1)
        length_byte = serial_handle.read(1)
        if not id_byte or not length_byte:
            continue
        servo_id = id_byte[0]
        length = length_byte[0]
        rest = serial_handle.read(length - 1)
        if len(rest) != length - 1 or len(rest) < 2:
            continue
        command = rest[0]
        params = list(rest[1:-1])
        checksum = rest[-1]
        body = [servo_id, length, command, *params]
        if _direct_checksum(body) != checksum:
            continue
        return servo_id, command, params
    raise TimeoutError("Timed out reading a direct bus-servo response.")


def _direct_write_and_read(
    serial_handle,
    servo_id: int,
    command: int,
    params: Sequence[int] = (),
    timeout_s: float = 0.5,
) -> List[int]:
    serial_handle.reset_input_buffer()
    serial_handle.write(_build_direct_frame(servo_id, command, params))
    serial_handle.flush()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rid, rcmd, rparams = _read_direct_packet(serial_handle, deadline)
        if rid == int(servo_id) and rcmd == int(command):
            return rparams
    raise TimeoutError(f"Timed out waiting for servo {servo_id} command {command}.")


def _signed_short_le(lo: int, hi: int) -> int:
    value = (int(lo) & 0xFF) | ((int(hi) & 0xFF) << 8)
    if value >= 0x8000:
        value -= 0x10000
    return value


def _read_direct_position(serial_handle, servo_id: int, timeout_s: float = 0.5) -> int:
    params = _direct_write_and_read(
        serial_handle, servo_id, DIRECT_CMD_SERVO_POS_READ, timeout_s=timeout_s
    )
    if len(params) < 2:
        raise ValueError(f"Servo {servo_id} returned too few position bytes: {params!r}")
    return _signed_short_le(params[0], params[1])


def _read_positions(
    serial_handle, protocol: str, servo_ids: Sequence[int] = SERVO_IDS
) -> Dict[int, int]:
    if protocol == PROTOCOL_CONTROLLER:
        return _read_controller_positions(serial_handle, servo_ids)
    if protocol == PROTOCOL_DIRECT:
        out: Dict[int, int] = {}
        for sid in servo_ids:
            out[int(sid)] = _read_direct_position(serial_handle, int(sid))
            time.sleep(0.02)
        return out
    raise ValueError(f"Unknown protocol: {protocol}")


def _write_move(
    serial_handle,
    protocol: str,
    servo_targets: Iterable[Tuple[int, int]],
    move_time_ms: int,
    min_position: int = SERVO_MIN_POSITION,
    max_position: int = SERVO_MAX_POSITION,
) -> None:
    targets = list(servo_targets)
    if protocol == PROTOCOL_CONTROLLER:
        frame = _build_controller_move_frame(
            targets, move_time_ms, min_position, max_position
        )
        serial_handle.reset_output_buffer()
        serial_handle.write(frame)
        serial_handle.flush()
        return
    if protocol == PROTOCOL_DIRECT:
        clipped_time = max(0, min(int(move_time_ms), 30000))
        for servo_id, position in targets:
            clipped_pos = _clamp_int(position, min_position, max_position)
            params = [
                clipped_pos & 0xFF,
                (clipped_pos >> 8) & 0xFF,
                clipped_time & 0xFF,
                (clipped_time >> 8) & 0xFF,
            ]
            serial_handle.write(
                _build_direct_frame(servo_id, DIRECT_CMD_SERVO_MOVE_TIME_WRITE, params)
            )
            serial_handle.flush()
            time.sleep(0.005)
        return
    raise ValueError(f"Unknown protocol: {protocol}")


# -----------------------------------------------------------------------------
# 标定加载与角度换算
# -----------------------------------------------------------------------------

def _load_motor_to_sim_calibration(path: Path) -> Dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "motor_to_sim_calibration_v1":
        raise ValueError(f"Unsupported calibration format in {path}")
    joints = payload.get("joints")
    if not isinstance(joints, dict):
        raise ValueError(f"Calibration file {path} missing joints object")
    for sid, actuator_name in SERVO_ID_TO_ACTUATOR.items():
        entry = joints.get(str(sid))
        if not isinstance(entry, dict):
            raise ValueError(f"Calibration missing servo ID {sid}")
        if entry.get("actuator") != actuator_name:
            raise ValueError(
                f"Servo ID {sid} maps to {entry.get('actuator')!r}, expected {actuator_name!r}"
            )
    return payload


def _angles_deg_to_servo_targets(
    angles_deg: Sequence[float],
    actuator_names: Sequence[str],
    calibration: Mapping[str, object],
    gain: float = 1.0,
    min_position: int = SERVO_MIN_POSITION,
    max_position: int = SERVO_MAX_POSITION,
) -> List[Tuple[int, int]]:
    """模型角度(度) → 舵机原始位置(0..1000)。

    标定使用的正变换：
        sim_rad = sign * radians((position - center) * scale / 1000) * gain + offset_rad
    反过来求 position：
        position = center + degrees((sim_rad - offset_rad) / (sign * gain)) * 1000 / scale
    """

    by_name = {name: math.radians(float(a)) for name, a in zip(actuator_names, angles_deg)}
    joints = calibration["joints"]  # type: ignore[index]
    targets: List[Tuple[int, int]] = []
    for sid in SERVO_IDS:
        actuator = SERVO_ID_TO_ACTUATOR[sid]
        entry = joints[str(sid)]  # type: ignore[index]
        center = float(entry.get("center_position", SERVO_ZERO_POSITION))
        scale = float(entry.get("scale_degrees_per_1000", 240.0))
        sign = float(entry.get("sign", 1.0))
        joint_gain = float(entry.get("gain", 1.0))
        offset_rad = float(entry.get("offset_rad", 0.0))
        if scale <= 0 or abs(sign * joint_gain) < 1e-12:
            raise ValueError(f"Servo ID {sid} has invalid calibration parameters")

        target_rad = by_name.get(actuator, offset_rad)
        target_rad = offset_rad + (target_rad - offset_rad) * float(gain)
        deg = math.degrees((target_rad - offset_rad) / (sign * joint_gain))
        position = center + deg * 1000.0 / scale
        targets.append((sid, _clamp_int(position, min_position, max_position)))
    return targets


def _servo_position_to_angle_deg(raw_position: int, entry: Mapping[str, object]) -> float:
    center = float(entry.get("center_position", SERVO_ZERO_POSITION))
    scale = float(entry.get("scale_degrees_per_1000", 240.0))
    sign = float(entry.get("sign", 1.0))
    gain = float(entry.get("gain", 1.0))
    offset_rad = float(entry.get("offset_rad", 0.0))
    sim_rad = (
        sign * math.radians((float(raw_position) - center) * scale / 1000.0) * gain
        + offset_rad
    )
    return math.degrees(sim_rad)


# -----------------------------------------------------------------------------
# HandDriver 主类
# -----------------------------------------------------------------------------

class HandDriver:
    """hand2 总线舵机统一驱动。

    用法：
        driver = HandDriver(profile)
        driver.connect("COM3", 9600, protocol="controller")
        driver.set_all_joint_angles([...])     # profile.num_joints 路角度（度）
        driver.get_latest_telemetry()
        driver.cleanup()
    """

    def __init__(self, profile):
        self.profile = profile
        self.config: Dict[str, Any] = (
            getattr(profile, "motor_adapter_config", {}) or {}
        )
        self.connected = False
        self.enabled = False
        self.protocol = self._normalize_protocol(
            getattr(profile, "default_protocol", PROTOCOL_CONTROLLER)
        )

        # 上位机下发用的角度缓存（度）
        self.angles: List[float] = [
            joint.default_angle
            for finger in profile.finger_configs
            for joint in finger.joints
        ]

        # 接入向导可调的两个间隔
        merge_interval_ms = self.config.get(
            "merge_interval_ms",
            self.config.get("command_debounce_ms", _DEFAULT_MERGE_INTERVAL_MS),
        )
        telemetry_interval_ms = self.config.get(
            "telemetry_interval_ms",
            self.config.get("telemetry_period_ms", _DEFAULT_TELEMETRY_INTERVAL_MS),
        )
        self._merge_interval_s = max(
            0.0,
            float(merge_interval_ms) / 1000.0,
        )
        self._telemetry_interval_s = max(
            0.02,
            float(telemetry_interval_ms) / 1000.0,
        )
        self.teleop_move_time_ms = int(
            self.config.get("teleop_move_time_ms", _TELEOP_MOVE_TIME_MS)
        )
        self._position_polling_enabled = bool(
            self.config.get("position_polling_enabled", False)
        )

        # 串口与遥测线程
        self._serial = None
        self._serial_lock = threading.RLock()
        self._latest_telemetry: Dict[int, Dict] = {}
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()

        # 合并下发：滑条/遥操作的高频更新先写到 pending，到期一次性发硬件
        self._pending_lock = threading.Lock()
        self._pending_angles: Optional[List[float]] = None
        self._pending_speed: float = 0.5
        self._pending_move_time_ms: Optional[int] = None
        self._debounce_timer: Optional[threading.Timer] = None

        # 标定（模型角度 ↔ 舵机位置）。允许 profile 没配置文件时退化为线性默认。
        self._calibration: Dict[str, object] = self._load_calibration()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def connect(self, port: str, baudrate: int, protocol: str | None = None) -> bool:
        if not HAS_SERIAL:
            raise RuntimeError("pyserial 未安装，无法连接 hand2 串口")
        if not port:
            raise ValueError("port 为空，请在连接页选择正确的串口")

        self.disconnect()
        self.protocol = self._normalize_protocol(
            protocol or getattr(self.profile, "default_protocol", PROTOCOL_CONTROLLER)
        )
        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=int(baudrate),
                timeout=0.05,
                write_timeout=0.1,
            )
        except Exception:
            self._serial = None
            self.connected = False
            raise

        self.connected = True
        self._start_polling()
        return True

    def disconnect(self) -> None:
        self._cancel_pending()
        self._stop_polling()
        if self._serial is not None:
            try:
                if self._serial.is_open:
                    self._serial.close()
            except Exception:
                pass
        self._serial = None
        self.connected = False

    def cleanup(self) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # 使能 / 力控
    # ------------------------------------------------------------------

    def enable_all(self, enable: bool = True) -> None:
        # hand2 的两个底层协议都是“下发位置即驱动”，没有独立的使能命令；
        # 这里只更新内部状态，使遥测里的 fsm_state 反映用户意图。
        self.enabled = bool(enable)

    def set_zero_point(self, motor_id: int) -> None:
        # hand2 的零点应通过 motor_to_sim_calibration.json 的 center_position 写入。
        return

    def set_max_current(self, motor_id: int, value_q15: int) -> None:
        # hand2 协议没有按电流的力控保护，向导“力控能力”勾选框应保持未勾选。
        return

    # ------------------------------------------------------------------
    # 角度下发
    # ------------------------------------------------------------------

    def set_motor_angle(
        self, motor_id: int, angle_deg: float, speed: float = 0.5
    ) -> None:
        idx = self.profile.motor_id_to_joint_index.get(int(motor_id))
        if idx is None:
            return
        with self._pending_lock:
            angles = list(self._pending_angles) if self._pending_angles else list(self.angles)
            angles[idx] = float(angle_deg)
            self._pending_angles = angles
            self._pending_speed = float(speed)
            self._pending_move_time_ms = self._interactive_move_time_for_angles(angles)
        self._schedule_flush()

    def set_angles(self, angles_dict: Dict[int, float]) -> None:
        with self._pending_lock:
            angles = list(self._pending_angles) if self._pending_angles else list(self.angles)
            for motor_id, angle in angles_dict.items():
                idx = self.profile.motor_id_to_joint_index.get(int(motor_id))
                if idx is not None:
                    angles[idx] = float(angle)
            self._pending_angles = angles
            self._pending_speed = 0.5
            self._pending_move_time_ms = self._interactive_move_time_for_angles(angles)
        self._schedule_flush()

    def set_all_joint_angles(
        self, angles_list: List[float], move_time_ms: Optional[int] = None
    ) -> None:
        if len(angles_list) != self.profile.num_joints:
            raise ValueError(
                f"需要 {self.profile.num_joints} 个角度，收到 {len(angles_list)} 个"
            )
        angles = [float(v) for v in angles_list]
        with self._pending_lock:
            self._pending_angles = angles
            self._pending_speed = 0.5
            self._pending_move_time_ms = (
                int(move_time_ms)
                if move_time_ms is not None
                else self._interactive_move_time_for_angles(angles)
            )
        self._schedule_flush()

    def home_all_motors(
        self,
        target_angles: Dict[int, float] | None = None,
        speed: float = 0.3,
    ) -> None:
        # home 是一次性大动作，不走合并下发：取消未发的 pending，立刻整手发出。
        self._cancel_pending()

        if target_angles:
            angles = list(self.angles)
            for motor_id, angle in target_angles.items():
                idx = self.profile.motor_id_to_joint_index.get(int(motor_id))
                if idx is not None:
                    angles[idx] = float(angle)
            self._send_angles_now(angles, speed=speed, move_time_ms=_HOME_MOVE_TIME_MS)
            return

        # 没有显式目标 → 直接走标定中心位置（等价硬件零位）。
        if self._calibration:
            targets = [
                (
                    sid,
                    _clamp_int(
                        float(
                            self._calibration["joints"][str(sid)].get(  # type: ignore[index]
                                "center_position", SERVO_ZERO_POSITION
                            )
                        ),
                        SERVO_MIN_POSITION,
                        SERVO_MAX_POSITION,
                    ),
                )
                for sid in SERVO_IDS
            ]
            self._write_targets(targets, move_time_ms=_HOME_MOVE_TIME_MS)
            self.angles = [
                joint.default_angle
                for finger in self.profile.finger_configs
                for joint in finger.joints
            ]
            return

        defaults = [
            joint.default_angle
            for finger in self.profile.finger_configs
            for joint in finger.joints
        ]
        self._send_angles_now(defaults, speed=speed, move_time_ms=_HOME_MOVE_TIME_MS)

    # ------------------------------------------------------------------
    # 遥测
    # ------------------------------------------------------------------

    def get_latest_telemetry(self) -> Dict[int, Dict]:
        if self._latest_telemetry:
            return {mid: dict(data) for mid, data in self._latest_telemetry.items()}
        return self._fallback_telemetry()

    def get_status(self, motor_id: int) -> Dict:
        return self.get_latest_telemetry().get(int(motor_id), {"online": False})

    # ------------------------------------------------------------------
    # 合并下发
    # ------------------------------------------------------------------

    def _schedule_flush(self) -> None:
        # merge_interval_ms <= 0 时直接同步发；否则用 Timer 合并到下一帧统一发。
        if self._merge_interval_s <= 0.0:
            self._flush_pending()
            return
        with self._pending_lock:
            if self._debounce_timer is not None and self._debounce_timer.is_alive():
                return
            timer = threading.Timer(self._merge_interval_s, self._flush_pending)
            timer.daemon = True
            self._debounce_timer = timer
        timer.start()

    def _flush_pending(self) -> None:
        with self._pending_lock:
            angles = self._pending_angles
            speed = self._pending_speed
            move_time_ms = self._pending_move_time_ms
            self._pending_angles = None
            self._pending_move_time_ms = None
            self._debounce_timer = None
        if angles is None:
            return
        self._send_angles_now(angles, speed=speed, move_time_ms=move_time_ms)

    def _cancel_pending(self) -> None:
        with self._pending_lock:
            timer = self._debounce_timer
            self._pending_angles = None
            self._pending_move_time_ms = None
            self._debounce_timer = None
        if timer is not None:
            timer.cancel()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _is_serial_open(self) -> bool:
        return bool(self._serial and getattr(self._serial, "is_open", False))

    def _normalize_protocol(self, name: Any) -> str:
        if not name:
            return PROTOCOL_CONTROLLER
        return PROTOCOL_ALIASES.get(str(name).strip().lower(), PROTOCOL_CONTROLLER)

    def _load_calibration(self) -> Dict[str, object]:
        path = getattr(self.profile, "motor_to_sim_calibration_path", None)
        if not path:
            return {}
        try:
            return _load_motor_to_sim_calibration(Path(path))
        except FileNotFoundError:
            return {}
        except Exception:
            # 标定文件存在但格式异常时退化为线性默认，避免连接整体失败。
            return {}

    def _interactive_move_time_for_angles(self, target_angles: Sequence[float]) -> int:
        if len(target_angles) != len(self.angles):
            return self.teleop_move_time_ms
        max_delta = max(
            (abs(float(new) - float(old)) for new, old in zip(target_angles, self.angles)),
            default=0.0,
        )
        if max_delta <= 1e-6:
            return _INTERACTIVE_MIN_MOVE_TIME_MS
        move_time = int(round(1000.0 * max_delta / _INTERACTIVE_DEG_PER_SEC))
        return max(
            _INTERACTIVE_MIN_MOVE_TIME_MS,
            min(_INTERACTIVE_MAX_MOVE_TIME_MS, move_time),
        )

    def _send_angles_now(
        self,
        angles_deg: Sequence[float],
        speed: float,
        move_time_ms: Optional[int] = None,
    ) -> None:
        angles = [float(v) for v in angles_deg]
        if move_time_ms is None:
            move_time_ms = self._interactive_move_time_for_angles(angles)
        # 即使没连串口也更新 angles 缓存：上位机仍能读到最新目标值
        self.angles = angles
        if not self._is_serial_open():
            return
        targets = self._angles_to_targets(angles)
        self._write_targets(targets, move_time_ms=int(move_time_ms))

    def _angles_to_targets(self, angles_deg: Sequence[float]) -> List[Tuple[int, int]]:
        if self._calibration:
            return _angles_deg_to_servo_targets(
                angles_deg,
                self.profile.actuator_names,
                self._calibration,
            )
        # 没有标定时退化为线性映射：default 角度 -> SERVO_ZERO_POSITION，1° ≈ 4.17 单位
        targets: List[Tuple[int, int]] = []
        for idx, angle in enumerate(angles_deg):
            motor_id = self.profile.joint_index_to_motor_id[idx]
            position = SERVO_ZERO_POSITION + float(angle) * 1000.0 / 240.0
            targets.append(
                (motor_id, _clamp_int(position, SERVO_MIN_POSITION, SERVO_MAX_POSITION))
            )
        return targets

    def _write_targets(
        self, targets: Sequence[Tuple[int, int]], move_time_ms: int
    ) -> None:
        if not self._is_serial_open():
            return
        with self._serial_lock:
            try:
                _write_move(
                    self._serial,
                    self.protocol,
                    targets,
                    move_time_ms=move_time_ms,
                    min_position=SERVO_MIN_POSITION,
                    max_position=SERVO_MAX_POSITION,
                )
            except Exception:
                # 单次写失败不应让上层崩溃；下一次 set_* 会再次尝试。
                pass

    # ------------------------------------------------------------------
    # 后台位置轮询
    # ------------------------------------------------------------------

    def _start_polling(self) -> None:
        self._stop_polling()
        if not self._position_polling_enabled:
            self._latest_telemetry = self._fallback_telemetry()
            return
        self._poll_stop.clear()
        thread = threading.Thread(
            target=self._poll_loop, name="hand2-poll", daemon=True
        )
        self._poll_thread = thread
        thread.start()

    def _stop_polling(self) -> None:
        self._poll_stop.set()
        thread = self._poll_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._poll_thread = None

    def _poll_loop(self) -> None:
        while not self._poll_stop.is_set():
            if not self._is_serial_open():
                break
            try:
                with self._serial_lock:
                    positions = _read_positions(self._serial, self.protocol, SERVO_IDS)
                self._update_telemetry_from_positions(positions)
            except Exception:
                self._mark_all_offline()
            self._poll_stop.wait(self._telemetry_interval_s)

    def _update_telemetry_from_positions(self, positions: Dict[int, int]) -> None:
        joints_cal = (
            self._calibration.get("joints", {})  # type: ignore[union-attr]
            if self._calibration
            else {}
        )
        now = time.time()
        telemetry: Dict[int, Dict] = {}
        for sid in SERVO_IDS:
            raw = positions.get(sid)
            online = raw is not None
            entry = joints_cal.get(str(sid), {}) if isinstance(joints_cal, dict) else {}
            if online and entry:
                position_deg = _servo_position_to_angle_deg(int(raw), entry)
            elif online:
                position_deg = (float(raw) - SERVO_ZERO_POSITION) * 240.0 / 1000.0
            else:
                position_deg = 0.0
            telemetry[sid] = {
                "position": float(position_deg),
                "raw_position": int(raw) if online else 0,
                "velocity": 0.0,
                "current": 0.0,
                "torque": 0.0,
                "temperature": 0.0,
                "bus_voltage": 0.0,
                "online": online,
                "error_code": 0,
                "fsm_state": 1 if self.enabled else 0,
                "timestamp": now,
            }
        self._latest_telemetry = telemetry

    def _mark_all_offline(self) -> None:
        now = time.time()
        snapshot = self._latest_telemetry
        new_telemetry: Dict[int, Dict] = {}
        for sid in SERVO_IDS:
            previous = snapshot.get(sid, {}) if snapshot else {}
            new_telemetry[sid] = {
                "position": float(previous.get("position", 0.0)),
                "raw_position": int(previous.get("raw_position", 0)),
                "velocity": 0.0,
                "current": 0.0,
                "torque": 0.0,
                "temperature": 0.0,
                "bus_voltage": 0.0,
                "online": False,
                "error_code": 0,
                "fsm_state": 0,
                "timestamp": now,
            }
        self._latest_telemetry = new_telemetry

    def _fallback_telemetry(self) -> Dict[int, Dict]:
        # 还没收到任何遥测时（例如刚连接、首帧未到），用上位机最后下发的目标角度兜底。
        now = time.time()
        out: Dict[int, Dict] = {}
        for index, angle in enumerate(self.angles):
            motor_id = self.profile.joint_index_to_motor_id[index]
            out[motor_id] = {
                "position": float(angle),
                "raw_position": 0,
                "velocity": 0.0,
                "current": 0.0,
                "torque": 0.0,
                "temperature": 0.0,
                "bus_voltage": 0.0,
                "online": self.connected,
                "error_code": 0,
                "fsm_state": 1 if self.enabled else 0,
                "timestamp": now,
            }
        return out


def create_driver(profile) -> HandDriver:
    """通用上位机统一入口：根据 profile 实例化驱动。"""

    return HandDriver(profile)
