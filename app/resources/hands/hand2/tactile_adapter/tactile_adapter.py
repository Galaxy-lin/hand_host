"""hand2 触觉传感器适配器（接入通用上位机版本）。

把 hand2 的触觉串口协议解析成上位机要求的三种标准格式之一：

  * raw          —— 6×8 矩阵，对应旧版 AA 55 协议。
  * finger+palm  —— 4 指 × 3 段 + 8 个掌心区域，由整手帧聚合得到。
  * patches      —— 整手分区矩阵（palm / if_tip / if_distal_pad / if_mid_pad /
                    mf_* / rf_* / th_pad / th_tip），适合高密度触觉皮肤。

接入向导第 5 步建议填写：
  * 触觉来源：       Python 触觉适配器
  * 默认波特率：     460800（整手协议） 或 115200（旧版 6×8 协议）
  * 数值最大值：     255.0（本适配器原样返回 0..255 的 ADC 字节，由上位机做归一化）
  * 基线采集帧数：   30
  * 接触阈值：       0.00（适配器不做阈值，上位机统一扣除）
  * 适配器输出格式： 适配器返回 patches 整手分区（默认）；如硬件只产生旧版 6×8 帧，
                    可选 “适配器返回 raw 6x8 矩阵”；选 finger+palm 时本适配器会
                    把 patches 自动聚合为 4×3 + 8 的形式。

适配器只做协议解析与（可选的）格式聚合，不做基线扣除、阈值、归一化、
显示着色，这些都是上位机的统一处理逻辑。
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

try:
    import serial  # type: ignore
    HAS_SERIAL = True
except ImportError:
    serial = None  # type: ignore
    HAS_SERIAL = False


# -----------------------------------------------------------------------------
# 协议常量
# -----------------------------------------------------------------------------

WHOLE_HAND_HEADER = bytes([0xAA, 0xBB])
SEGMENT_HEADER = bytes([0xAA, 0x55])

# 段尺寸（与上位机 hand-specific tactile adapter 保持一致）
FINGER_ROWS, FINGER_COLS = 14, 7
THUMB_ROWS, THUMB_COLS = 10, 7
PALM_ROWS, PALM_COLS = 16, 16

# 整手帧片段顺序及尺寸：(name, rows, cols)
WHOLE_HAND_SEGMENTS = (
    ("ring", FINGER_ROWS, FINGER_COLS),
    ("middle", FINGER_ROWS, FINGER_COLS),
    ("index", FINGER_ROWS, FINGER_COLS),
    ("thumb", THUMB_ROWS, THUMB_COLS),
    ("palm", PALM_ROWS, PALM_COLS),
)

WHOLE_HAND_FRAME_LENGTH = 2 + sum(2 + r * c for _, r, c in WHOLE_HAND_SEGMENTS)  # 632

# 旧版 AA 55 协议
LEGACY_HEADER = bytes([0xAA, 0x55])
LEGACY_ROWS, LEGACY_COLS = 6, 8
LEGACY_FRAME_LENGTH = 2 + LEGACY_ROWS * LEGACY_COLS  # 50

# 手指 / 拇指内部按行划分的子片区顺序，与上位机默认一致
FINGER_ROW_ORDER_DEFAULT = "tip-distal-mid"
THUMB_ROW_ORDER_DEFAULT = "tip-pad"

# 适配器输出格式选项
OUTPUT_NATIVE = "native"           # 原帧：整手协议 → patches，旧版协议 → raw
OUTPUT_PATCHES = "patches"
OUTPUT_RAW = "raw"
OUTPUT_FINGER_PALM = "finger_palm"

_OUTPUT_ALIASES: Dict[str, str] = {
    "": OUTPUT_NATIVE,
    "native": OUTPUT_NATIVE,
    "auto": OUTPUT_NATIVE,
    "adapter_dict": OUTPUT_NATIVE,
    "patches": OUTPUT_PATCHES,
    "patch": OUTPUT_PATCHES,
    "whole_hand": OUTPUT_PATCHES,
    "whole-hand": OUTPUT_PATCHES,
    "raw": OUTPUT_RAW,
    "raw_6x8": OUTPUT_RAW,
    "raw6x8": OUTPUT_RAW,
    "6x8": OUTPUT_RAW,
    "finger_palm": OUTPUT_FINGER_PALM,
    "finger+palm": OUTPUT_FINGER_PALM,
    "fingerpalm": OUTPUT_FINGER_PALM,
    "finger": OUTPUT_FINGER_PALM,
}

# 串口线协议选项：决定 _FrameParser 接受哪种字节流。
# 仅在用户明确把硬件标记为旧版 6×8 时才放开 AA 55 解析。
PROTOCOL_WHOLE_HAND = "whole_hand"  # 仅接受 AA BB 整手帧（hand2 硬件默认）
PROTOCOL_LEGACY = "legacy"          # 仅接受 AA 55 旧版 6×8 帧
PROTOCOL_AUTO = "auto"              # 同时接受两种协议（兼容旧配置）

_PROTOCOL_ALIASES: Dict[str, str] = {
    "": PROTOCOL_WHOLE_HAND,
    "auto": PROTOCOL_AUTO,
    "any": PROTOCOL_AUTO,
    "both": PROTOCOL_AUTO,
    "whole_hand": PROTOCOL_WHOLE_HAND,
    "wholehand": PROTOCOL_WHOLE_HAND,
    "whole-hand": PROTOCOL_WHOLE_HAND,
    "aa_bb": PROTOCOL_WHOLE_HAND,
    "aabb": PROTOCOL_WHOLE_HAND,
    "patches": PROTOCOL_WHOLE_HAND,
    "patch": PROTOCOL_WHOLE_HAND,
    "finger_palm": PROTOCOL_WHOLE_HAND,
    "finger+palm": PROTOCOL_WHOLE_HAND,
    "fingerpalm": PROTOCOL_WHOLE_HAND,
    "finger": PROTOCOL_WHOLE_HAND,
    "adapter_dict": PROTOCOL_WHOLE_HAND,
    "legacy": PROTOCOL_LEGACY,
    "raw": PROTOCOL_LEGACY,
    "raw_6x8": PROTOCOL_LEGACY,
    "raw6x8": PROTOCOL_LEGACY,
    "6x8": PROTOCOL_LEGACY,
    "aa55": PROTOCOL_LEGACY,
    "aa_55": PROTOCOL_LEGACY,
}


# -----------------------------------------------------------------------------
# 字节流 → 矩阵换算
# -----------------------------------------------------------------------------

def _bytes_to_matrix(data: bytes, rows: int, cols: int) -> List[List[float]]:
    """按 reshape((cols, rows)).T 的方式还原 rows×cols 矩阵。"""

    if len(data) != rows * cols:
        raise ValueError(f"expected {rows * cols} bytes, got {len(data)}")
    matrix: List[List[float]] = [[0.0] * cols for _ in range(rows)]
    for c in range(cols):
        base = c * rows
        for r in range(rows):
            matrix[r][c] = float(data[base + r])
    return matrix


def _split_finger_patches(
    matrix: List[List[float]], prefix: str, row_order: str
) -> Dict[str, List[List[float]]]:
    if row_order == "tip-distal-mid":
        tip = [row[:] for row in matrix[0:6]]
        distal = [row[:] for row in matrix[6:10]]
        mid = [row[:] for row in matrix[10:14]]
    elif row_order == "mid-distal-tip":
        mid = [row[:] for row in matrix[0:4]]
        distal = [row[:] for row in matrix[4:8]]
        tip = [row[:] for row in matrix[8:14]]
    else:
        raise ValueError(f"unknown finger_row_order: {row_order}")
    return {
        f"{prefix}_mid_pad": mid,
        f"{prefix}_distal_pad": distal,
        f"{prefix}_tip": tip,
    }


def _split_thumb_patches(
    matrix: List[List[float]], row_order: str
) -> Dict[str, List[List[float]]]:
    if row_order == "tip-pad":
        tip = [row[:] for row in matrix[0:6]]
        pad = [row[:] for row in matrix[6:10]]
    elif row_order == "pad-tip":
        pad = [row[:] for row in matrix[0:4]]
        tip = [row[:] for row in matrix[4:10]]
    else:
        raise ValueError(f"unknown thumb_row_order: {row_order}")
    return {"th_pad": pad, "th_tip": tip}


# -----------------------------------------------------------------------------
# 增量帧解析器
# -----------------------------------------------------------------------------

class _FrameParser:
    """累积串口字节，先尝试整手帧，再降级旧版 6×8 帧。"""

    _MAX_BUFFER_BYTES = 4 * WHOLE_HAND_FRAME_LENGTH

    def __init__(
        self,
        finger_row_order: str = FINGER_ROW_ORDER_DEFAULT,
        thumb_row_order: str = THUMB_ROW_ORDER_DEFAULT,
        protocol: str = PROTOCOL_WHOLE_HAND,
    ):
        self.finger_row_order = finger_row_order
        self.thumb_row_order = thumb_row_order
        self.protocol = protocol
        self._buffer = bytearray()

    def clear(self) -> None:
        self._buffer.clear()

    def feed(self, data: bytes) -> List[Dict]:
        if data:
            self._buffer.extend(data)
        frames: List[Dict] = []
        while True:
            frame = self._try_parse_one()
            if frame is None:
                break
            frames.append(frame)
        if len(self._buffer) > self._MAX_BUFFER_BYTES:
            tail = self._buffer[-WHOLE_HAND_FRAME_LENGTH:]
            self._buffer.clear()
            self._buffer.extend(tail)
        return frames

    def _try_parse_one(self) -> Optional[Dict]:
        accept_whole = self.protocol in (PROTOCOL_WHOLE_HAND, PROTOCOL_AUTO)
        accept_legacy = self.protocol in (PROTOCOL_LEGACY, PROTOCOL_AUTO)

        whole_idx = self._buffer.find(WHOLE_HAND_HEADER) if accept_whole else -1
        legacy_idx = self._buffer.find(LEGACY_HEADER) if accept_legacy else -1

        if whole_idx >= 0:
            available = len(self._buffer) - whole_idx
            if available >= WHOLE_HAND_FRAME_LENGTH:
                candidate = bytes(
                    self._buffer[whole_idx:whole_idx + WHOLE_HAND_FRAME_LENGTH]
                )
                try:
                    parsed = self._parse_whole_hand(candidate)
                except ValueError:
                    del self._buffer[: whole_idx + 1]
                    return None
                if whole_idx > 0:
                    del self._buffer[:whole_idx]
                del self._buffer[:WHOLE_HAND_FRAME_LENGTH]
                return parsed

            if (
                accept_legacy
                and 0 <= legacy_idx < whole_idx
                and len(self._buffer) - legacy_idx >= LEGACY_FRAME_LENGTH
            ):
                return self._consume_legacy(legacy_idx)
            return None

        if (
            accept_legacy
            and legacy_idx >= 0
            and len(self._buffer) - legacy_idx >= LEGACY_FRAME_LENGTH
        ):
            return self._consume_legacy(legacy_idx)
        return None

    def _consume_legacy(self, idx: int) -> Dict:
        if idx > 0:
            del self._buffer[:idx]
        frame = bytes(self._buffer[:LEGACY_FRAME_LENGTH])
        del self._buffer[:LEGACY_FRAME_LENGTH]
        payload = frame[2:]
        matrix = [
            [float(payload[r * LEGACY_COLS + c]) for c in range(LEGACY_COLS)]
            for r in range(LEGACY_ROWS)
        ]
        return {"raw": matrix}

    def _parse_whole_hand(self, frame: bytes) -> Dict:
        if len(frame) != WHOLE_HAND_FRAME_LENGTH:
            raise ValueError(f"expected {WHOLE_HAND_FRAME_LENGTH} bytes")
        if not frame.startswith(WHOLE_HAND_HEADER):
            raise ValueError("frame does not start with AA BB")

        offset = 2
        segments: Dict[str, bytes] = {}
        for name, rows, cols in WHOLE_HAND_SEGMENTS:
            seg_len = 2 + rows * cols
            seg = frame[offset:offset + seg_len]
            if len(seg) != seg_len:
                raise ValueError(f"{name} segment incomplete")
            if seg[:2] != SEGMENT_HEADER:
                raise ValueError(
                    f"{name} segment header mismatch: got {seg[:2].hex(' ').upper()}"
                )
            segments[name] = bytes(seg[2:])
            offset += seg_len

        ring_mat = _bytes_to_matrix(segments["ring"], FINGER_ROWS, FINGER_COLS)
        middle_mat = _bytes_to_matrix(segments["middle"], FINGER_ROWS, FINGER_COLS)
        index_mat = _bytes_to_matrix(segments["index"], FINGER_ROWS, FINGER_COLS)
        thumb_mat = _bytes_to_matrix(segments["thumb"], THUMB_ROWS, THUMB_COLS)
        palm_mat = _bytes_to_matrix(segments["palm"], PALM_ROWS, PALM_COLS)

        patches: Dict[str, List[List[float]]] = {"palm": palm_mat}
        patches.update(_split_finger_patches(index_mat, "if", self.finger_row_order))
        patches.update(_split_finger_patches(middle_mat, "mf", self.finger_row_order))
        patches.update(_split_finger_patches(ring_mat, "rf", self.finger_row_order))
        patches.update(_split_thumb_patches(thumb_mat, self.thumb_row_order))
        return {"patches": patches}


# -----------------------------------------------------------------------------
# 格式聚合：把整手 patches 转成 raw 6×8 / finger+palm 形式
# -----------------------------------------------------------------------------

def _matrix_mean(matrix: List[List[float]]) -> float:
    if not matrix:
        return 0.0
    flat = [float(v) for row in matrix for v in row]
    return sum(flat) / max(len(flat), 1)


def _patches_to_finger_palm(
    patches: Dict[str, List[List[float]]],
) -> Dict[str, Any]:
    """把整手 patches 聚合成 {"finger": 4×3, "palm": 8}。

    手指顺序：拇指 / 食指 / 中指 / 无名指。
    每根手指 3 段顺序：mid_pad / distal_pad / tip。
    掌心 8 维：把 16×16 掌心阵按 8 列分组、每列对所有行求均值。
    """

    finger_specs = (
        ("th_pad", "th_pad", "th_tip"),
        ("if_mid_pad", "if_distal_pad", "if_tip"),
        ("mf_mid_pad", "mf_distal_pad", "mf_tip"),
        ("rf_mid_pad", "rf_distal_pad", "rf_tip"),
    )
    finger: List[List[float]] = []
    for spec in finger_specs:
        finger.append([_matrix_mean(patches.get(name, [])) for name in spec])

    palm_mat = patches.get("palm", [])
    palm: List[float] = [0.0] * 8
    if palm_mat:
        rows = len(palm_mat)
        cols = len(palm_mat[0]) if palm_mat else 0
        if rows > 0 and cols > 0:
            # 每 cols/8 列聚成一段（cols=16 时每 2 列一段）
            seg_width = max(1, cols // 8)
            for seg in range(8):
                lo = seg * seg_width
                hi = (seg + 1) * seg_width if seg < 7 else cols
                values = [palm_mat[r][c] for r in range(rows) for c in range(lo, hi)]
                palm[seg] = sum(values) / max(len(values), 1) if values else 0.0
    return {"finger": finger, "palm": palm}


def _patches_to_raw_6x8(
    patches: Dict[str, List[List[float]]],
) -> List[List[float]]:
    """把整手 patches 聚合成上位机旧版 6×8 矩阵。

    映射规则（与 tactile_service._map_to_finger_palm 兼容）：
        Row 0 → 拇指 3 段 + 5 个 0；Row 1..3 → 食/中/无名 3 段 + 5 个 0；
        Row 4..5 → 掌心两行（每行 8 段）。
    """

    fp = _patches_to_finger_palm(patches)
    finger = fp["finger"]
    palm = fp["palm"]
    raw = [[0.0] * 8 for _ in range(6)]
    for r in range(4):
        seg = finger[r] if r < len(finger) else [0.0, 0.0, 0.0]
        # 把 3 段铺到 8 列：col 0..2 mid，col 3..4 distal，col 5..7 tip
        for c in range(3):
            raw[r][c] = float(seg[0])
        for c in range(3, 5):
            raw[r][c] = float(seg[1])
        for c in range(5, 8):
            raw[r][c] = float(seg[2])
    for c in range(8):
        raw[4][c] = float(palm[c])
        raw[5][c] = float(palm[c])
    return raw


# -----------------------------------------------------------------------------
# TactileAdapter 主类
# -----------------------------------------------------------------------------

class TactileAdapter:
    """hand2 触觉传感器统一适配器。

    使用方式：
        adapter = TactileAdapter(profile)
        adapter.connect("COM5", 460800)
        while running:
            frame = adapter.read_frame(timeout=0.05)
            if frame is None:
                continue
            ...
        adapter.cleanup()
    """

    _FRAME_QUEUE_SIZE = 8     # 缓存最近 N 帧，溢出时丢弃最早的
    _POLL_SLEEP_S = 0.001     # in_waiting 为 0 时的轻量等待

    def __init__(self, profile=None):
        self.profile = profile
        self.config: Dict[str, Any] = (
            getattr(profile, "tactile_adapter_config", {}) if profile else {}
        ) or {}
        self.connected = False

        finger_row_order = str(
            self.config.get("finger_row_order", FINGER_ROW_ORDER_DEFAULT)
        )
        thumb_row_order = str(
            self.config.get("thumb_row_order", THUMB_ROW_ORDER_DEFAULT)
        )
        # 接入向导第 5 步只暴露一个 frame_format 字段，含义同时覆盖：
        #   1. 适配器返回的字典形态（OUTPUT_*）
        #   2. 串口线协议（PROTOCOL_*）。
        # 其中 patches / finger_palm / adapter_dict（向导默认）都对应 hand2
        # 的整手 AA BB 协议；只有显式选 raw_6x8 才允许走 AA 55 旧版协议。
        # 兼容旧配置：output_format / protocol / wire_protocol 仍可单独指定。
        frame_format = self.config.get("frame_format")
        self._output_format = self._normalize_output(
            self.config.get("output_format", frame_format)
        )
        self._protocol = self._normalize_protocol(
            self.config.get(
                "protocol", self.config.get("wire_protocol", frame_format)
            )
        )

        self._serial = None
        self._parser = _FrameParser(
            finger_row_order, thumb_row_order, protocol=self._protocol
        )
        self._frames: Deque[Dict] = deque(maxlen=self._FRAME_QUEUE_SIZE)

    # ---------- 连接管理 ----------

    def connect(self, port: str, baudrate: int) -> bool:
        if not HAS_SERIAL:
            raise RuntimeError("pyserial 未安装，无法打开触觉串口")
        if not port:
            raise ValueError("port 为空，请在连接页选择正确的触觉串口")
        self.disconnect()
        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=int(baudrate),
                timeout=0,            # 非阻塞读，由 read_frame 控制 timeout
                write_timeout=0.1,
            )
        except Exception:
            self._serial = None
            self.connected = False
            raise
        self._parser.clear()
        self._frames.clear()
        self.connected = True
        return True

    def disconnect(self) -> None:
        if self._serial is not None:
            try:
                if self._serial.is_open:
                    self._serial.close()
            except Exception:
                pass
        self._serial = None
        self._parser.clear()
        self._frames.clear()
        self.connected = False

    @property
    def is_connected(self) -> bool:
        return bool(self._serial and getattr(self._serial, "is_open", False))

    # ---------- 数据读取 ----------

    def read_frame(self, timeout: float = 0.05) -> Optional[Dict]:
        # 已有缓存帧直接返回
        if self._frames:
            return self._format_frame(self._frames.popleft())
        if not self.is_connected:
            return None

        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            try:
                pending = self._serial.in_waiting
            except Exception:
                return None
            if pending > 0:
                try:
                    chunk = self._serial.read(pending)
                except Exception:
                    return None
                if chunk:
                    for frame in self._parser.feed(bytes(chunk)):
                        self._frames.append(frame)
                    if self._frames:
                        return self._format_frame(self._frames.popleft())
            if time.monotonic() >= deadline:
                return None
            time.sleep(self._POLL_SLEEP_S)

    # ---------- 收尾 ----------

    def cleanup(self) -> None:
        self.disconnect()

    # ---------- 内部 ----------

    @staticmethod
    def _normalize_output(value: Any) -> str:
        if value is None:
            return OUTPUT_NATIVE
        return _OUTPUT_ALIASES.get(str(value).strip().lower(), OUTPUT_NATIVE)

    @staticmethod
    def _normalize_protocol(value: Any) -> str:
        if value is None:
            return PROTOCOL_WHOLE_HAND
        return _PROTOCOL_ALIASES.get(
            str(value).strip().lower(), PROTOCOL_WHOLE_HAND
        )

    def _format_frame(self, frame: Dict) -> Dict:
        """根据向导设置把原帧转换成上位机要求的输出格式。"""

        target = self._output_format

        if target == OUTPUT_NATIVE:
            return frame

        if target == OUTPUT_PATCHES:
            if "patches" in frame:
                return frame
            # 旧版 6×8 帧没法还原成完整 patches（缺少分指数据），原样返回。
            return frame

        if target == OUTPUT_RAW:
            if "raw" in frame:
                return frame
            patches = frame.get("patches")
            if isinstance(patches, dict):
                return {"raw": _patches_to_raw_6x8(patches)}
            return frame

        if target == OUTPUT_FINGER_PALM:
            patches = frame.get("patches")
            if isinstance(patches, dict):
                return _patches_to_finger_palm(patches)
            raw = frame.get("raw")
            if isinstance(raw, list) and len(raw) == 6:
                return self._raw_to_finger_palm(raw)
            return frame

        return frame

    @staticmethod
    def _raw_to_finger_palm(raw: List[List[float]]) -> Dict[str, Any]:
        """旧版 6×8 矩阵 → finger+palm；与上位机 _map_to_finger_palm 同规则。"""

        finger: List[List[float]] = []
        for r in range(min(4, len(raw))):
            row = raw[r]
            if len(row) < 8:
                row = list(row) + [0.0] * (8 - len(row))
            mid = sum(row[0:3]) / 3.0
            distal = sum(row[3:5]) / 2.0
            tip = sum(row[5:8]) / 3.0
            finger.append([float(mid), float(distal), float(tip)])
        while len(finger) < 4:
            finger.append([0.0, 0.0, 0.0])

        palm = [0.0] * 8
        if len(raw) >= 6:
            for c in range(8):
                palm[c] = (float(raw[4][c]) + float(raw[5][c])) / 2.0
        elif len(raw) == 5:
            palm = [float(v) for v in raw[4][:8]]
            palm += [0.0] * max(0, 8 - len(palm))
        return {"finger": finger, "palm": palm}


def create_tactile_adapter(profile) -> TactileAdapter:
    """通用上位机统一入口：根据 profile 实例化触觉适配器。"""

    return TactileAdapter(profile)
