# 通用灵巧手上位机

这是一个基于 `PySide6` 的通用灵巧手上位机空壳程序。核心程序不绑定任何具体手型、电机数量、通信协议、触觉阵列尺寸或 DexPilot 关节命名；真实硬件能力都通过独立手型包声明和适配器插件接入。

## 新手接入文档

- 完整适配包制作指南：[`docs/new_hand_integration_guide.md`](docs/new_hand_integration_guide.md)

## 当前边界

- 手型：由 `manifest.json` 声明，可以是一指、三指、五指、六指或其他拓扑。
- 电机：由关节表映射 `motor_id`，数量不固定，24 路或更多由手型包决定。
- 通信：核心只把 `endpoint`、`baudrate/rate`、`protocol` 传给驱动适配器；串口、CAN、EtherCAT、网关、厂家 SDK 都应封装在 `driver.py` 里。
- 触觉：核心只接收标准 `patches` 或单块 `raw` 二维矩阵；AA55、AA BB、6x8 等具体协议必须放在手型包的 `tactile_adapter.py` 中。
- 遥操作：DexPilot 默认关节名不再写死在核心里；使用 DexPilot 时必须在手型 manifest 显式声明 `teleop.joint_names` 和 `teleop.tip_link_names`，也可以完全改用外部 `teleop_adapter.py`。
- 自动寻限：核心会优先调用驱动适配器的 `find_motor_limits(...)` 或 `auto_find_limits(...)`；没有实现时才使用通用低速试探流程。

## 目录

- `app/hands/`：手型 profile、manifest 解析、手型注册表、驱动加载。
- `app/services/`：只保留通用运行时服务，不包含具体电机线协议或触觉帧解析。
- `app/widgets/`：连接、关节控制、标定、触觉显示、仿真遥操作、记录、手势、手型管理页面。
- `app/resources/hands/`：已安装手型包目录。仓库默认不内置真实手型。
- `hand_integration_kits/`：示例接入材料。旧协议和具体硬件适配保留在这里，作为手型包示例。

## 手型包结构

一个已安装手型包位于：

```text
app/resources/hands/<hand_id>/
  manifest.json
  driver.py 或 driver_adapter/driver.py
  tactile_adapter/tactile_adapter.py        # 可选
  teleop_adapter/teleop_adapter.py          # 可选
  model/<hand>.xml
  calibration.json
  gestures/
```

`manifest.json` 至少声明：

- `fingers[].joints[]`：关节名、显示名、`motor_id`、角度范围、默认角度、可选 `actuator_name`。
- `motor.protocols`：连接页展示的协议名，例如 `serial`、`canfd`、`ethercat`、`sdk`。
- `driver.path`：电机驱动适配器入口。

触觉声明示例：

```json
{
  "tactile": {
    "kind": "external_python",
    "adapter_path": "tactile_adapter/tactile_adapter.py",
    "default_baudrate": 0,
    "layout": {
      "patches": [
        {"name": "thumb_tip", "display_name": "拇指指尖", "group": "拇指", "rows": 4, "cols": 4},
        {"name": "palm_grid", "display_name": "掌心阵列", "group": "手掌", "rows": 8, "cols": 8, "display_transform": "rotate_180"}
      ]
    },
    "adapter_config": {
      "value_scale": 255,
      "baseline_frames": 30,
      "requires_endpoint": true
    }
  }
}
```

## 适配器契约

电机 `driver.py` 需要提供 `create_driver(profile)` 或 `HandDriver`，并实现：

```python
connect(port, baudrate, protocol=None) -> bool
disconnect()
enable_all(enable=True)
set_motor_angle(motor_id, angle_deg, speed=0.5)
set_angles({motor_id: angle_deg})
set_all_joint_angles([angle_deg, ...])
home_all_motors(target_angles=None, speed=0.3)
get_latest_telemetry() -> dict[int, dict]
get_status(motor_id) -> dict
cleanup()
```

可选实现：

- `set_zero_point(motor_id)`
- `set_max_current(motor_id, value_q15)`
- `find_motor_limits(motor_id, start_angle, config)` 或 `auto_find_limits(...)`

触觉 `tactile_adapter.py` 需要返回标准帧：

```python
{"patches": {"thumb_tip": [[0, 12], [3, 40]], "palm_grid": [[...]]}}
```

或：

```python
{"raw": [[0.0, 0.2], [0.1, 0.7]]}
```

## 使用流程

1. 运行 `python main.py`。
2. 打开 `手型管理`，使用接入向导导入或创建手型包。
3. 在 `连接管理` 中选择通信端点、协议和速率。非串口适配器可在 manifest 中设置 `requires_endpoint: false`，并用 `0` 表示速率由适配器自行处理。
4. 在 `关节控制`、`电机标定`、`触觉可视化`、`仿真遥操作` 等页面调试。

## 一键安装依赖

Windows 下可以直接双击运行仓库根目录的 [`install_requirements.bat`](install_requirements.bat)。

脚本会自动：

1. 检查 `py` 启动器是否可用。
2. 在当前目录创建 `.venv` 虚拟环境（如果还没有）。
3. 使用虚拟环境里的 Python 执行 `pip install -r requirements.txt`。

安装完成后，可用下面的命令启动程序：

```bash
.venv\Scripts\python.exe main.py
```

## 示例包说明

`hand_integration_kits/hand1`、`hand2`、`hand3` 是示例/兼容接入材料，其中可能包含固定 16 电机、AA55/AA BB、6x8、DexPilot 旧命名等具体硬件逻辑。这些逻辑不属于核心上位机，只作为特定手型包插件保留。

## 验证

```bash
python -m pytest -q
python -m compileall app tests hand_integration_kits
```
