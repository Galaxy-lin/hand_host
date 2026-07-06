from __future__ import annotations

import math
import shutil
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / "软著材料"
SOURCE_DOCX = MATERIALS / "通用多指灵巧手人机交互软件系统V1.0_设计说明书_重写版.docx"
OUTPUT_DOCX = MATERIALS / "通用多指灵巧手人机交互软件系统V1.0_设计说明书_优化版.docx"
FIG3_PATH = MATERIALS / "图3_用户控制设备执行反馈闭环_优化.png"
BUILD_IMAGES = MATERIALS / "_build_images"


FONT_BODY = "宋体"
FONT_HEAD = "微软雅黑"
FONT_MONO = "Consolas"


def ensure_parent() -> None:
    MATERIALS.mkdir(parents=True, exist_ok=True)
    BUILD_IMAGES.mkdir(parents=True, exist_ok=True)


def set_style_font(style, font_name: str, size_pt: float, bold: bool = False, color: str | None = None):
    style.font.name = font_name
    style.font.size = Pt(size_pt)
    style.font.bold = bold
    if color:
        style.font.color.rgb = RGBColor.from_string(color)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), font_name)
    rfonts.set(qn("w:ascii"), font_name)
    rfonts.set(qn("w:hAnsi"), font_name)


def set_run_font(run, font_name: str = FONT_BODY, size_pt: float | None = None, bold: bool | None = None):
    run.font.name = font_name
    if size_pt:
        run.font.size = Pt(size_pt)
    if bold is not None:
        run.font.bold = bold
    rfonts = run._element.get_or_add_rPr().rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        run._element.get_or_add_rPr().append(rfonts)
    rfonts.set(qn("w:eastAsia"), font_name)
    rfonts.set(qn("w:ascii"), font_name)
    rfonts.set(qn("w:hAnsi"), font_name)


def style_document(doc: Document) -> None:
    section = doc.sections[0]
    section.start_type = WD_SECTION_START.NEW_PAGE
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.1)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    section.header_distance = Cm(1.2)
    section.footer_distance = Cm(1.2)

    styles = doc.styles
    normal = styles["Normal"]
    set_style_font(normal, FONT_BODY, 11)
    normal.paragraph_format.line_spacing = 1.18
    normal.paragraph_format.space_after = Pt(6)

    for name, size, before, after, color in [
        ("Heading 1", 16, 16, 8, "1F4E79"),
        ("Heading 2", 13, 10, 5, "2E74B5"),
        ("Heading 3", 12, 8, 4, "1F4E79"),
    ]:
        style = styles[name]
        set_style_font(style, FONT_HEAD, size, bold=True, color=color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    if "TitleDoc" not in styles:
        styles.add_style("TitleDoc", WD_STYLE_TYPE.PARAGRAPH)
    set_style_font(styles["TitleDoc"], FONT_HEAD, 22, bold=True, color="0B2545")
    styles["TitleDoc"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    styles["TitleDoc"].paragraph_format.space_after = Pt(8)

    if "DocSubtitle" not in styles:
        styles.add_style("DocSubtitle", WD_STYLE_TYPE.PARAGRAPH)
    set_style_font(styles["DocSubtitle"], FONT_HEAD, 15, bold=False, color="3B4B5A")
    styles["DocSubtitle"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    styles["DocSubtitle"].paragraph_format.space_after = Pt(18)

    if "MetaText" not in styles:
        styles.add_style("MetaText", WD_STYLE_TYPE.PARAGRAPH)
    set_style_font(styles["MetaText"], FONT_BODY, 10.5, color="4B5563")
    styles["MetaText"].paragraph_format.space_after = Pt(3)

    if "CaptionText" not in styles:
        styles.add_style("CaptionText", WD_STYLE_TYPE.PARAGRAPH)
    set_style_font(styles["CaptionText"], FONT_BODY, 10, color="4B5563")
    styles["CaptionText"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    styles["CaptionText"].paragraph_format.space_before = Pt(3)
    styles["CaptionText"].paragraph_format.space_after = Pt(8)
    styles["CaptionText"].paragraph_format.keep_with_next = True

    if "LeadText" not in styles:
        styles.add_style("LeadText", WD_STYLE_TYPE.PARAGRAPH)
    set_style_font(styles["LeadText"], FONT_BODY, 11, color="111827")
    styles["LeadText"].paragraph_format.line_spacing = 1.18
    styles["LeadText"].paragraph_format.space_after = Pt(8)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.style = styles["MetaText"]
    footer.add_run("通用多指灵巧手人机交互软件系统 V1.0 设计说明书  第 ")
    add_page_number(footer)
    footer.add_run(" 页")


def add_page_number(paragraph):
    run = paragraph.add_run()
    fld_char_1 = OxmlElement("w:fldChar")
    fld_char_1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char_2 = OxmlElement("w:fldChar")
    fld_char_2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char_1)
    run._r.append(instr_text)
    run._r.append(fld_char_2)


def add_body(doc: Document, text: str, style: str = "Normal", indent: bool = True):
    p = doc.add_paragraph(style=style)
    if indent:
        p.paragraph_format.first_line_indent = Pt(22)
    else:
        p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.line_spacing = 1.18
    p.paragraph_format.space_after = Pt(6)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for idx, part in enumerate(text.split("`")):
        run = p.add_run(part)
        if idx % 2 == 1:
            set_run_font(run, FONT_MONO, 10)
        else:
            set_run_font(run, FONT_BODY, 11)
    return p


def add_heading(doc: Document, text: str, level: int = 1):
    p = doc.add_paragraph(text, style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    for run in p.runs:
        set_run_font(run, FONT_HEAD, 16 if level == 1 else 13, bold=True)
    return p


def add_caption(doc: Document, text: str):
    p = doc.add_paragraph(text, style="CaptionText")
    for run in p.runs:
        set_run_font(run, FONT_BODY, 10)
    return p


def add_picture(doc: Document, path: Path, width: float = 6.25):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))
    return p


def extract_source_images() -> dict[int, Path]:
    images: dict[int, Path] = {}
    with ZipFile(SOURCE_DOCX) as zf:
        for idx in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]:
            member = f"word/media/image{idx}.png"
            target = BUILD_IMAGES / f"image{idx}.png"
            target.write_bytes(zf.read(member))
            images[idx] = target
    return images


def font_path(name: str) -> str:
    candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc") if name == "bold" else Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "arial.ttf"


def wrapped_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        current = ""
        for ch in raw:
            probe = current + ch
            if draw.textlength(probe, font=font) <= max_width:
                current = probe
            else:
                if current:
                    lines.append(current)
                current = ch
        if current:
            lines.append(current)
    return lines


def centered_text(draw, box, title, subtitle, title_font, sub_font, title_color="#111827"):
    x, y, w, h = box
    title_lines = wrapped_lines(draw, title, title_font, w - 48)
    subtitle_lines = wrapped_lines(draw, subtitle, sub_font, w - 52) if subtitle else []
    line_gap = 7
    title_h = len(title_lines) * (title_font.size + line_gap)
    sub_h = len(subtitle_lines) * (sub_font.size + 5)
    total_h = title_h + (10 if subtitle_lines else 0) + sub_h
    cy = y + (h - total_h) / 2
    for line in title_lines:
        tw = draw.textlength(line, font=title_font)
        draw.text((x + (w - tw) / 2, cy), line, font=title_font, fill=title_color)
        cy += title_font.size + line_gap
    cy += 5
    for line in subtitle_lines:
        tw = draw.textlength(line, font=sub_font)
        draw.text((x + (w - tw) / 2, cy), line, font=sub_font, fill="#374151")
        cy += sub_font.size + 5


def arrow(draw, start, end, color="#8FA3B8", width=5):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    length = 20
    spread = 0.48
    pts = [
        (x2, y2),
        (x2 - length * math.cos(angle - spread), y2 - length * math.sin(angle - spread)),
        (x2 - length * math.cos(angle + spread), y2 - length * math.sin(angle + spread)),
    ]
    draw.polygon(pts, fill=color)


def make_fig3() -> Path:
    W, H = 1800, 1120
    img = Image.new("RGB", (W, H), "#F6F9FC")
    draw = ImageDraw.Draw(img)
    title_font = ImageFont.truetype(font_path("bold"), 48)
    subtitle_font = ImageFont.truetype(font_path("regular"), 25)
    box_title_font = ImageFont.truetype(font_path("bold"), 30)
    box_sub_font = ImageFont.truetype(font_path("regular"), 23)
    center_font = ImageFont.truetype(font_path("bold"), 39)
    center_sub_font = ImageFont.truetype(font_path("regular"), 24)

    draw.text((92, 54), "控制与反馈闭环：同一状态源贯穿界面、仿真与真实手", font=title_font, fill="#111827")
    draw.text(
        (96, 112),
        "用户动作、命令整形、服务下发、硬件反馈和界面刷新都围绕 HandState 同步。",
        font=subtitle_font,
        fill="#4B5563",
    )

    boxes = {
        "input": (670, 185, 460, 132, "用户输入", "滑块 / 手势 / 遥操作"),
        "shape": (1245, 306, 390, 136, "命令整形", "限幅 / 防抖 / 映射"),
        "dispatch": (1245, 620, 390, 136, "服务下发", "电机 / 仿真 / 记录"),
        "execute": (675, 826, 450, 136, "设备执行", "电机动作 / 模型运动"),
        "feedback": (175, 620, 390, 136, "反馈采集", "遥测 / 触觉 / 画面"),
        "ui": (175, 306, 390, 136, "界面刷新", "状态灯 / 热图 / 日志"),
    }
    fills = {
        "input": ("#EAF2FF", "#3B82F6"),
        "shape": ("#EEF2FF", "#64748B"),
        "dispatch": ("#EAFBF1", "#22A06B"),
        "execute": ("#FFF7E6", "#D79023"),
        "feedback": ("#FFECEC", "#D95B59"),
        "ui": ("#EAF8FC", "#2A9DBF"),
    }

    for key, (x, y, w, h, title, sub) in boxes.items():
        fill, accent = fills[key]
        draw.rounded_rectangle((x, y, x + w, y + h), radius=26, fill=fill, outline="#D7E2EC", width=2)
        draw.rounded_rectangle((x + 22, y + 24, x + 34, y + h - 24), radius=6, fill=accent)
        centered_text(draw, (x + 42, y, w - 54, h), title, sub, box_title_font, box_sub_font)

    cx, cy, cw, ch = 660, 438, 480, 205
    draw.rounded_rectangle((cx, cy, cx + cw, cy + ch), radius=28, fill="#FFFFFF", outline="#C8D7E5", width=3)
    centered_text(draw, (cx, cy + 16, cw, ch - 32), "HandState\n当前手型运行态", "角度 / 遥测 / 触觉 / 状态信号", center_font, center_sub_font, "#111827")

    arrow(draw, (1130, 251), (1245, 348))
    arrow(draw, (1440, 442), (1440, 620))
    arrow(draw, (1245, 686), (1125, 860))
    arrow(draw, (675, 894), (565, 690))
    arrow(draw, (370, 620), (370, 442))
    arrow(draw, (565, 348), (670, 251))
    arrow(draw, (900, 317), (900, 438), "#B5C4D2", 4)
    arrow(draw, (1140, 540), (1245, 650), "#B5C4D2", 4)
    arrow(draw, (565, 690), (660, 552), "#B5C4D2", 4)
    arrow(draw, (660, 500), (565, 375), "#B5C4D2", 4)

    note = (1165, 860, 500, 138)
    draw.rounded_rectangle((note[0], note[1], note[0] + note[2], note[1] + note[3]), radius=22, fill="#FFFFFF", outline="#D7E2EC", width=2)
    centered_text(
        draw,
        note,
        "异常处理",
        "连接失败、适配器异常、无效数据写入错误信号与日志，界面提示并阻止高风险操作。",
        ImageFont.truetype(font_path("bold"), 26),
        ImageFont.truetype(font_path("regular"), 20),
    )

    img.save(FIG3_PATH, quality=95)
    return FIG3_PATH


def add_title_page(doc: Document):
    doc.add_paragraph("通用多指灵巧手人机交互软件系统 V1.0", style="TitleDoc")
    doc.add_paragraph("设计说明书", style="TitleDoc")
    doc.add_paragraph("软件著作权登记材料", style="DocSubtitle")
    for line in [
        "软件简称：灵巧手上位机",
        "版本号：V1.0",
        "文档用途：用于说明软件总体设计、运行机制、关键模块和实际界面",
        "文档日期：2026 年 5 月 28 日",
    ]:
        p = doc.add_paragraph(line, style="MetaText")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            set_run_font(run, FONT_BODY, 10.5)
    doc.add_page_break()


def build_doc(images: dict[int, Path], fig3: Path) -> Document:
    doc = Document()
    style_document(doc)
    add_title_page(doc)

    add_heading(doc, "一、软件概述", 1)
    add_body(doc, "通用多指灵巧手人机交互软件系统 V1.0 是一套基于 Python 和 PySide6 开发的桌面上位机程序，主要用于多指灵巧手的连接管理、关节控制、标定维护、触觉显示、仿真遥操作、数据记录和手型接入。软件运行在 PC 端，提供统一的人机交互界面，并通过手型包和适配器机制连接不同型号、不同关节数量、不同通信协议和不同触觉布局的灵巧手设备。")
    add_body(doc, "本软件的开发目的，是把灵巧手研发调试中分散的电机控制、传感器观察、模型验证和实验数据管理集中到同一套上位机中。操作者不需要分别打开多个调试工具，也不需要在主程序里反复修改硬件常量；只要当前手型按约定提供 manifest.json、驱动适配器、触觉适配器、模型文件和标定资源，软件即可在统一界面下加载并使用。")
    add_body(doc, "软件面向机器人灵巧操作、人形机器人末端执行器、仿生手、假肢控制、触觉感知、远程遥操作、机器人仿真验证以及高校科研教学等应用场景。典型使用者包括机器人算法研究人员、硬件调试工程师、灵巧手系统集成工程师和实验平台操作人员。对于这些用户，软件既是设备调试入口，也是关节状态、触觉反馈和实验轨迹的集中观察平台。")
    add_body(doc, "主要功能包括手型包管理、设备连接管理、关节控制、电机标定、触觉可视化、MuJoCo 仿真、摄像头遥操作、预设手势、力控保护、数据记录与轨迹回放。各功能页面共享同一个 HandState 运行状态，页面之间不各自维护独立的角度和遥测副本，从而降低多页面协同时的状态不一致风险。")
    add_body(doc, "技术特点体现在三个方面。第一，软件核心不绑定具体硬件，把手指数量、关节名称、motor_id 映射、触觉 patch、通信协议和遥操作参数下放到手型包。第二，服务层通过统一适配器接口连接电机、触觉、仿真和遥操作模块，使界面页面不直接依赖底层协议。第三，程序使用 Qt 信号、定时器和独立线程处理遥测轮询、触觉采集、摄像头输入和离屏渲染，保证界面操作保持响应。")

    add_heading(doc, "二、软件定位与设计边界", 1)
    add_body(doc, "本软件定位为研发调试型桌面上位机，而不是面向终端用户的封闭式控制台。它直接面对灵巧手样机、调试人员和实验数据，重点解决连接、观察、标定、验证和二次接入问题。软件不要求部署数据库或后端服务，核心数据以 JSON、CSV、模型文件和 Python 适配器形式保存在工程目录及手型包目录中，便于随设备资料一起备份、审查和迁移。")
    add_body(doc, "程序边界划分为上位机核心、手型包、外部设备和算法三层。上位机核心负责主窗口、页面切换、状态模型、服务调度和通用异常提示；手型包负责声明当前灵巧手的关节、电机、触觉、模型、手势、标定和适配器；外部设备与算法包括电机驱动板、触觉采集板、摄像头、MuJoCo 模型和遥操作算法。核心程序只调用统一接口，不直接解析某一厂商的私有硬件帧。")
    add_body(doc, "这种边界使软件具有较强的扩展性。新增一只灵巧手时，开发人员不需要重写主窗口和各功能页，而是按接入规范准备手型包。程序启动或刷新后由注册表读取 manifest.json，生成 HandProfile，并把页面、状态和服务重建到当前手型。若没有安装任何手型，软件会进入“未添加手型”的占位状态，并引导用户先进入手型管理页面。")
    add_picture(doc, images[1])
    add_caption(doc, "图 1  软件设计边界与手型包接入关系")
    add_body(doc, "图 1 表达了本软件的基本边界：上位机核心保持稳定，手型包描述差异，外部设备和算法通过适配器接入。界面页面不直接理解具体硬件协议，页面动作先进入统一状态和服务层，再由当前手型的驱动适配器转换为电机命令；触觉、遥测和仿真反馈也按相同路径回到状态模型，再触发界面刷新。")

    add_heading(doc, "三、总体运行架构", 1)
    add_body(doc, "程序入口位于 main.py。启动时，程序先在导入 PySide6 前配置 Qt 运行环境并抑制常见字体警告，随后创建 QApplication，设置全局浅色主题，实例化 MainWindow 并进入 Qt 事件循环。MainWindow 负责加载用户上次选择的 active_hand，调用手型注册表设置当前 HandProfile，并创建全局共享的 HandState。")
    add_body(doc, "主窗口采用左侧导航栏和右侧 QStackedWidget 页面栈结构。当前实现的功能页包括连接管理、关节控制、电机标定、触觉可视化、仿真遥操作、数据记录、预设手势和手型管理。页面在创建时接收同一个 HandState，并通过主窗口或连接页取得当前电机服务、触觉服务和仿真遥操作服务，因此页面之间可以共享连接状态、当前角度、遥测数据和手型切换结果。")
    add_body(doc, "运行时对象可以分为四类：界面页面负责展示控件、接收用户动作和输出日志；状态与配置对象负责描述当前手型能力并保存运行时状态；服务对象负责连接、轮询、命令防抖、数据归一化、异常转换和线程调度；手型包中的适配器和资源负责接入具体硬件、模型和算法。这种架构把稳定的上位机能力和频繁变化的硬件差异分开。")
    add_picture(doc, images[2])
    add_caption(doc, "图 2  软件运行时分层架构")
    add_body(doc, "图 2 中的分层不是为了增加接口数量，而是为了让不同变化速度的内容分开。界面布局和状态信号属于主程序核心，变化相对较少；驱动协议、触觉阵列、模型结构和遥操作映射随手型变化更频繁，因此放在手型包和适配器层。服务层处在两者之间，既理解上位机需要的统一接口，也负责调用当前手型的适配器。")

    add_heading(doc, "四、数据与控制闭环", 1)
    add_body(doc, "上位机的典型运行过程可以理解为一个连续闭环。用户在关节滑块、预设手势、数据回放或遥操作页面产生控制意图后，页面不会直接操作硬件，而是先把目标角度写入 HandState，或通过服务层形成完整关节目标。HandState 根据当前手型的关节范围进行限幅，服务层再把角度列表或 motor_id 映射后的角度字典交给电机适配器。")
    add_body(doc, "真实硬件执行后，遥测数据通过 PluginMotorService 的轮询定时器回到服务层，再更新电机在线状态、位置、电流、温度、错误码等字段。触觉链路采用相同思路：TactileSensorService 可以加载外部 Python 触觉适配器，也可以启动模拟触觉数据；适配器读取到的数据被整理成按 patch 命名的二维阵列，再进入 HandState 的 tactile_patches。")
    add_body(doc, "仿真遥操作页面同样使用这套闭环。摄像头输入和重定向算法生成目标关节角后，目标既可以写入 MuJoCo 模型用于离线验证，也可以在配置允许时下发到真实电机。界面刷新、数据记录和异常日志都从同一个状态源读取信息，因此切换手型、断开连接、加载标定或执行手势时，相关页面可以通过信号同步更新。")
    add_picture(doc, fig3)
    add_caption(doc, "图 3  用户控制、设备执行与反馈更新闭环")
    add_body(doc, "图 3 重新梳理了用户输入、命令整形、服务下发、设备执行、反馈采集和界面刷新的关系。闭环中最重要的约束是“同一状态源”：页面显示、记录回放、仿真渲染和硬件控制尽量从 HandState 获取当前值，而不是各自保存临时状态。这样可以减少状态不同步导致的误操作，也便于在异常出现时统一给出提示和日志。")

    add_heading(doc, "五、手型包与二次接入机制", 1)
    add_body(doc, "手型包是本软件支持通用多指灵巧手的主要设计。每个手型包位于 app/resources/hands/<hand_id> 目录下，至少包含 manifest.json、驱动适配器和基础资源。manifest.json 使用 dexhand.hand.v1 模式描述手型 ID、显示名称、手指与关节、motor_id 映射、角度范围、默认协议、默认速率、触觉布局、遥操作配置和资源路径。")
    add_body(doc, "驱动适配器把上位机统一方法转换成具体硬件命令，例如连接、断开、使能、单电机角度设置、整手角度设置、回零、读取遥测、设置零点和查找极限等。PluginMotorService 对适配器做 Qt 服务封装，并提供命令防抖与遥测轮询。这样关节控制页面、标定页面、手势页面和遥操作页面都可以调用相同服务接口，而不需要关心底层是串口、CAN、EtherCAT、网关还是厂家 SDK。")
    add_body(doc, "触觉和遥操作也采用插件化接入。触觉适配器负责连接采集设备并输出 patches 或 raw 结构，软件按 hand profile 中的 rows、cols、group、display_transform 和 site_transform 信息进行显示与仿真映射。遥操作适配器负责把摄像头或外部算法输出映射成目标关节角，必要时结合 tip link、wrist link、模型标定和速率限制。")
    add_body(doc, "手型管理页面提供接入向导，用于生成手型包框架、填写电机协议、触觉布局和遥操作适配信息。生成后的包可以被刷新加载，删除时移动到 .trash 目录，避免直接丢失接入材料。对软著材料而言，这一部分说明了软件不仅是固定硬件的控制界面，还包含可扩展的手型描述、适配、注册和管理能力。")

    add_heading(doc, "六、功能界面与人机交互设计", 1)
    add_body(doc, "软件界面围绕研发调试流程组织：先选择或接入手型，再建立电机与触觉连接，随后进行关节控制、标定维护、触觉观察、仿真遥操作、数据记录、手势调用和新手型管理。各页面既相互独立，又共享当前 HandProfile 与 HandState，使操作者在同一套界面中完成从设备连接到实验验证的完整流程。", style="LeadText", indent=False)

    screenshots = [
        (4, "图 4  连接管理界面", "连接管理界面是硬件会话入口。页面同时处理电机通信和触觉通信，提供端点选择、协议与速率选择、连接断开、使能失能、开机归位和运行日志。电机状态以网格方式展示，便于调试时快速观察多个电机是否在线；触觉通信放在同一页，是因为实验开始前通常需要同时确认电机和触觉两条链路。"),
        (5, "图 5  关节控制界面", "关节控制界面按手指组织关节滑块和数值输入，滑块范围来自当前 HandProfile 或 calibration.json。用户调整某个关节时，页面先更新 HandState，再通过连接管理页中的电机服务下发到硬件。右侧姿态视图用于观察整手姿态变化，使研发人员可以把数值角度与实际动作对应起来。"),
        (6, "图 6  电机标定界面", "电机标定界面用于读取实时角度、设置每个关节的最小值、最大值和默认位置，并将结果保存到手型包的 calibration.json。页面还提供零点设置、极限查找和标定应用功能。标定完成后，关节控制界面的滑块范围会同步更新，避免操作者继续使用旧范围控制电机。"),
        (7, "图 7  触觉可视化界面", "触觉可视化界面根据当前手型声明的触觉 patch 布局展示压力分布。页面按区域名称和矩阵尺寸绘制热力图，并统计最大值、平均值和激活点数量。由于页面关注的是归一化后的触觉结果，只要适配器输出约定结构，不同采集板的原始帧格式都可以被统一显示。"),
        (8, "图 8  仿真遥操作界面", "仿真遥操作界面把 MuJoCo 仿真和摄像头遥操作放在同一工作区。左侧显示模型渲染，右侧显示摄像头画面和手部检测状态，顶部提供加载模型、启动仿真、启动摄像头、启动遥操作和力矩缩放等控制。该页面用于验证视觉重定向结果，也能在合适配置下同时驱动真实电机。"),
        (9, "图 9  数据记录与回放界面", "数据记录页面负责把当前关节角轨迹保存为 CSV，并支持加载轨迹文件后按设定速率回放。记录功能保存的是一段连续动作过程，不同于单个静态手势。调试人员可以用它复现实验动作、比较不同标定下的控制效果，或把轨迹作为后续算法测试数据。"),
        (10, "图 10  预设手势界面", "预设手势界面提供张开、握拳、捏取、指向等常用姿态的一键执行，也允许把当前关节角保存为自定义手势。手势数据与当前手型关联，长度必须与当前关节数量一致。页面还提供力矩缩放等控制选项，使操作者在真实硬件上执行手势时可以根据负载和安全需要调整力度。"),
        (11, "图 11  手型管理界面", "手型管理界面展示已接入手型的 ID、显示名称、关节数、驱动、触觉、遥操作和包目录，并提供添加新手型、刷新列表、删除或隐藏手型等操作。它把手型接入从手工拷贝文件变成可视化流程，是软件通用化设计的重要界面证据。"),
    ]
    for image_id, caption, text in screenshots:
        add_picture(doc, images[image_id], width=6.1)
        add_caption(doc, caption)
        add_body(doc, text)

    add_heading(doc, "七、关键模块设计说明", 1)
    add_heading(doc, "7.1 主窗口与页面组织", 2)
    add_body(doc, "MainWindow 是软件的主容器，左侧侧边栏承担手型选择、页面切换和连接状态提示，右侧 QStackedWidget 承载具体功能页。页面在创建时接收共享的 HandState，并通过窗口对象访问连接管理页中的电机服务。这个设计避免每个页面重复创建硬件连接，也使页面之间可以通过统一状态协作。")
    add_body(doc, "当用户切换当前手型时，MainWindow 会调用 set_active_profile 更新注册表，重置 HandState，并重建所有页面。这样每个页面都能拿到新的关节数量、触觉布局、模型路径和手势目录，避免旧手型的界面控件与新手型状态混用。")

    add_heading(doc, "7.2 HandProfile 与 HandState", 2)
    add_body(doc, "HandProfile 是当前手型的静态描述，包含 hand_id、display_name、手指关节配置、模型路径、标定路径、手势目录、电机协议、触觉布局和遥操作配置。它来自手型包 manifest.json 和 calibration.json，是主程序理解当前设备能力的入口。")
    add_body(doc, "HandState 是运行时状态，保存当前角度数组、电机遥测、触觉 patches 和相关 Qt 信号。它在设置单个关节或整手角度时进行范围约束，并在状态变化后发出信号。页面只需要监听这些信号即可刷新界面，不必关心数据来自滑块、手势、遥操作还是回放。")

    add_heading(doc, "7.3 电机服务与驱动适配器", 2)
    add_body(doc, "PluginMotorService 负责加载当前手型的 Python 驱动适配器，并把连接、断开、使能、回零、角度下发和遥测读取封装成 Qt 信号接口。服务内部维护连接状态、最近角度、待下发角度和轮询定时器；当页面快速拖动滑块时，命令会经过短时间防抖再集中发送，减少通信压力。")
    add_body(doc, "驱动适配器需要实现 connect、disconnect、enable_all、set_motor_angle、set_angles、set_all_joint_angles、home_all_motors、get_latest_telemetry、get_status 和 cleanup 等方法。若适配器不支持整手角度设置，服务层会退回到按 motor_id 组织的逐电机下发方式，从而提高不同协议之间的兼容性。")

    add_heading(doc, "7.4 触觉服务与数据归一化", 2)
    add_body(doc, "TactileSensorService 面向当前 HandState 工作，可以连接外部触觉适配器，也可以启动模拟触觉数据。外部适配器返回数据后，服务会处理 patches、raw、finger 和 palm 等兼容入口，并把结果写回 HandState。触觉页面由 HandState 触发刷新，因此只需要按照当前手型的 patch 布局绘制热力图。")
    add_body(doc, "触觉布局由 TactilePatchInfo 描述，包括 patch 名称、显示名称、行列数、分组、site 前缀和显示变换。这个设计让触觉采集板的原始通信协议留在手型包中，核心程序只处理标准化后的矩阵，便于同一界面显示不同尺寸和不同区域划分的触觉阵列。")

    add_heading(doc, "7.5 标定、手势、记录、力控与遥操作", 2)
    add_body(doc, "标定模块以关节为单位保存 min_angle、max_angle 和 default_angle，既影响控制范围，也影响切换手型后的默认姿态。手势模块保存一组完整关节角，适合快速调用常用姿态；记录模块保存连续轨迹，更适合复现实验过程。二者都与当前手型关联，避免把一只手的姿态数据误用于另一只手。")
    add_body(doc, "力控模块基于电机电流或力矩遥测进行逐电机堵转检测，在合拢方向上根据阈值进行减速或冻结，并在恢复时采用渐进逼近目标角度的方式减少振荡。仿真遥操作模块把 MuJoCo 模型、摄像头输入和 teleop_adapter 结合起来，使同一套手型资源可用于离线验证、在线预览和真实电机控制。")

    add_heading(doc, "八、数据保存、异常处理与安全边界", 1)
    add_body(doc, "软件中的持久化数据以文件形式保存。手型静态描述保存在 manifest.json，标定结果保存在 calibration.json，预设手势和自定义手势保存在手型包对应目录，轨迹记录以 CSV 保存，用户最近选择的手型保存在 config.json。这样的设计适合研发调试场景，文件可以随手型包一起备份，也便于审查、比较和版本管理。")
    add_body(doc, "异常处理主要体现在服务层。连接失败、适配器加载失败、遥测读取失败、触觉数据结构不合法、MuJoCo 模型加载失败和摄像头不可用等问题，都会被转换为错误信号或日志提示，而不是让异常直接穿透到界面线程。界面页面据此提示操作者，并保持主程序继续运行。")
    add_body(doc, "安全边界主要包括角度限幅、连接状态校验、使能控制、高风险操作集中管理和力控保护。HandState 根据标定范围限制输入角度；服务层在未连接时不会盲目下发命令；回零、极限查找和零点设置等操作集中在连接管理或标定页面；力控模块在电流或力矩异常时限制继续合拢。")
    add_body(doc, "软件仍保留必要的工程边界。真实硬件运行需要相应串口、CAN、网关或厂家 SDK 支持；没有硬件时，手型管理、标定文件维护、触觉模拟、预设手势维护、轨迹回放和部分仿真功能仍可用于开发验证。该边界说明软件既能服务真实设备，也能在离线条件下支持接入和算法调试。")

    add_heading(doc, "九、运行环境与部署方式", 1)
    add_body(doc, "软件基于 Python 运行，图形界面使用 PySide6，状态计算和矩阵处理使用 NumPy，触觉热力图和部分可视化依赖 Matplotlib，摄像头链路可使用 OpenCV，仿真功能按需使用 MuJoCo。项目入口为 main.py，运行时从 app/resources/hands 读取已安装手型包，并从 config.json 获取上次选择的 active_hand。")
    add_body(doc, "部署时需要保持工程目录、app 子包、手型资源目录和依赖库完整。手型包中的模型文件、STL 网格、标定文件、驱动适配器、触觉适配器和遥操作适配器应与 manifest.json 中的路径保持一致。若移植到另一台电脑，应同步复制手型包和 config.json，或通过手型管理页面重新导入。")
    add_body(doc, "验证方式包括静态编译检查、单元测试和界面运行测试。项目中已有 tests 目录用于检查手型包、触觉适配器等逻辑；实际交付前可执行 python -m compileall app tests hand_integration_kits 和 python -m pytest -q 进行基础验证。界面层验证则应覆盖启动、切换手型、连接断开、标定保存、触觉显示、手势执行和轨迹回放等主要路径。")

    add_heading(doc, "十、设计结论", 1)
    add_body(doc, "通用多指灵巧手人机交互软件系统 V1.0 以“稳定上位机核心 + 可替换手型包”为主要设计思想。主程序提供统一界面、状态模型、服务调度和反馈闭环；手型包提供具体手的关节、电机、触觉、模型、标定、手势和适配器。通过这种划分，软件既能完成当前灵巧手样机的连接、控制、标定、触觉可视化、遥操作和记录回放，也能为后续新手型接入保留清晰路径。")
    add_body(doc, "本文中的结构图说明了软件边界、运行分层和数据闭环，界面截图展示了主要功能页面及其实际操作形态。文档内容与当前工程中的 main.py、app/main_window.py、app/hands、app/models、app/services 和 app/widgets 等模块相对应，能够反映软件的真实运行方式和主要功能组成。")
    return doc


def main() -> None:
    ensure_parent()
    images = extract_source_images()
    fig3 = make_fig3()
    doc = build_doc(images, fig3)
    doc.save(OUTPUT_DOCX)
    if BUILD_IMAGES.exists():
        shutil.rmtree(BUILD_IMAGES)
    print(OUTPUT_DOCX)
    print(FIG3_PATH)


if __name__ == "__main__":
    main()
