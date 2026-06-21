#!/usr/bin/env python3
"""自动扫描 ExtraModules 目录，生成模块清单 Markdown。

用法:
    python tools/generate_module_list.py

输出:
    docs/模块清单.md — 包含所有模块的分类列表、类名、文件路径
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
EXTRA_MODULES = ROOT / "ultralytics" / "nn" / "ExtraModules"
OUTPUT = ROOT / "docs" / "模块清单.md"

# 子目录中文名映射
DIR_NAMES = {
    "attention": "注意力机制",
    "conv": "卷积变体",
    "fusion": "双模态融合",
    "head": "检测头",
    "frequency": "频域模块",
    "mamba": "SSM状态空间模型",
    "dynamic": "动态/自适应卷积",
    "context": "全局上下文聚合",
    "neck": "颈部模块",
    "loss": "独立损失/工具",
}

CLASS_PATTERN = re.compile(r"^class\s+([A-Za-z_]\w*)\s*[\(:]", re.MULTILINE)


def scan_modules():
    """扫描 ExtraModules 下所有子目录，返回 {子目录: [(类名, 文件名), ...]}"""
    modules = defaultdict(list)

    for subdir in sorted(EXTRA_MODULES.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        for py_file in sorted(subdir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for match in CLASS_PATTERN.finditer(content):
                class_name = match.group(1)
                # 跳过内部辅助类和基类
                if class_name.startswith("_"):
                    continue
                modules[subdir.name].append((class_name, py_file.name))

    # 也扫描 common.py 中的包装器
    common_py = EXTRA_MODULES / "common.py"
    if common_py.exists():
        content = common_py.read_text(encoding="utf-8", errors="ignore")
        for match in CLASS_PATTERN.finditer(content):
            class_name = match.group(1)
            if not class_name.startswith("_"):
                modules["common.py"].append((class_name, "common.py"))

    return modules


def generate_markdown(modules):
    """生成模块清单 Markdown 内容。"""
    lines = [
        "# 模块清单（自动生成）",
        "",
        f"> 自动生成时间: 由 `python tools/generate_module_list.py` 生成",
        f"> 扫描目录: `ultralytics/nn/ExtraModules/`",
        "> **请勿手动编辑此文件**，重新运行脚本即可更新。",
        "",
        "> 相关文档: [指令.md](指令.md) | [创新点.md](创新点.md) | [改进位置.md](改进位置.md)",
        "",
        "---",
        "",
    ]

    total = sum(len(v) for v in modules.values())
    lines.append(f"## 总计: {total} 个模块")
    lines.append("")

    # 按子目录输出
    for subdir in sorted(modules.keys()):
        class_list = modules[subdir]
        dir_label = DIR_NAMES.get(subdir, subdir)
        lines.append(f"### {subdir}/ ({dir_label}) — {len(class_list)} 个")
        lines.append("")
        lines.append("| 序号 | 模块类名 | 文件 |")
        lines.append("|------|----------|------|")
        for i, (cls_name, file_name) in enumerate(class_list, 1):
            lines.append(f"| {i} | `{cls_name}` | `{file_name}` |")
        lines.append("")

    # 统计摘要表
    lines.append("---")
    lines.append("")
    lines.append("## 分类统计")
    lines.append("")
    lines.append("| 子目录 | 中文名 | 模块数 |")
    lines.append("|--------|--------|--------|")
    for subdir in sorted(modules.keys()):
        dir_label = DIR_NAMES.get(subdir, subdir)
        lines.append(f"| `{subdir}/` | {dir_label} | {len(modules[subdir])} |")
    lines.append(f"| **合计** | | **{total}** |")
    lines.append("")

    return "\n".join(lines)


def main():
    modules = scan_modules()
    md = generate_markdown(modules)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(md, encoding="utf-8")
    print(f"Generated {OUTPUT} ({sum(len(v) for v in modules.values())} modules)")


if __name__ == "__main__":
    main()
