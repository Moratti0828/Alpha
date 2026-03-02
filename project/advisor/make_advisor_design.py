"""
make_advisor_design.py

用途：
- 独立生成投顾设计展示所需的美化素材（配色、环图、风险仪表盘、横幅、README），输出到 `advisor_design/` 目录。
- 可在不运行完整实验流程的情况下，单独生成设计素材与示例文本建议（若需要可扩展）。

使用方式（Windows, cmd.exe / PowerShell）：
- 在项目根目录下执行：
  python make_advisor_design.py

输出：
- advisor_design/palette.png：配色方案
- advisor_design/portfolio_donut.png：资产配置环图（示例）
- advisor_design/risk_gauge.png：风险仪表盘（示例）
- advisor_design/banner.png：报告横幅
- advisor_design/README.md：素材使用说明

备注：依赖 `run_experiment.py` 中的工具函数（目录创建与图片保存函数）。
"""

# make_advisor_design.py
# 独立生成投顾设计美化素材到 advisor_design/
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_experiment import (
    ensure_advisor_dir,
    save_color_palette,
    save_portfolio_donut,
    save_risk_gauge,
    save_banner,
    write_design_readme,
)
import numpy as np


def main():
    """
    主入口：生成投顾设计素材到 `advisor_design/`。

    参数：
    - 无（可根据需要扩展，例如接受命令行参数来控制输出内容）

    返回：
    - 无（在控制台打印输出目录路径）

    使用示例：
    - python make_advisor_design.py
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    advisor_dir = ensure_advisor_dir(project_root)
    colors = save_color_palette(advisor_dir)
    save_portfolio_donut(advisor_dir)
    save_risk_gauge(advisor_dir, score=0.6)
    save_banner(advisor_dir)
    write_design_readme(advisor_dir, colors)
    print(f"投顾设计素材已生成：{advisor_dir}")


if __name__ == "__main__":
    main()
