# python_script_rebuild/temp.py

from pathlib import Path
from utils.visualization import XVGPlotter


def main():
    # 1. 定义文件路径
    file_path = "/mnt/d/yry/20251120_polymer/2_equil1/msd.xvg"

    # 2. 检查文件是否存在 (可选，但在脚本中是好习惯)
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return

    # 3. 实例化绘图器 (选择样式: 'professional', 'bright', 'pastel' 等)
    plotter = XVGPlotter(style="professional")

    # 4. 绘图
    # plot 接收文件列表，支持同时绘制多个文件
    # output_name 可选，不填则默认保存在原目录下
    plotter.plot(
        file_list=[file_path],
        output_name="/mnt/d/yry/20251120_polymer/2_equil1/msd_plot.png",
        window_size=100,  # 滑动平均窗口大小
    )


if __name__ == "__main__":
    main()
