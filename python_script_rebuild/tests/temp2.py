# python_script_rebuild/temp2.py
import sys
from pathlib import Path

# 获取当前脚本的绝对路径
current_file = Path(__file__).resolve()

# 获取项目根目录 (即 python_script_rebuild)
# current_file.parent 是 tests 目录
# current_file.parent.parent 是 python_script_rebuild 目录
project_root = current_file.parent.parent

# 将根目录添加到 sys.path
sys.path.append(str(project_root))


# python_script_rebuild/temp2.py

from utils.xvg_handler import XVG  # noqa: E402


def process_and_plot(input_file, output_xvg_path):
    # 1. 数据处理
    xvg = XVG(input_file)
    ys = xvg.data_columns[1]
    xs = xvg.data_columns[0]

    # 你的数据处理逻辑...
    for i in range(len(ys)):
        if 52.2 <= xs[i]:
            ys[i] += 120000
        if 0 < xs[i]:
            ys[i] -= 20000

    # 保存
    new_xvg = XVG(output_xvg_path, is_file=False, new_file=True)
    new_xvg.df = xvg.df.copy()
    new_xvg.df.iloc[:, 1] = ys
    new_xvg.title = xvg.title
    new_xvg.xlabel = xvg.xlabel
    new_xvg.ylabel = xvg.ylabel
    new_xvg.save(output_xvg_path)

    # 2. 绘图 (现在直接用 XVG 类方法)
    # 方式 A: 批量绘图 (兼容原 XVGPlotter 用法)
    XVG.plot_files(file_list=[output_xvg_path], style="bright", window_size=100)

    # 方式 B: 直接绘制当前对象 (新特性)
    # new_xvg.plot(style="professional", window_size=100)


if __name__ == "__main__":
    input_xvg = "/mnt/d/yry/50_poly_box/pull/45000/pull_pullf_new.xvg"
    processed_xvg = "/mnt/d/yry/50_poly_box/pull/2800/pull_pullf_processed.xvg"
    process_and_plot(input_xvg, processed_xvg)
