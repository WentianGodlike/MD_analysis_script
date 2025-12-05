# python_script_rebuild/temp2.py

from utils.xvg_handler import XVG
from utils.visualization import XVGPlotter


def process_and_plot(input_file, output_xvg_path):
    """
    读取 XVG，应用特定数据修正，保存新文件，然后绘图。
    """
    # === 1. 数据处理 (保留原有的业务逻辑) ===
    xvg = XVG(input_file)

    # 获取数据列引用
    xs = xvg.data_columns[0]
    ys = xvg.data_columns[1]

    # 原有的数据修正逻辑
    for i in range(len(ys)):
        if 52.2 <= xs[i]:
            ys[i] = ys[i] + 120000
        if 0 < xs[i]:
            ys[i] = ys[i] - 20000

    # 创建并保存新的 XVG 对象
    # 注意：这里直接利用 XVG 类保存数据，不需要手写文件写入
    new_xvg = XVG(output_xvg_path, is_file=False, new_file=True)
    new_xvg.df = xvg.df.copy()  # 复制 DataFrame 结构
    # 更新修改后的数据 (假设是第1列)
    new_xvg.df.iloc[:, 1] = ys

    # 继承元数据
    new_xvg.title = xvg.title
    new_xvg.xlabel = xvg.xlabel
    new_xvg.ylabel = xvg.ylabel
    new_xvg.legends = xvg.legends

    new_xvg.save(output_xvg_path)
    print(f"Modified data saved to: {output_xvg_path}")

    # === 2. 可视化 (调用 XVGPlotter) ===
    plotter = XVGPlotter(style="bright")  # 可以尝试不同的风格

    # 绘制处理后的文件
    plotter.plot(file_list=[output_xvg_path], window_size=100)


if __name__ == "__main__":
    input_xvg = "/mnt/d/yry/50_poly_box/pull/45000/pull_pullf_new.xvg"  # 原始文件
    # 建议把处理后的文件和原始文件区分开，或者覆盖
    processed_xvg = "/mnt/d/yry/50_poly_box/pull/2800/pull_pullf_processed.xvg"

    process_and_plot(input_xvg, processed_xvg)
