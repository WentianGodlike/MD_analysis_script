import matplotlib.pyplot as plt
from utils.xvg_handler import XVG


def plot_xvg_file(filename):
    """读取GROMACS xvg文件,使用XVG类进行解析,并生成图表

    参数:
        filename: xvg文件路径
    """
    # 使用XVG类解析xvg文件
    xvg = XVG(filename)

    # 获取标题、X轴标签、Y轴标签
    title = xvg.title
    x_label = xvg.xlabel
    y_label = xvg.ylabel

    # 获取数据 - 假设第一列是X轴，第二列是Y轴
    xs = xvg.data_columns[0]
    ys = xvg.data_columns[1]
    for i in range(len(ys)):
        if 52.2 <= xs[i]:
            ys[i] = ys[i] + 120000
        if 0 < xs[i]:
            ys[i] = ys[i] - 20000
    window_size = 100

    new_xvg = XVG(
        "/mnt/d/yry/50_poly_box/pull/45000/pull_pullf_new.xvg",
        is_file=False,
        new_file=True,
    )
    new_xvg.data_columns = [xs, ys]
    new_xvg.data_heads = xvg.data_heads
    new_xvg.xlabel = xvg.xlabel
    new_xvg.ylabel = xvg.ylabel
    new_xvg.title = xvg.title
    new_xvg.save("/mnt/d/yry/50_poly_box/pull/2800/pull_pullf_new.xvg")

    mavaves1, _, _ = xvg.calc_mvave(window_size, 0.7, 1)
    xs_smooth = xs[window_size:]
    ys_smooth = mavaves1[window_size:]

    # 创建图表
    plt.figure(figsize=(12, 7), dpi=120)
    color1 = "#004080"  # 更亮的蓝色（专业学术色）
    color11 = "#0066CC"

    # 开始绘图
    plt.scatter(
        xs, ys, color=color11, alpha=0.5, s=10, label=y_label + "(Raw Data)", zorder=3
    )
    plt.plot(
        xs_smooth,
        ys_smooth,
        color=color1,
        linewidth=2.5,
        alpha=0.9,
        label=y_label + " (100ps Avg)",
        zorder=4,
    )

    # 优化图表样式
    plt.title(title, fontsize=16, pad=20, fontweight="bold")

    plt.xlabel(x_label, fontsize=14, labelpad=12, fontweight="bold")

    plt.ylabel(y_label, fontsize=14, labelpad=12, fontweight="bold")

    # 优化网格（更细、更淡）
    plt.grid(True, linestyle="-", alpha=0.3, color="gray", linewidth=0.8)

    # 优化图例（更清晰，放在合适位置）
    plt.legend(
        fontsize=12,
        frameon=True,
        framealpha=0.8,
        loc="best",
        borderpad=0.5,
        labelspacing=0.5,
    )

    # 优化坐标轴（更清晰）
    plt.tick_params(axis="both", which="major", labelsize=11)

    # 优化布局（避免标签被截断）
    plt.tight_layout(pad=3.0)

    # 保存图表
    output_filename = filename.replace(".xvg", "_plot.png")
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to: {output_filename}")


if __name__ == "__main__":
    # 请将下面的路径替换为你的实际文件路径
    xvg_file = "/mnt/d/yry/50_poly_box/pull/2800/pull_pullf.xvg"  # 例如: "path/to/pullf.xvg"  # 例如: "path/to/pullf.xvg"
    plot_xvg_file(xvg_file)
