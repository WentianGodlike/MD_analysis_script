# utils/plot_XVG_moveavg.py
import matplotlib.pyplot as plt
from scipy.special import j0
from xvg_handler import XVG  # pyright: ignore[reportMissingImports]

COLOR_SCHEMES = {
    "professional": [
        ["#004080", "#0066CC"],  # 深蓝系
        ["#006400", "#009933"],  # 深绿系
        ["#6A51A6", "#D0D1E6"],  # 紫色系
        ["#8B4513", "#CD853F"],  # 棕色系
        ["#8B0000", "#DC143C"],  # 红色系
        ["#006D75", "#00A8B5"],  # 青色系
    ],
    "bright": [
        ["#FF6B6B", "#FF9999"],  # 红色系
        ["#4ECDC4", "#88D8D8"],  # 青色系
        ["#45B7D1", "#87CEEB"],  # 蓝色系
        ["#96CEB4", "#C4E1A4"],  # 绿色系
        ["#FFEAA7", "#FFEFC1"],  # 黄色系
        ["#DDA0DD", "#E6E6FA"],  # 紫色系
    ],
    "pastel": [
        ["#A8E6CF", "#D4F1E6"],  # 薄荷绿
        ["#FFD3B6", "#FFE8D6"],  # 珊瑚橙
        ["#FFAAA5", "#FFC7C4"],  # 粉红色
        ["#B5EAD7", "#D9F2E6"],  # 淡绿色
        ["#C7CEEA", "#E1E6FA"],  # 淡紫色
        ["#F8B195", "#FBCEB1"],  # 桃色系
    ],
    "classic": [
        ["#1F77B4", "#8FBBD9"],  # 蓝色
        ["#FF7F0E", "#FFB07C"],  # 橙色
        ["#2CA02C", "#8FD18F"],  # 绿色
        ["#D62728", "#EB9393"],  # 红色
        ["#9467BD", "#C9B0D4"],  # 紫色
        ["#8C564B", "#C4A69D"],  # 棕色
    ],
}


def get_data(filename) -> tuple:
    """读取xvg文件数据, 并返回给绘图模块"""
    xvg = XVG(filename)

    # 获取标题、X轴标签、Y轴标签
    xvg.title
    xvg.xlabel
    xvg.ylabel

    # 获取数据 - 假设第一列是X轴，第二列是Y轴
    xvg.data_columns[0]
    xvg.data_columns[1]
    return xvg.title, xvg.xlabel, xvg.ylabel, xvg.data_columns[0], xvg.data_columns[1]


def optimization_plot(title, x_label, y_label):
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


def plot_xvg_file(filename):
    """读取GROMACS xvg文件,使用XVG类进行解析,并生成图表

    参数:
        filename: xvg文件路径
    """
    title, x_label, y_label, xs, ys = get_data(filename)

    # 创建图表
    plt.figure(figsize=(12, 7), dpi=120)
    color11 = "#0066CC"
    # 开始绘图
    plt.scatter(
        xs, ys, color=color11, alpha=0.5, s=10, label=y_label + "(Raw Data)", zorder=3
    )

    # 优化图表样式
    optimization_plot(title, x_label, y_label)

    # 保存图表
    output_filename = filename.replace(".xvg", "_plot.png")
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to: {output_filename}")


def plot_multiple_xvg_files(file_list, color_scheme="professional", window_size=100):
    """
    读取多个GROMACS xvg文件并绘制图表

    参数:
    file_list: xvg文件路径列表
    color_scheme: 颜色方案名称 ("professional", "bright", "pastel", "classic")
    window_size: 滑动平均窗口大小
    """
    if not file_list:
        print("错误: 文件列表为空")
        return

    # 获取颜色方案
    if color_scheme not in COLOR_SCHEMES:
        print(f"警告: 颜色方案 '{color_scheme}' 不存在，使用默认方案")
        color_scheme = "professional"

    colors = COLOR_SCHEMES[color_scheme]

    # 如果文件数量超过颜色数量，循环使用颜色
    if len(file_list) > len(colors):
        print(
            f"警告: 文件数量({len(file_list)})超过颜色数量({len(colors)})，将循环使用颜色"
        )
        colors = colors * (len(file_list) // len(colors) + 1)

    # 读取所有文件
    xvg_objects = []
    for file_path in file_list:
        try:
            xvg_obj = XVG(file_path)
            xvg_objects.append(xvg_obj)
        except Exception as e:
            print(f"错误: 无法读取文件 {file_path}: {e}")
            return

    # 从第一个文件获取标题和标签
    title = xvg_objects[0].title
    x_label = xvg_objects[0].xlabel
    y_label = xvg_objects[0].ylabel

    plt.figure(figsize=(12, 7), dpi=120)

    # 处理每个文件
    for i, xvg_obj in enumerate(xvg_objects):
        # 获取数据
        # xvg_obj.apply_shift(28.7, -40000)
        # xvg_obj.apply_shift(44.4, 120000)
        # xvg_obj.apply_shift(100.5, -20000)

        xs = xvg_obj.data_columns[0]
        ys = xvg_obj.data_columns[1]

        # 计算滑动平均
        mavaves, _, _ = xvg_obj.calc_mvave(window_size, 0.95, 1)

        # 处理平滑数据
        xs_smooth = xs[window_size:]
        ys_smooth = mavaves[window_size:]

        # 获取颜色
        color_main = colors[i][0]  # 主色（用于平滑线）
        color_light = colors[i][1]  # 浅色（用于原始数据点）

        # 从文件名生成标签（去掉路径和扩展名）
        file_label = file_list[i].split("/")[-1].replace(".xvg", "")

        # 绘制原始数据点
        plt.scatter(
            xs,
            ys,
            color=color_light,
            alpha=0.5,
            s=10,
            label=f"{file_label} (Raw Data)",
            zorder=3,
        )

        # 绘制滑动平均线
        plt.plot(
            xs_smooth,
            ys_smooth,
            color=color_main,
            linewidth=2.5,
            alpha=0.9,
            label=f"{file_label} ({window_size}ps Avg)",
            zorder=4,
        )

    # 优化图表样式

    optimization_plot(title, x_label, y_label)

    # 保存图表
    output_filename = file_list[0].replace(".xvg", f"_combined_{color_scheme}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_filename}")

    # 显示图表
    plt.show()


if __name__ == "__main__":
    # 请将下面的路径替换为你的实际文件路径
    """xvg_file = (
        "/mnt/d/yry/6_poly_box/data/gyrate.xvg"  # 例如: "path/to/pullf.xvg"
    )
    plot_xvg_file(xvg_file)
"""
    xvg_file = [
        "/mnt/d/yry/50_poly_box/pull/pull_referance/pull_referance_pullf.xvg",
        "/mnt/d/yry/50_poly_box/pull/pull_referance/pull_50ns_pullf.xvg",
    ]
    plot_multiple_xvg_files(xvg_file)
