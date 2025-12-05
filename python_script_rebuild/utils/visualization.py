import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import colorsys
from pathlib import Path
import sys

# 尝试导入 XVG 处理器，如果不存在则提示
try:
    from utils.xvg_handler import XVG
except ImportError:
    print(
        "错误: 找不到 'xvg_handler.py' 模块。请确保它在同一目录下或在 PYTHONPATH 中。"
    )
    sys.exit(1)


class XVGPlotter:
    def __init__(self, style="professional"):
        """
        初始化绘图器

        参数:
            style: 预设样式名称 ('professional', 'bright', 'dark', 'default')
        """
        self.style_config = {
            "figsize": (12, 7),
            "dpi": 120,
            "title_fontsize": 16,
            "label_fontsize": 14,
            "tick_fontsize": 11,
            "grid_alpha": 0.3,
            "line_width": 2.5,
            "scatter_size": 10,
            "window_size": 100,  # 默认滑动平均窗口
        }

        # 预设的 Matplotlib Colormap 映射
        self.colormap_map = {
            "professional": "viridis",  # 专业、深沉
            "bright": "tab10",  # 明亮、对比度高
            "pastel": "Pastel1",  # 柔和
            "cool": "coolwarm",  # 冷暖色
            "default": "jet",
        }
        self.current_cmap = self.colormap_map.get(style, "viridis")

    def _get_colors(self, n_files):
        """根据文件数量自动生成颜色列表"""
        cmap = plt.get_cmap(self.current_cmap)
        # 生成 0 到 1 之间的均匀分布
        indices = np.linspace(0, 1, n_files)
        return [cmap(i) for i in indices]

    def _generate_lighter_color(self, color, factor=0.4):
        """
        生成颜色的浅色/高亮版本 (基于 HLS 空间调整亮度)

        参数:
            color: 输入颜色 (Hex 或 RGB)
            factor: 亮度提升因子 (0.0 - 1.0)。
                    0.0 表示不变，1.0 表示接近白色。
                    0.3-0.5 通常能产生不错的浅色效果。
        """
        try:
            # 1. 转换为 RGB
            c_rgb = mcolors.to_rgb(color)
            # 2. 转换为 HLS (Hue, Lightness, Saturation)
            h, l, s = colorsys.rgb_to_hls(*c_rgb)

            # 3. 提高亮度
            # 算法: 当前亮度 + (剩余亮度的空间 * 因子)
            # 这样保证了亮度永远在 0-1 之间，且暗色变亮效果明显
            new_l = l + (1.0 - l) * factor

            # 4. 转回 RGB
            return colorsys.hls_to_rgb(h, new_l, s)
        except Exception:
            return color  # 如果转换失败则返回原色

    def _setup_plot_style(self, ax, title, xlabel, ylabel):
        """应用通用的图表美化样式"""
        ax.set_title(
            title,
            fontsize=self.style_config["title_fontsize"],
            pad=20,
            fontweight="bold",
        )
        ax.set_xlabel(
            xlabel,
            fontsize=self.style_config["label_fontsize"],
            labelpad=12,
            fontweight="bold",
        )
        ax.set_ylabel(
            ylabel,
            fontsize=self.style_config["label_fontsize"],
            labelpad=12,
            fontweight="bold",
        )

        ax.grid(
            True,
            linestyle="-",
            alpha=self.style_config["grid_alpha"],
            color="gray",
            linewidth=0.8,
        )

        ax.tick_params(
            axis="both", which="major", labelsize=self.style_config["tick_fontsize"]
        )

        ax.legend(
            fontsize=12,
            frameon=True,
            framealpha=0.8,
            loc="best",
            borderpad=0.5,
            labelspacing=0.5,
        )

    def plot(self, file_list, output_name=None, window_size=None, show=True):
        """
        绘制 XVG 文件列表
        """
        # 统一转换为 Path 对象并过滤不存在的文件
        valid_files = []
        for f in file_list:
            p = Path(f)
            if p.exists():
                valid_files.append(p)
            else:
                print(f"警告: 文件不存在，已跳过: {f}")

        if not valid_files:
            print("错误: 没有有效的文件可供绘制。")
            return

        # 使用局部变量或默认配置
        win_size = (
            window_size if window_size is not None else self.style_config["window_size"]
        )

        # 获取基础颜色 (作为滑动平均线的主色)
        main_colors = self._get_colors(len(valid_files))

        # 创建画布
        fig, ax = plt.subplots(
            figsize=self.style_config["figsize"], dpi=self.style_config["dpi"]
        )

        meta_title, meta_xlabel, meta_ylabel = "Unknown", "X", "Y"

        print(f"--- 开始绘制 (窗口大小: {win_size}) ---")

        for i, file_path in enumerate(valid_files):
            try:
                # 读取 XVG
                xvg_obj = XVG(str(file_path))

                # 如果是第一个文件，记录标签信息
                if i == 0:
                    meta_title = xvg_obj.title
                    meta_xlabel = xvg_obj.xlabel
                    meta_ylabel = xvg_obj.ylabel

                # 获取数据
                xs = xvg_obj.data_columns[0]
                ys = xvg_obj.data_columns[1]

                # 计算滑动平均
                mavaves, _, _ = xvg_obj.calc_mvave(win_size, 0.95, 1)

                # 切片数据以匹配窗口
                xs_smooth = xs[win_size:]
                ys_smooth = mavaves[win_size:]

                # 准备颜色
                # 1. 主色 (Line Color): 来自 Colormap
                main_color = main_colors[i]
                # 2. 浅色 (Scatter Color): 算法计算生成 (提亮 40%)
                scatter_color = self._generate_lighter_color(main_color, factor=0.4)

                label_base = file_path.stem.replace(".xvg", "")

                # 绘制原始数据 (散点) - 使用计算出的浅色
                ax.scatter(
                    xs,
                    ys,
                    color=scatter_color,
                    alpha=0.5,  # 保持一定的透明度，方便观察点的密度
                    s=self.style_config["scatter_size"],
                    label=f"{label_base} (Raw)",
                    zorder=3,
                )

                # 绘制平滑曲线 (实线) - 使用深色/主色
                ax.plot(
                    xs_smooth,
                    ys_smooth,
                    color=main_color,
                    linewidth=self.style_config["line_width"],
                    alpha=0.9,
                    label=f"{label_base} (Avg)",
                    zorder=4,
                )

                print(f"已处理: {file_path.name}")

            except Exception as e:
                print(f"错误: 处理文件 {file_path.name} 时失败: {e}")
                continue

        # 应用美化
        self._setup_plot_style(ax, meta_title, meta_xlabel, meta_ylabel)
        plt.tight_layout(pad=3.0)

        # 保存文件
        if output_name:
            save_path = output_name
        else:
            save_path = valid_files[0].parent / f"{valid_files[0].stem}_combined.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图表已保存至: {save_path}")

        if show:
            plt.show()


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 1. 定义文件列表
    files_to_plot = [
        "/mnt/d/yry/50_poly_box/pull/pull_referance/pull_referance_pullf.xvg",
        # "/mnt/d/yry/50_poly_box/pull/pull_referance/pull_50ns_pullf.xvg",
    ]

    # 2. 初始化绘图器
    # 建议使用 'bright' (Tab10) 或 'professional' (Viridis)
    # 它们生成的基色饱和度较高，生成浅色版后的对比效果最好
    plotter = XVGPlotter(style="bright")

    # 3. 执行绘制
    plotter.plot(files_to_plot, window_size=100)
