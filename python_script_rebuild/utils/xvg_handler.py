import os
import time
import io
import colorsys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats


class XVG:
    """
    XVG 文件处理核心类。

    该类集成了 GROMACS .xvg 文件的解析、数据处理、可视化（绘图）以及文件合并功能。
    底层数据结构基于 Pandas DataFrame，以实现高效的向量化计算。

    Attributes:
        xvgfile (str): XVG 文件路径。
        comments (List[str]): 文件头部的注释行（以 # 开头）。
        title (str): 图表标题（@ title）。
        xlabel (str): X 轴标签（@ xaxis label）。
        ylabel (str): Y 轴标签（@ yaxis label）。
        legends (List[str]): 图例列表（@ s[n] legend）。
        df (pd.DataFrame): 存储数值数据的 DataFrame。
    """

    # =========================================================================
    #  全局配置 (Global Configuration)
    # =========================================================================

    # 绘图样式默认配置
    STYLE_CONFIG = {
        "figsize": (12, 7),
        "dpi": 120,
        "title_fontsize": 16,
        "label_fontsize": 14,
        "tick_fontsize": 11,
        "grid_alpha": 0.3,
        "line_width": 2.5,
        "scatter_size": 10,
        "window_size": 100,  # 默认滑动平均窗口大小
    }

    # 预设颜色映射表 (用于自动配色)
    COLORMAP_MAP = {
        "professional": "viridis",  # 专业、学术风 (蓝-绿-黄)
        "bright": "tab10",  # 高对比度 (Matplotlib 默认)
        "pastel": "Pastel1",  # 柔和色系
        "cool": "coolwarm",  # 冷暖渐变 (红-蓝)
        "default": "jet",  # 经典彩虹色
    }

    # =========================================================================
    #  初始化与核心属性 (Initialization & Properties)
    # =========================================================================

    def __init__(
        self,
        xvgfile: str,
        is_file: bool = True,
        new_file: bool = False,
    ) -> None:
        """
        初始化 XVG 对象。

        Args:
            xvgfile: XVG 文件的路径。
            is_file: 是否视为真实文件进行读取。默认为 True。
            new_file: 是否创建一个新的空 XVG 对象（不读取磁盘文件）。默认为 False。

        Raises:
            FileNotFoundError: 当 is_file=True 且文件不存在时抛出。
        """
        self.xvgfile = xvgfile
        self.comments: List[str] = []
        self.title: str = ""
        self.xlabel: str = ""
        self.ylabel: str = ""
        self.legends: List[str] = []

        # 数据存储：使用 Pandas DataFrame 初始化为空
        self.df: pd.DataFrame = pd.DataFrame()

        # GROMACS 视图/视口设置默认值
        self.view: str = "0.15, 0.15, 0.75, 0.85"
        self.world_xmin: Optional[float] = None
        self.world_xmax: Optional[float] = None
        self.world_ymin: Optional[float] = None
        self.world_ymax: Optional[float] = None

        if not new_file and is_file:
            if not os.path.exists(xvgfile):
                raise FileNotFoundError(f"未检测到文件 {xvgfile}！")
            self._load_xvg()

    @property
    def row_num(self) -> int:
        """获取数据行数。"""
        return self.df.shape[0]

    @property
    def column_num(self) -> int:
        """获取数据列数。"""
        return self.df.shape[1]

    @property
    def data_heads(self) -> List[str]:
        """
        基于 xlabel 和 legends 智能生成表头列表。
        用于调试或导出数据时识别列含义。
        """
        heads = [self.xlabel] if self.xlabel else ["X"]

        if self.legends:
            heads.extend(self.legends)
        elif self.ylabel:
            # 如果有多列数据但只有一个 ylabel，尝试自动编号
            if self.column_num > 1:
                heads.append(self.ylabel)
                for i in range(2, self.column_num):
                    heads.append(f"{self.ylabel}_{i}")
        return heads

    @property
    def data_columns(self) -> List[List[float]]:
        """
        [向后兼容] 将 DataFrame 数据转换为 List of Lists (按列优先)。

        Returns:
            List[List[float]]: 包含每一列数据的列表。
        """
        if self.df.empty:
            return []
        return [self.df[col].tolist() for col in self.df.columns]

    # =========================================================================
    #  文件 I/O (File I/O)
    # =========================================================================

    def _load_xvg(self) -> None:
        """
        内部方法：高效加载 XVG 文件。
        策略：先分离元数据行，再使用 pandas.read_csv 批量读取数值部分。
        """
        metadata_lines = []
        data_buffer = []

        with open(self.xvgfile, "r", encoding="UTF-8") as f:
            for line in f:
                sline = line.strip()
                if not sline:
                    continue
                # 分离元数据行（以 # 或 @ 开头）
                if sline.startswith(("#", "@")):
                    metadata_lines.append(sline)
                else:
                    data_buffer.append(line)

        # 1. 解析元数据
        self._parse_metadata(metadata_lines)

        # 2. 解析数值数据
        if data_buffer:
            csv_io = io.StringIO("".join(data_buffer))
            try:
                # sep=r"\s+" 处理不定长空格分隔符
                self.df = pd.read_csv(csv_io, sep=r"\s+", header=None)
            except Exception as e:
                raise ValueError(f"解析文件 {self.xvgfile} 中的数据失败: {e}")
            print(f"成功解析 {self.xvgfile}: {self.row_num} 行, {self.column_num} 列。")
        else:
            print(f"警告: {self.xvgfile} 仅包含元数据。")

    def _parse_metadata(self, lines: List[str]) -> None:
        """解析 GROMACS 特有的元数据语法。"""
        title_flag = False
        xlabel_flag = False
        ylabel_flag = False

        for line in lines:
            if line.startswith("#"):
                self.comments.append(line)
                continue

            if line.startswith("@"):
                clean_line = line.lstrip("@").strip()

                # 解析标题、轴标签、图例等
                if 'title "' in line and not title_flag:
                    self.title = line.split('"')[-2]
                    title_flag = True
                elif 'xaxis  label "' in line and not xlabel_flag:
                    self.xlabel = line.split('"')[-2]
                    xlabel_flag = True
                elif 'yaxis  label "' in line and not ylabel_flag:
                    self.ylabel = line.split('"')[-2]
                    ylabel_flag = True
                elif ' legend "' in line:
                    self.legends.append(line.split('"')[-2])
                elif " view " in line:
                    self.view = clean_line.replace("view", "").strip()
                elif " world xmin " in line:
                    self.world_xmin = float(line.split()[-1])

    def save(self, outxvg: str, check: bool = True) -> None:
        """
        将当前对象状态保存为符合 GROMACS 格式的 .xvg 文件。

        Args:
            outxvg: 输出文件路径。
            check: 是否检查数据为空。默认为 True。
        """
        if check and self.df.empty:
            raise ValueError("无法保存空的 XVG 数据。")

        # 确保图例数量与数据列对应 (补全缺失的图例)
        data_col_count = self.column_num - 1
        if len(self.legends) < data_col_count:
            base_label = self.ylabel if self.ylabel else "Data"
            for i in range(len(self.legends), data_col_count):
                self.legends.append(f"{base_label}_{i}")

        time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        with open(outxvg, "w", encoding="UTF-8") as fo:
            # 1. 写入头部信息
            fo.write(f"# This file was created at {time_info}\n")
            for comm in self.comments:
                fo.write(f"{comm}\n" if not comm.endswith("\n") else comm)

            fo.write(f'@    title "{self.title}"\n')
            if self.xlabel:
                fo.write(f'@    xaxis label "{self.xlabel}"\n')
            if self.ylabel:
                fo.write(f'@    yaxis label "{self.ylabel}"\n')

            fo.write(f"@ view {self.view}\n")
            fo.write("@ legend on\n@ legend box on\n@ legend loctype view\n")
            fo.write("@ legend 0.78, 0.8\n")

            for i, leg in enumerate(self.legends):
                fo.write(f'@ s{i} legend "{leg}"\n')

            # 2. 写入数值数据 (模拟 GROMACS 格式对齐)
            np_data = self.df.values
            for row in np_data:
                line_str = ""
                for val in row:
                    if val.is_integer():
                        line_str += f"{int(val):>8d} "
                    else:
                        line_str += f"{val:>16.6f} "
                fo.write(line_str.strip() + "\n")

        print(f"XVG 文件已成功保存至 {outxvg}。")

    # =========================================================================
    #  数据计算与处理 (Computation)
    # =========================================================================
    # 在 XVG 类中添加

    def aggregate_columns(self, operation="sum", start_col=1) -> "XVG":
        """创建一个新的 XVG 对象，其中包含时间列和其余列的聚合结果"""
        if self.df.empty:
            raise ValueError("数据为空")

        time_col = self.df.iloc[:, 0]
        data_cols = self.df.iloc[:, start_col:]

        if operation == "sum":
            agg_res = data_cols.sum(axis=1)
            suffix = "Sum"
        elif operation == "mean":
            agg_res = data_cols.mean(axis=1)
            suffix = "Mean"
        else:
            raise ValueError(f"不支持的操作: {operation}")

        new_df = pd.DataFrame({0: time_col, 1: agg_res})

        new_xvg = XVG(self.xvgfile, is_file=False, new_file=True)
        new_xvg.df = new_df
        new_xvg.title = f"{self.title} ({suffix})"
        new_xvg.xlabel = self.xlabel
        new_xvg.ylabel = self.ylabel
        new_xvg.legends = [f"Total_{suffix}"]

        return new_xvg

    def calc_mvave(
        self, windowsize: int, confidence: float, column_index: int
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        计算移动平均 (Moving Average) 及其置信区间。

        Args:
            windowsize: 窗口大小（数据点个数）。
            confidence: 置信度 (0 < c < 1)，用于计算标准差范围。
            column_index: 要计算的列索引 (通常 1 为 Y 轴数据)。

        Returns:
            Tuple[List[float], List[float], List[float]]:
            (平均值列表, 上界列表, 下界列表)。
            列表前端不足窗口大小的部分为 NaN。
        """
        if windowsize <= 0 or windowsize > self.row_num // 2:
            raise ValueError("窗口大小 (windowsize) 设置不当。")
        if not (0 < confidence < 1):
            raise ValueError("置信度 (confidence) 必须在 (0, 1) 之间。")
        self.check_column_index(column_index)

        series = self.df.iloc[:, column_index]

        # 使用 Pandas rolling 进行向量化计算
        rolling = series.rolling(window=windowsize)
        mvaves = rolling.mean()
        stds = rolling.std()

        # 计算置信区间 Z-Score
        z_score = stats.norm.ppf((1 + confidence) / 2)
        lows = mvaves - (z_score * stds)
        highs = mvaves + (z_score * stds)

        return mvaves.tolist(), highs.tolist(), lows.tolist()

    def calc_ave(
        self, begin: int, end: int, dt: int, column_index: int
    ) -> Tuple[str, float, float]:
        """
        计算选定区间内某列数据的平均值和标准差。

        Args:
            begin: 起始行索引。
            end: 结束行索引 (None 表示到末尾)。
            dt: 步长。
            column_index: 列索引。

        Returns:
            Tuple[str, float, float]: (图例名称, 平均值, 标准差)。
        """
        self.check_column_index(column_index)

        s_end = end if end is not None else self.row_num
        subset = self.df.iloc[begin:s_end:dt, column_index]

        legend = (
            self.data_heads[column_index]
            if column_index < len(self.data_heads)
            else "Unknown"
        )
        ave = float(subset.mean())
        std = float(subset.std(ddof=1))  # 样本标准差

        return legend, ave, std

    def check_column_index(self, column_index: int) -> None:
        """辅助方法：检查列索引合法性。"""
        if column_index < 0 or column_index >= self.column_num:
            raise ValueError(
                f"列索引 {column_index} 越界 (范围 0-{self.column_num - 1})。"
            )

    def apply_shift(self, threshold_x, shift_value, col_idx=1, operator=">="):
        """
        根据 X 轴数值对 Y 轴数据进行平移修正 (处理周期性跳变等)。

        Args:
            threshold_x: X 轴阈值。
            shift_value: 平移量 (加到 Y 轴)。
            col_idx: 操作的列索引。
            operator: 判断条件 ">=" 或 "<="。
        """
        if self.df is None or self.df.empty:
            return
        if operator == ">=":
            self.df.loc[self.df[0] >= threshold_x, col_idx] += shift_value
        elif operator == "<=":
            self.df.loc[self.df[0] <= threshold_x, col_idx] += shift_value

    # =========================================================================
    #  可视化功能 (Visualization - Integrated)
    # =========================================================================

    @staticmethod
    def _generate_lighter_color(color, factor=0.4):
        """
        [内部工具] 生成指定颜色的浅色版本。
        用于绘制原始数据的散点图，使其不喧宾夺主。
        """
        try:
            c_rgb = mcolors.to_rgb(color)
            h, l, s = colorsys.rgb_to_hls(*c_rgb)  # noqa: E741
            # 提高亮度 (L)
            new_l = l + (1.0 - l) * factor
            return colorsys.hls_to_rgb(h, new_l, s)
        except Exception:
            return color

    @staticmethod
    def _setup_plot_style(ax, title, xlabel, ylabel, config):
        """[内部工具] 应用标准图表样式 (标题、标签、网格等)。"""
        ax.set_title(
            title, fontsize=config["title_fontsize"], pad=20, fontweight="bold"
        )
        ax.set_xlabel(
            xlabel, fontsize=config["label_fontsize"], labelpad=12, fontweight="bold"
        )
        ax.set_ylabel(
            ylabel, fontsize=config["label_fontsize"], labelpad=12, fontweight="bold"
        )

        ax.grid(
            True, linestyle="-", alpha=config["grid_alpha"], color="gray", linewidth=0.8
        )

        ax.tick_params(axis="both", which="major", labelsize=config["tick_fontsize"])
        ax.legend(fontsize=12, frameon=True, framealpha=0.8, loc="best", borderpad=0.5)

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        style: str = "professional",
        color: Optional[str] = None,
        window_size: Optional[int] = None,
        show: bool = False,
        save_path: Optional[str] = None,
        auto_title: bool = True,
    ) -> None:
        """
        [实例方法] 绘制当前 XVG 对象的数据。

        Args:
            ax: Matplotlib Axes 对象。如果为 None，则新建窗口。
            style: 预设配色风格 ('professional', 'bright' 等)。
            color: 强制指定主色调 (Hex 或 Name)。覆盖 style 设置。
            window_size: 滑动平均窗口大小。None 则使用默认值 100。
            show: 是否立即调用 plt.show()。
            save_path: 图片保存路径。如果提供，将自动保存。
            auto_title: 是否自动设置标题和轴标签。默认为 True。
        """
        if self.df.empty:
            print("错误: 数据为空，无法绘图。")
            return

        config = self.STYLE_CONFIG
        win_size = window_size if window_size is not None else config["window_size"]

        # 1. 创建画布
        if ax is None:
            fig, ax = plt.subplots(figsize=config["figsize"], dpi=config["dpi"])

        # 2. 确定颜色
        if color is None:
            cmap_name = self.COLORMAP_MAP.get(style, "viridis")
            cmap = plt.get_cmap(cmap_name)
            main_color = cmap(0.1)  # 默认取色带前端颜色
        else:
            main_color = color

        scatter_color = self._generate_lighter_color(main_color, factor=0.4)

        # 3. 准备数据 (默认 X=col0, Y=col1)
        xs = self.df.iloc[:, 0]
        ys = self.df.iloc[:, 1]

        # 计算滑动平均曲线
        try:
            mvaves, _, _ = self.calc_mvave(win_size, 0.95, 1)
            # 截去前端 NaN 部分以匹配绘图
            xs_smooth = xs[win_size:]
            ys_smooth = mvaves[win_size:]
        except Exception:
            # 数据不足或出错时，回退到原始数据
            xs_smooth = xs
            ys_smooth = ys

        label_base = self.ylabel if self.ylabel else "Data"

        # 4. 绘图
        # 散点图 (原始数据背景)
        ax.scatter(
            xs,
            ys,
            color=scatter_color,
            alpha=0.5,
            s=config["scatter_size"],
            label=f"{label_base} (Raw)",
            zorder=3,
        )
        # 线图 (平滑趋势)
        ax.plot(
            xs_smooth,
            ys_smooth,
            color=main_color,
            linewidth=config["line_width"],
            alpha=0.9,
            label=f"{label_base} (Avg)",
            zorder=4,
        )

        # 5. 美化与保存
        if auto_title:
            self._setup_plot_style(ax, self.title, self.xlabel, self.ylabel, config)

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"图表已保存至: {save_path}")
        elif ax is None and not show:
            # 默认保存行为
            default_out = self.xvgfile.replace(".xvg", "_plot.png")
            plt.tight_layout()
            plt.savefig(default_out, dpi=300, bbox_inches="tight")
            print(f"图表已保存至: {default_out}")

        if show:
            plt.show()

    @classmethod
    def plot_files(
        cls,
        file_list: List[str],
        style: str = "professional",
        output_name: Optional[str] = None,
        window_size: Optional[int] = None,
        show: bool = True,
    ):
        """
        [类方法] 批量读取并绘制多个 XVG 文件。
        替代了原 visualization.py 中的功能。

        Args:
            file_list: 文件路径列表。
            style: 配色风格。
            output_name: 合成图表的保存路径。
            window_size: 滑动平均窗口大小。
            show: 是否显示图表窗口。
        """
        valid_files = [Path(f) for f in file_list if Path(f).exists()]
        if not valid_files:
            print("错误: 没有有效的文件可供绘制。")
            return

        config = cls.STYLE_CONFIG
        win_size = window_size if window_size is not None else config["window_size"]

        # 生成颜色序列
        cmap_name = cls.COLORMAP_MAP.get(style, "viridis")
        cmap = plt.get_cmap(cmap_name)
        indices = np.linspace(0, 1, len(valid_files))
        colors = [cmap(i) for i in indices]

        fig, ax = plt.subplots(figsize=config["figsize"], dpi=config["dpi"])
        meta_title, meta_xlabel, meta_ylabel = "Unknown", "X", "Y"

        print(f"--- 开始批量绘制 (窗口大小: {win_size}) ---")

        for i, file_path in enumerate(valid_files):
            try:
                xvg_obj = cls(str(file_path))

                # 使用第一个文件的元数据作为主图标题
                if i == 0:
                    meta_title = xvg_obj.title
                    meta_xlabel = xvg_obj.xlabel
                    meta_ylabel = xvg_obj.ylabel

                # 准备颜色和数据
                main_color = colors[i]
                scatter_color = cls._generate_lighter_color(main_color, factor=0.4)

                xs = xvg_obj.data_columns[0]
                ys = xvg_obj.data_columns[1]
                mvaves, _, _ = xvg_obj.calc_mvave(win_size, 0.95, 1)

                label_base = file_path.stem.replace(".xvg", "")

                # 绘图
                ax.scatter(
                    xs,
                    ys,
                    color=scatter_color,
                    alpha=0.5,
                    s=config["scatter_size"],
                    label=f"{label_base} (Raw)",
                    zorder=3,
                )
                ax.plot(
                    xs[win_size:],
                    mvaves[win_size:],
                    color=main_color,
                    linewidth=config["line_width"],
                    alpha=0.9,
                    label=f"{label_base} (Avg)",
                    zorder=4,
                )

                print(f"已处理: {file_path.name}")

            except Exception as e:
                print(f"错误: 处理文件 {file_path.name} 时失败: {e}")
                continue

        cls._setup_plot_style(ax, meta_title, meta_xlabel, meta_ylabel, config)
        plt.tight_layout(pad=3.0)

        if output_name:
            save_path = output_name
        else:
            save_path = valid_files[0].parent / f"{valid_files[0].stem}_combined.png"

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图表已保存至: {save_path}")
        if show:
            plt.show()

    # =========================================================================
    #  文件合并功能 (Merging - Integrated)
    # =========================================================================

    @classmethod
    def combine_files(
        cls,
        input_files: List[str],
        columns_to_extract: List[List[int]],
        output_file: str = "combined_output.xvg",
        legends: Optional[List[str]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        begin: int = 0,
        end: Optional[int] = None,
        dt: int = 1,
        yshrink: float = 1.0,
    ) -> None:
        """
        [类方法] 合并多个 XVG 文件的特定列到一个新文件。

        功能：
            1. 读取每个文件的指定列 (columns_to_extract)。
            2. 对齐时间轴 (默认以第一个文件为准)。
            3. 将所有提取的数据列组合成一个新的宽表 DataFrame。
            4. 输出为新的 .xvg 文件。

        Args:
            input_files: 输入文件路径列表。
            columns_to_extract: 每个文件对应的要提取的列索引列表。
                                例如 [[0, 1], [1]] 表示第一个文件取第0,1列，第二个取第1列。
            output_file: 输出文件路径。
            legends: 自定义图例列表（可选，若不填则尝试自动获取）。
            xlabel, ylabel, title: 自定义元数据（可选）。
            begin, end, dt: 数据切片参数。
            yshrink: Y 轴数据缩放因子 (乘数)。
        """
        # 1. 验证输入
        if not input_files:
            raise ValueError("输入文件列表不能为空。")
        if len(input_files) != len(columns_to_extract):
            raise ValueError("输入文件数量与提取列配置的数量不匹配。")

        print("开始合并 XVG 文件 (Optimized)...")

        collected_titles = []
        final_legends = []
        data_dict = {}

        # 图例迭代器
        legend_iter = iter(legends) if legends else None

        # 读取第一个文件作为基准 (Reference)
        ref_xvg = cls(input_files[0])

        # 定义切片对象
        slice_obj = slice(begin, end, dt)

        # 提取时间轴 (Time)
        time_data = ref_xvg.df.iloc[slice_obj, 0].values
        data_dict["Time"] = time_data

        # 迭代处理所有文件
        for f_idx, filename in enumerate(input_files):
            # 优化：复用第一个文件的对象，避免重复读取
            if f_idx == 0:
                current_xvg = ref_xvg
            else:
                current_xvg = cls(filename)

            # 收集标题以供参考
            if current_xvg.title and current_xvg.title not in collected_titles:
                collected_titles.append(current_xvg.title)

            target_cols = columns_to_extract[f_idx]
            valid_cols = []

            # 过滤列索引
            for col_idx in target_cols:
                current_xvg.check_column_index(col_idx)
                # 警告并跳过：除了第一个文件外，不要重复提取时间列(0)
                if f_idx > 0 and col_idx == 0:
                    print(f"警告: 已跳过 {filename} 的时间列 (0) 以避免重复。")
                    continue
                valid_cols.append(col_idx)

            if not valid_cols:
                continue

            # 批量提取数据并应用缩放
            extracted_block = current_xvg.df.iloc[slice_obj, valid_cols] * yshrink

            for i, original_col_idx in enumerate(valid_cols):
                col_data = extracted_block.iloc[:, i].values

                # 长度对齐处理 (取交集长度)
                if len(col_data) != len(time_data):
                    min_len = min(len(col_data), len(time_data))
                    col_data = col_data[:min_len]

                # --- 确定图例名称 (优先级: 自定义 > 源文件图例 > 自动命名) ---
                col_name = None

                # 1. 优先使用传入的 legends
                if legend_iter:
                    try:
                        col_name = next(legend_iter)
                    except StopIteration:
                        pass

                # 2. 尝试从源文件获取 legend
                if col_name is None:
                    if hasattr(current_xvg, "legends") and current_xvg.legends:
                        # 假设 XVG legends 列表不包含 Time 列 (即 col 0)，所以 index = col_idx - 1
                        leg_idx = original_col_idx - 1
                        if 0 <= leg_idx < len(current_xvg.legends):
                            col_name = current_xvg.legends[leg_idx]

                # 3. 默认命名
                if col_name is None:
                    col_name = f"File{f_idx}_Col{original_col_idx}"

                final_legends.append(col_name)
                data_dict[col_name] = col_data

        # 2. 构建最终 DataFrame
        combined_df = pd.DataFrame(data_dict)

        # 3. 创建输出对象
        out_xvg = cls(output_file, is_file=False, new_file=True)
        out_xvg.df = combined_df

        # 填充元数据
        out_xvg.title = title if title else " & ".join(collected_titles)
        out_xvg.xlabel = xlabel if xlabel else ref_xvg.xlabel
        out_xvg.ylabel = ylabel if ylabel else ref_xvg.ylabel
        out_xvg.legends = final_legends
        out_xvg.comments.append(
            f"# 合并自 {len(input_files)} 个文件 (Pandas Optimized)。"
        )

        out_xvg.save(output_file)
