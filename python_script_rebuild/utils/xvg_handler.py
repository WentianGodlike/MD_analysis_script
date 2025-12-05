import os
import time
import io
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


class XVG:
    """
    XVG 类：用于解析和处理 GROMACS xvg 文件。
    基于 Pandas 进行了深度优化，以实现高性能数据处理。
    """

    def __init__(
        self,
        xvgfile: str,
        is_file: bool = True,
        new_file: bool = False,
    ) -> None:
        self.xvgfile = xvgfile
        self.comments: List[str] = []
        self.title: str = ""
        self.xlabel: str = ""
        self.ylabel: str = ""
        self.legends: List[str] = []

        # 数据存储：使用 Pandas DataFrame 代替旧版的 list of lists
        self.df: pd.DataFrame = pd.DataFrame()

        # 元数据默认值
        self.view: str = "0.15, 0.15, 0.75, 0.85"
        self.world_xmin: Optional[float] = None
        self.world_xmax: Optional[float] = None
        self.world_ymin: Optional[float] = None
        self.world_ymax: Optional[float] = None

        if not new_file and is_file:
            if not os.path.exists(xvgfile):
                raise FileNotFoundError(f"未检测到文件 {xvgfile}！")

            # 使用分离式解析方法
            self._load_xvg()

    @property
    def row_num(self) -> int:
        """返回数据行数"""
        return self.df.shape[0]

    @property
    def column_num(self) -> int:
        """返回数据列数"""
        return self.df.shape[1]

    @property
    def data_heads(self) -> List[str]:
        """基于标签和图例生成表头列表"""
        heads = [self.xlabel] if self.xlabel else ["X"]

        if self.legends:
            heads.extend(self.legends)
        elif self.ylabel:
            # 如果有多列数据但只有一个 ylabel，进行适当处理
            if self.column_num > 1:
                heads.append(self.ylabel)
                # 如果还有剩余列，自动填充命名
                for i in range(2, self.column_num):
                    heads.append(f"{self.ylabel}_{i}")
        return heads

    @property
    def data_columns(self) -> List[List[float]]:
        """
        向后兼容属性：
        将数据以 list of lists (列优先) 的形式返回，以兼容旧代码。
        """
        if self.df.empty:
            return []
        return [self.df[col].tolist() for col in self.df.columns]

    def _load_xvg(self) -> None:
        """
        高效加载 XVG 文件：
        将元数据（Metadata）与数值数据（Data）分离解析。
        """
        metadata_lines = []
        data_buffer = []

        # 1. 读取文件并分离注释/元数据与数据
        with open(self.xvgfile, "r", encoding="UTF-8") as f:
            for line in f:
                sline = line.strip()
                if not sline:
                    continue
                if sline.startswith(("#", "@")):
                    metadata_lines.append(sline)
                else:
                    data_buffer.append(line)

        # 2. 解析元数据
        self._parse_metadata(metadata_lines)

        # 3. 使用 Pandas 解析数据 (向量化操作，速度极快)
        if data_buffer:
            # 使用 io.StringIO 将字符串缓冲作为文件对象处理
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
        """解析 GROMACS 特有的元数据行 (@ 和 #)"""
        title_flag = False
        xlabel_flag = False
        ylabel_flag = False
        for line in lines:
            if line.startswith("#"):
                self.comments.append(line)
                continue

            # 解析 '@' 命令
            if line.startswith("@"):
                clean_line = line.lstrip("@").strip()

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
                    # 例如: @ s0 legend "Potential"
                    self.legends.append(line.split('"')[-2])
                elif " view " in line:
                    self.view = clean_line.replace("view", "").strip()
                elif " world xmin " in line:
                    self.world_xmin = float(line.split()[-1])

    def save(self, outxvg: str, check: bool = True) -> None:
        """将 XVG 类转储（保存）到文件"""
        if check:
            if self.df.empty:
                raise ValueError("无法保存空的 XVG 数据。")

        # 确保图例数量与数据列（不含X轴）对齐
        data_col_count = self.column_num - 1
        if len(self.legends) < data_col_count:
            # 自动生成缺失的图例
            base_label = self.ylabel if self.ylabel else "Data"
            for i in range(len(self.legends), data_col_count):
                self.legends.append(f"{base_label}_{i}")

        time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        with open(outxvg, "w", encoding="UTF-8") as fo:
            # 写入文件头 (Header)
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

            # 写入数据 (Data)
            # 使用循环进行精确的格式控制 (模拟原始 GROMACS 格式)
            # 标准的 pandas to_csv 可能无法完美对齐空格

            # 转为 numpy 数组进行迭代比 df.iterrows 更快
            np_data = self.df.values
            for row in np_data:
                # 自定义格式化：通常第一列是时间（整数或浮点），其余为数据
                line_str = ""
                for val in row:
                    if val.is_integer():
                        line_str += f"{int(val):>8d} "
                    else:
                        line_str += f"{val:>16.6f} "
                fo.write(line_str.strip() + "\n")

        print(f"XVG 文件已成功保存至 {outxvg}。")

    def calc_mvave(
        self, windowsize: int, confidence: float, column_index: int
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        使用 Pandas 向量化操作计算移动平均。
        显著快于基于循环的方法。
        """
        if windowsize <= 0 or windowsize > self.row_num // 2:
            raise ValueError("窗口大小 (windowsize) 设置不当。")
        if not (0 < confidence < 1):
            raise ValueError("置信度 (confidence) 必须在 (0, 1) 之间。")
        if column_index >= self.column_num:
            raise ValueError("列索引 (column_index) 超出范围。")

        # 提取 Series
        series = self.df.iloc[:, column_index]

        # 向量化滚动计算
        # min_periods=windowsize 模拟了起始处存在 NaN 的行为
        rolling = series.rolling(window=windowsize)
        mvaves = rolling.mean()
        stds = rolling.std()

        # 计算置信区间
        # ppf: 百分位点函数 (CDF 的反函数)
        z_score = stats.norm.ppf((1 + confidence) / 2)

        lows = mvaves - (z_score * stds)
        highs = mvaves + (z_score * stds)

        # 返回列表以匹配原始函数的返回签名
        return mvaves.tolist(), highs.tolist(), lows.tolist()

    def calc_ave(
        self, begin: int, end: int, dt: int, column_index: int
    ) -> Tuple[str, float, float]:
        """利用切片计算选定列的平均值"""
        if column_index >= self.column_num:
            raise ValueError(f"列索引 {column_index} 超出范围。")

        # 处理切片用的 None
        s_end = end if end is not None else self.row_num

        # DataFrame 切片操作基本是 O(1) 的
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
        """检查列索引是否有效"""
        if column_index < 0 or column_index >= self.column_num:
            raise ValueError(
                f"列索引 {column_index} 越界 (范围 0-{self.column_num - 1})。"
            )

    def plot_xvg_file(
        self,
        moveavg: bool = True,
        window_size: int = 100,
        ax: plt.Axes = None,
        save_path: str = None,
    ) -> None:
        """
        优化后的绘图函数。
        Args:
            ax: 可选的 matplotlib Axes 对象，用于在现有图形上绘制。
            save_path: 可选保存路径，如果提供则立即保存图片。
        """
        # 获取数据 (假设第0列是X，第1列是Y)
        xs = self.df.iloc[:, 0]
        ys = self.df.iloc[:, 1]

        # 设置画布
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7), dpi=120)

        color_raw = "#0066CC"
        color_avg = "#004080"

        # 绘制原始数据
        ax.scatter(
            xs,
            ys,
            color=color_raw,
            alpha=0.5,
            s=10,
            label=f"{self.ylabel} (Row_data)",
            zorder=3,
        )

        # 绘制移动平均
        if moveavg:
            # 计算逻辑：假设 X 轴是时间 (ps)
            mvaves, _, _ = self.calc_mvave(window_size, 0.7, 1)
            # mvaves 在起始处包含 NaNs，matplotlib 可以自动处理

            # 估算时间窗口用于标签显示
            try:
                dt = xs.iloc[1] - xs.iloc[0]
                time_window_ps = int(window_size * dt)
                avg_label = f"{self.ylabel} ({time_window_ps}ps Avg)"
            except:  # noqa: E722
                avg_label = f"{self.ylabel} ({window_size} Avg)"

            ax.plot(
                xs,
                mvaves,
                color=color_avg,
                linewidth=2.5,
                alpha=0.9,
                label=avg_label,
                zorder=4,
            )

        # 装饰图表
        ax.set_title(self.title, fontsize=16, pad=20, fontweight="bold")

        ax.set_xlabel(self.xlabel, fontsize=14, labelpad=12, fontweight="bold")
        print(self.xlabel)
        ax.set_ylabel(self.ylabel, fontsize=14, labelpad=12, fontweight="bold")
        ax.grid(True, linestyle="-", alpha=0.3, color="gray", linewidth=0.8)

        ax.legend(fontsize=12, frameon=True, framealpha=0.8, loc="best", borderpad=0.5)
        ax.tick_params(axis="both", which="major", labelsize=11)

        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            print(f"图表已保存至: {save_path}")
        elif ax is None:
            # 如果是内部创建的 figure，显示或保存为默认名
            default_out = self.xvgfile.replace(".xvg", "_plot.png")
            plt.tight_layout()
            plt.savefig(default_out, dpi=300)
            print(f"图表已保存至: {default_out}")

    def apply_shift(self, threshold_x, shift_value, col_idx=1, operator=">="):
        """
        根据 X 轴的值，对 Y 轴数据进行平移修正
        :param threshold_x: X轴阈值
        :param shift_value: 要加上的值 (可以是负数)
        :param col_idx: 要修改的列索引，默认是1
        :param operator: 判断条件 ">=" 或 "<="
        """
        if self.df is None or self.df.empty:
            return

        if operator == ">=":
            self.df.loc[self.df[0] >= threshold_x, col_idx] += shift_value
        elif operator == "<=":
            self.df.loc[self.df[0] <= threshold_x, col_idx] += shift_value


class XVG_combiner:
    """
    XVG 合并工具类 (Pandas 优化版)。
    用于将多个 XVG 文件的特定列合并为一个文件。
    """

    def __init__(
        self,
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
    ):
        self.input_files = input_files
        self.columns_to_extract = columns_to_extract
        self.output_file = output_file
        self.legends = legends
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.begin = begin
        self.end = end
        self.dt = dt
        self.yshrink = yshrink

        self._validate_inputs()

    def _validate_inputs(self):
        if not self.input_files:
            raise ValueError("输入文件列表不能为空。")
        if len(self.input_files) != len(self.columns_to_extract):
            raise ValueError("输入文件数量与提取列配置的数量不匹配。")
        # 实际运行时请取消注释以下检查（模拟环境中不检查文件存在性）
        # for f in self.input_files:
        #     if not os.path.exists(f):
        #         raise FileNotFoundError(f"未找到文件: {f}")

    def combine(self):
        """执行 XVG 文件合并过程 (优化版)。"""
        print("开始合并 XVG 文件 (Optimized)...")

        # 1. 初始化
        collected_titles = []
        final_legends = []

        # 使用字典收集数据列，最后一次性构建 DataFrame，效率高于逐列赋值
        data_dict = {}

        # 准备图例迭代器，方便按需提取
        legend_iter = iter(self.legends) if self.legends else None

        # 2. 读取第一个文件作为基准 (Reference)
        # 假设 XVG 类读取文件开销较大，我们只在循环外初始化一次 ref_xvg
        # 如果 self.input_files[0] 在后续也需要处理，我们已经在循环逻辑中覆盖了
        ref_xvg = XVG(self.input_files[0])

        # 定义切片对象，pandas 支持 slice 对象直接索引
        # 如果 end 是 None，slice 会自动处理到末尾
        slice_obj = slice(self.begin, self.end, self.dt)

        # 提取时间轴 (基准)
        # .values 转换为 numpy array，避免索引对齐问题，且速度更快
        time_data = ref_xvg.df.iloc[slice_obj, 0].values
        data_dict["Time"] = time_data

        # 3. 迭代处理文件
        for f_idx, filename in enumerate(self.input_files):
            # 优化：如果是第一个文件，复用 ref_xvg，避免重复 IO
            if f_idx == 0:
                current_xvg = ref_xvg
            else:
                current_xvg = XVG(filename)

            # 收集标题
            if current_xvg.title and current_xvg.title not in collected_titles:
                collected_titles.append(current_xvg.title)

            # 获取当前文件需要提取的列索引列表
            target_cols = self.columns_to_extract[f_idx]

            # 过滤列：如果不是第一个文件，跳过索引为 0 的列 (Time)
            valid_cols = []
            for col_idx in target_cols:
                current_xvg.check_column_index(col_idx)  # 保留原有的检查逻辑
                if f_idx > 0 and col_idx == 0:
                    print(f"警告: 已跳过 {filename} 的时间列 (0) 以避免重复。")
                    continue
                valid_cols.append(col_idx)

            if not valid_cols:
                continue

            # --- 核心优化点 ---
            # 批量提取数据：一次 iloc 提取所有列，而不是在循环中提取
            # 乘以标量 yshrink 是向量化操作，非常快
            extracted_block = current_xvg.df.iloc[slice_obj, valid_cols] * self.yshrink

            # 遍历提取出的块，分配图例名称并存入字典
            # zip(valid_cols, range(len(valid_cols))) 用于对应原始列号和提取块中的位置
            for i, original_col_idx in enumerate(valid_cols):
                # 获取数据列 (Series -> numpy array)
                # 使用 .values 确保长度对齐，忽略原始索引
                col_data = extracted_block.iloc[:, i].values

                # 检查数据长度一致性
                if len(col_data) != len(time_data):
                    # 简单截断或填充以匹配时间轴 (根据 Time 列长度)
                    # 这里选择切片以匹配较短者，保证 DataFrame 创建成功
                    min_len = min(len(col_data), len(time_data))
                    col_data = col_data[:min_len]
                    # 注意：如果数据严重不齐，可能需要更复杂的 merge 逻辑，但为了保持原功能接口，此处做简单处理

                # 确定列名/图例
                col_name = None

                # 1. 尝试使用用户提供的覆盖图例
                if legend_iter:
                    try:
                        col_name = next(legend_iter)
                    except StopIteration:
                        pass

                # 2. 如果未提供，尝试从源文件自动获取
                if col_name is None:
                    # 假设 XVG 对象的 legends 属性存储了 Y 轴的标签
                    # 通常第0列是 Time，所以 legend index = col_idx - 1
                    if hasattr(current_xvg, "legends") and current_xvg.legends:
                        # 计算对应的 legend索引
                        # 注意：这里假设 legends 列表不包含 Time 列的说明
                        leg_idx = original_col_idx - 1
                        if 0 <= leg_idx < len(current_xvg.legends):
                            col_name = current_xvg.legends[leg_idx]

                # 3. 最后的兜底命名
                if col_name is None:
                    col_name = f"File{f_idx}_Col{original_col_idx}"

                final_legends.append(col_name)
                data_dict[col_name] = col_data

        # 4. 构建最终 DataFrame
        # dict -> DataFrame 非常高效
        combined_df = pd.DataFrame(data_dict)

        # 5. 输出保存
        out_xvg = XVG(self.output_file, is_file=False, new_file=True)

        # 填充元数据
        out_xvg.title = self.title if self.title else " & ".join(collected_titles)
        out_xvg.xlabel = self.xlabel if self.xlabel else ref_xvg.xlabel
        out_xvg.ylabel = self.ylabel if self.ylabel else ref_xvg.ylabel
        out_xvg.legends = final_legends
        out_xvg.comments.append(
            f"# 合并自 {len(self.input_files)} 个文件 (Pandas Optimized)。"
        )

        out_xvg.df = combined_df
        out_xvg.save(self.output_file)


# -----------------------------------------------------------
# 使用示例 / Example Usage
# -----------------------------------------------------------
if __name__ == "__main__":
    # 1. 单个文件分析示例
    try:
        # 假设有个文件叫 energy.xvg
        xvg = XVG("/mnt/d/yry/50_poly_box/pull/25000/pull_pullf_new.xvg")
        xvg.plot_xvg_file(
            window_size=50,
            save_path="/mnt/d/yry/50_poly_box/pull/25000/pull_pullf_new.png",
        )
        pass
    except Exception as e:
        print(e)

    # 2. 合并文件示例
    files = [
        "/mnt/d/yry/50_poly_box/data/energy/Coul_SR/summed_all_pairs_Coul_SR.xvg",
        "/mnt/d/yry/50_poly_box/data/energy/LJ-SR/LJ_SR.xvg",
    ]
    cols = [[0, 1], [1]]  # 提取每个文件的第1列
    combiner = XVG_combiner(files, cols, output_file="merged.xvg")
    combiner.combine()
