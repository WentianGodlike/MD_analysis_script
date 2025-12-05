import subprocess
import logging
import pandas as pd
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

# 尝试导入 XVG 处理模块，兼容不同环境的导入路径
try:
    from .xvg_handler import XVG
except ImportError:
    from xvg_handler import XVG

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """
    分析配置类。

    用于集中管理 GROMACS 分析所需的文件路径、目录结构以及分子拓扑参数。
    通过 dataclass 自动生成 __init__ 等方法，简化配置对象的创建。

    Attributes:
        work_dir (Path): 工作目录的绝对路径或相对路径。
        mole_prefix (str): 索引文件(index.ndx)中分子组的前缀 (例如 "poly")。
        mole_count (int): 分子/聚合物链的数量。

        # --- 文件与目录命名配置 (默认值) ---
        md_dir_name (str): 存放 MD 轨迹的目录名。
        data_dir_name (str): 分析结果输出目录名。
        xtc_name (str): 轨迹文件名 (.xtc)。
        tpr_name (str): 拓扑文件名 (.tpr)。
        edr_name (str): 能量文件名 (.edr)。
        ndx_name (str): 索引文件名 (.ndx)。
        top_name (str): 拓扑文件名 (.top)。
        mdp_name (str): 模拟参数文件名 (.mdp)。
        gro_name (str): 结构文件名 (.gro)。
    """

    work_dir: Path
    mole_prefix: str = "poly"
    mole_count: int = 12

    # 文件名配置
    md_dir_name: str = "md"
    data_dir_name: str = "data"
    xtc_name: str = "md.xtc"
    tpr_name: str = "md.tpr"
    edr_name: str = "md.edr"
    ndx_name: str = "index.ndx"
    top_name: str = "topol.top"
    mdp_name: str = "step5_production.mdp"
    gro_name: str = "minim.gro"

    @property
    def md_path(self) -> Path:
        """获取 MD 数据目录的完整路径。"""
        return self.work_dir / self.md_dir_name

    @property
    def data_path(self) -> Path:
        """获取分析数据输出目录的完整路径。"""
        return self.work_dir / self.data_dir_name


# =============================================================================
#  辅助函数 (Helper Functions)
# =============================================================================


def _sum_xvg_columns(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    legend_name: str,
    xlabel: str = "Time (ps)",
    ylabel: str = "Energy (kJ/mol)",
) -> None:
    """
    通用工具函数：读取 XVG 文件并对数据列求和。

    读取指定的 XVG 文件，保留第 0 列作为 X 轴（通常是时间），
    将第 1 列及之后的所有列按行求和，生成一个新的 XVG 文件。
    常用于合并多项能量或氢键数量。

    Args:
        input_file: 输入 XVG 文件路径。
        output_file: 输出求和后的 XVG 文件路径。
        legend_name: 新生成曲线的图例名称 (例如 "Total Energy")。
        xlabel: X 轴标签。
        ylabel: Y 轴标签。
    """
    try:
        # 统一转为字符串路径以适配 XVG 类接口
        input_str = str(input_file)
        output_str = str(output_file)

        xvg = XVG(input_str)
        if xvg.df.empty:
            logger.warning(f"文件为空，无法求和: {input_str}")
            return

        # 假设第0列是 Time
        time_series = xvg.df.iloc[:, 0]
        # axis=1 表示按行对列求和 (从第1列开始)
        total_series = xvg.df.iloc[:, 1:].sum(axis=1)

        # 构建新的 DataFrame
        new_df = pd.DataFrame({0: time_series, 1: total_series})

        # 创建并保存新的 XVG 对象
        summed_xvg = XVG(output_str, is_file=False, new_file=True)
        summed_xvg.df = new_df
        summed_xvg.data_heads = ["Time", legend_name]
        summed_xvg.xlabel = xlabel
        summed_xvg.ylabel = ylabel
        summed_xvg.legends = [legend_name]
        summed_xvg.title = f"Summed {legend_name}"

        summed_xvg.save(output_str)
        logger.info(f"汇总文件已保存: {output_str}")

    except Exception as e:
        logger.error(f"处理 XVG 求和失败 ({input_file}): {e}")
        raise


class GROMACSBase:
    """
    GROMACS 分析基类。

    提供基础的文件目录管理和 subprocess 命令执行封装。
    所有具体的分析器都应继承此类。
    """

    def __init__(self, config: AnalysisConfig):
        """
        初始化基类。

        Args:
            config: 分析配置对象，包含路径信息。
        """
        self.cfg = config
        self._ensure_dirs()

    def _ensure_dirs(self):
        """确保输出数据目录存在。"""
        self.cfg.data_path.mkdir(exist_ok=True, parents=True)

    def run_command(
        self, cmd: List[str], input_text: str = None, description: str = ""
    ):
        """
        执行 GROMACS 命令的通用包装器。

        Args:
            cmd: 命令列表，例如 ["gmx", "rms", ...]
            input_text: 需要通过 stdin 传递给命令的交互式文本 (例如组选择 "1\\n1\\n")。
            description: 用于日志记录的命令描述。

        Returns:
            subprocess.CompletedProcess: 命令执行结果对象。

        Raises:
            RuntimeError: 当命令执行返回非零状态码时抛出。
        """
        cmd_str = " ".join(str(c) for c in cmd)
        logger.info(f"执行命令 [{description}]: {cmd_str}")

        try:
            result = subprocess.run(
                cmd, input=input_text, check=True, text=True, capture_output=True
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"命令执行失败: {cmd_str}")
            logger.error(f"STDERR: {e.stderr}")
            if e.stdout:
                logger.error(f"STDOUT: {e.stdout}")
            raise RuntimeError(f"GROMACS command failed: {e.stderr}") from e


class PolymerEnergyAnalyzer(GROMACSBase):
    """
    聚合物能量分析器。

    专注于提取和分析聚合物链之间的非键相互作用能 (Coulomb, LJ 等)。
    """

    def __init__(
        self, config: AnalysisConfig, edr_file: Path, energy_type: str = "Coul-SR"
    ):
        """
        Args:
            config: 分析配置。
            edr_file: 输入的能量文件 (.edr) 路径。
            energy_type: 要提取的能量类型 (例如 "Coul-SR", "LJ-SR")。
        """
        super().__init__(config)
        self.edr_file = edr_file
        self.energy_type = energy_type
        # 根据配置生成分子名称列表
        self.mole_names = [
            f"{self.cfg.mole_prefix}{i}" for i in range(1, self.cfg.mole_count + 1)
        ]

        # 设置输出路径
        self.output_dir = self.cfg.data_path / "energy" / energy_type
        self.output_dir.mkdir(parents=True, exist_ok=True)

        file_prefix = f"all_pairs_{energy_type.replace('-', '_')}"
        self.output_file = self.output_dir / f"{file_prefix}.xvg"
        self.summed_output = self.output_dir / f"summed_{file_prefix}.xvg"

    def _generate_pairs(self) -> List[str]:
        """
        生成所有可能的聚合物链对组合，用于 gmx energy 的选择输入。

        Returns:
            List[str]: 格式化的能量项名称列表 (例如 "Coul-SR:poly1-poly2")。
        """
        terms = []
        n = len(self.mole_names)
        for i in range(n):
            for j in range(i + 1, n):
                terms.append(
                    f"{self.energy_type}:{self.mole_names[i]}-{self.mole_names[j]}"
                )
        return terms

    def analyze(self) -> Path:
        """
        执行能量分析流程。

        1. 生成能量项对列表。
        2. 使用 gmx energy 提取所有对的能量数据。
        3. 对提取出的所有能量项求和，得到总相互作用能。

        Returns:
            Path: 汇总后的 XVG 文件路径。
        """
        logger.info(f"开始分析 {self.energy_type} ...")
        energy_terms = self._generate_pairs()

        # 优化: 使用临时文件传递大量输入，避免管道缓冲区限制 (Pipe Buffer Limit)
        # 对于多链系统，energy_terms 可能非常大，直接传 stdin 可能导致死锁或截断
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
            tmp.write("\\n".join(energy_terms) + "\\n")
            tmp_input_path = tmp.name

        try:
            cmd = [
                "gmx",
                "energy",
                "-f",
                str(self.edr_file),
                "-o",
                str(self.output_file),
            ]

            # 使用 stdin 重定向读取临时文件
            with open(tmp_input_path, "r") as f_in:
                cmd_str = " ".join(cmd)
                logger.info(
                    f"执行命令 [Extract {self.energy_type} (via file)]: {cmd_str}"
                )

                try:
                    subprocess.run(
                        cmd, stdin=f_in, check=True, text=True, capture_output=True
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"GROMACS Energy 提取失败: {e.stderr}")
                    raise RuntimeError(f"GROMACS command failed: {e.stderr}")

        finally:
            # 清理临时文件
            if os.path.exists(tmp_input_path):
                os.remove(tmp_input_path)

        # 2. 汇总数据 (使用通用辅助函数)
        _sum_xvg_columns(
            input_file=self.output_file,
            output_file=self.summed_output,
            legend_name=f"Total {self.energy_type}",
            ylabel="Energy (kJ/mol)",
        )

        return self.summed_output


class TrajectoryAnalyzer(GROMACSBase):
    """
    轨迹分析器。

    提供结构特性分析功能，包括 RMSD, Rg 和氢键分析。
    """

    def __init__(self, config: AnalysisConfig):
        super().__init__(config)

        self.edr_path = self.cfg.md_path / self.cfg.edr_name
        self.tpr_path = self.cfg.md_path / self.cfg.tpr_name
        self.xtc_path = self.cfg.md_path / self.cfg.xtc_name
        self.index_path = self.cfg.work_dir / self.cfg.ndx_name

        self.xtc_no_pbc = self.cfg.md_path / "md_noPBC.xtc"
        self.rmsd_out = self.cfg.data_path / "RMSD_noPBC.xvg"
        self.rg_out = self.cfg.data_path / "Rg_noPBC.xvg"
        self.hbond_dir = self.cfg.data_path / "Hbond"

    def preprocess_trajectory(self):
        """
        预处理轨迹：去除周期性边界条件 (PBC)。
        生成 md_noPBC.xtc 文件。
        """
        if self.xtc_no_pbc.exists():
            logger.info("检测到已存在去PBC的轨迹文件，跳过处理。")
            return

        cmd = [
            "gmx",
            "trjconv",
            "-s",
            str(self.tpr_path),
            "-f",
            str(self.xtc_path),
            "-pbc",
            "nojump",
            "-n",
            str(self.index_path),
            "-o",
            str(self.xtc_no_pbc),
        ]
        # 输入 '0' 选择 System 组
        self.run_command(cmd, input_text="0\\n", description="Remove PBC")

    def analyze_structure(self, least_squares_fit, group_for_RMSD, group_for_Rg):
        """
        分析结构参数：RMSD 和 回转半径 (Rg)。

        Args:
            least_squares_fit: 用于 RMSD 最小二乘拟合的组编号/名称。
            group_for_RMSD: 用于计算 RMSD 的组编号/名称。
            group_for_Rg: 用于计算 Rg 的组编号/名称。
        """
        # RMSD
        cmd_rms = [
            "gmx",
            "rms",
            "-s",
            str(self.tpr_path),
            "-f",
            str(self.xtc_no_pbc),
            "-n",
            str(self.index_path),
            "-o",
            str(self.rmsd_out),
            "-tu",
            "ps",
        ]
        self.run_command(
            cmd_rms,
            input_text=f"{least_squares_fit}\\n{group_for_RMSD}\\n",
            description="Calculate RMSD",
        )

        # Rg
        cmd_rg = [
            "gmx",
            "gyrate",
            "-s",
            str(self.tpr_path),
            "-f",
            str(self.xtc_no_pbc),
            "-n",
            str(self.index_path),
            "-o",
            str(self.rg_out),
            "-tu",
            "ps",
        ]
        self.run_command(
            cmd_rg, input_text=f"{group_for_Rg}\\n", description="Calculate Rg"
        )

    def analyze_hbonds(self):
        """
        并行计算链间氢键。

        使用 ThreadPoolExecutor 并行运行 gmx hbond，显著加快多链体系的分析速度。
        """
        self.hbond_dir.mkdir(exist_ok=True)
        pairs = []
        n = self.cfg.mole_count

        # 生成所有链对
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                pairs.append((i, j))

        logger.info(f"开始计算链间氢键，共 {len(pairs)} 对 (启用并行加速)...")

        # 定义单个任务函数 (不在主线程运行，避免日志混乱)
        def _run_single_hbond_task(pair_idx):
            i, j = pair_idx
            out_name = self.hbond_dir / f"hbond_{i},{j}.xvg"

            # 如果文件已存在且非空，跳过
            if out_name.exists() and out_name.stat().st_size > 0:
                return out_name

            cmd = [
                "gmx",
                "hbond",
                "-f",
                str(self.xtc_path),
                "-s",
                str(self.tpr_path),
                "-n",
                str(self.index_path),
                "-num",
                str(out_name),
                "-tu",
                "ps",
            ]
            group_selection = (
                f"{self.cfg.mole_prefix}_{i}\\n{self.cfg.mole_prefix}_{j}\\n"
            )

            try:
                # 使用 capture_output=True 避免控制台输出混乱
                # 这是一个轻量级的 subprocess 调用
                subprocess.run(
                    cmd,
                    input=group_selection,
                    check=True,
                    text=True,
                    capture_output=True,
                )
                return out_name
            except subprocess.CalledProcessError:
                # 仅在失败时记录，注意多线程下 logger 是线程安全的
                logger.warning(f"氢键计算失败: Pair {i}-{j}")
                return None

        # 使用线程池并发执行
        # 获取 CPU 核心数，限制最大 worker 数以防过载 (通常 IO 也是瓶颈)
        max_workers = min(os.cpu_count() or 4, 8)

        valid_files_set = set()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_pair = {
                executor.submit(_run_single_hbond_task, p): p for p in pairs
            }

            # 处理结果
            for future in as_completed(future_to_pair):
                res = future.result()
                if res:
                    valid_files_set.add(res)

        # 重新整理文件列表以保持 pairs 的顺序 (这对矩阵合并很重要)
        sorted_files = []
        for i, j in pairs:
            f = self.hbond_dir / f"hbond_{i},{j}.xvg"
            if f in valid_files_set:
                sorted_files.append(f)
            elif f.exists() and f.stat().st_size > 0:
                sorted_files.append(f)

        logger.info(f"氢键计算完成，有效文件数: {len(sorted_files)}")
        self._combine_hbonds(sorted_files)

    def _combine_hbonds(self, files: List[Path]):
        """
        合并所有氢键文件并计算总氢键数。

        Args:
            files: 氢键 XVG 文件路径列表。
        """
        if not files:
            logger.warning("没有生成有效的氢键文件，跳过合并。")
            return

        combined_out = self.hbond_dir / "Hbond_combine.xvg"
        summed_out = self.hbond_dir / "Hbond_sum.xvg"

        # 配置提取列：第一个文件取 Time(0), Count(1)，后续只取 Count(1)
        col_cfg = [[0, 1]] + [[1] for _ in range(len(files) - 1)]

        try:
            # 转换 Path 为 str
            input_files_str = [str(f) for f in files]

            # 使用 XVG.combine_files 类方法合并数据
            XVG.combine_files(
                input_files=input_files_str,
                columns_to_extract=col_cfg,
                output_file=str(combined_out),
                xlabel="Time (ps)",
                ylabel="Hbond Count",
                title="Inter-chain Hydrogen Bonds",
            )

            # 使用通用求和函数计算总和
            _sum_xvg_columns(
                input_file=combined_out,
                output_file=summed_out,
                legend_name="Total Hbonds",
                ylabel="Count",
            )

        except Exception as e:
            logger.error(f"合并氢键文件失败: {e}")


class EnergyReRunAnalyzer(GROMACSBase):
    """
    能量重运行分析器 (Rerun Analysis)。

    用于在现有轨迹上使用新参数或提取特定能量项进行重计算。
    """

    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.rerun_tpr = self.cfg.md_path / "rerun.tpr"
        self.rerun_edr = self.cfg.md_path / "md_rerun.edr"
        self.energy_dir = self.cfg.data_path / "energy"
        self.energy_final = self.energy_dir / "Total_Interaction_Energy.xvg"

    def run(self):
        """执行完整的 Rerun 分析流程。"""
        self._prepare_tpr()
        self._run_rerun()

        coul_recip = self._analyze_coul_recip()
        short_range_files = self._analyze_short_range()

        # 合并列表：Short Range 文件 + Coul-Recip (如果存在)
        all_files = list(short_range_files)
        if coul_recip and coul_recip.exists():
            all_files.append(coul_recip)

        self._combine_all_energies(all_files)

    def _prepare_tpr(self):
        """生成用于 rerun 的 tpr 文件。"""
        if self.rerun_tpr.exists():
            return
        cmd = [
            "gmx",
            "grompp",
            "-f",
            str(self.cfg.work_dir / self.cfg.mdp_name),
            "-c",
            str(self.cfg.work_dir / self.cfg.gro_name),
            "-p",
            str(self.cfg.work_dir / self.cfg.top_name),
            "-n",
            str(self.cfg.work_dir / self.cfg.ndx_name),
            "-o",
            str(self.rerun_tpr),
            "-maxwarn",
            "20",
        ]
        self.run_command(cmd, description="Generate Rerun TPR")

    def _run_rerun(self):
        """运行 mdrun -rerun 计算能量。"""
        if self.rerun_edr.exists():
            logger.info("重运行 EDR 已存在，跳过计算。")
            return

        cmd = [
            "gmx",
            "mdrun",
            "-rerun",
            str(self.cfg.md_path / self.cfg.xtc_name),
            "-s",
            str(self.rerun_tpr),
            "-e",
            str(self.rerun_edr),
            "-v",
        ]
        self.run_command(cmd, description="Execute MD Rerun")

    def _analyze_coul_recip(self) -> Path:
        """提取长程静电 (Coul-recip)。"""
        out_file = self.energy_dir / "Coul_recip.xvg"
        cmd = ["gmx", "energy", "-f", str(self.rerun_edr), "-o", str(out_file)]
        try:
            self.run_command(
                cmd, input_text="Coul.-recip.\\n", description="Extract Coul-recip"
            )
            return out_file
        except RuntimeError:
            logger.warning("无法提取 Coul.-recip. (可能不适用于当前力场或设置)")
            return Path("non_existent_file")

    def _analyze_short_range(self) -> List[Path]:
        """提取短程相互作用 (LJ-SR, Coul-SR)。"""
        results = []
        for etype in ["LJ-SR", "Coul-SR"]:
            analyzer = PolymerEnergyAnalyzer(
                self.cfg, self.rerun_edr, energy_type=etype
            )
            res_file = analyzer.analyze()
            results.append(res_file)
        return results

    def _combine_all_energies(self, files: List[Path]):
        """合并所有能量项并计算总和。"""
        valid_files = [f for f in files if f.exists()]
        if not valid_files:
            return

        # 配置列提取：第一个文件取 Time, Energy，后续只取 Energy
        col_cfg = [[0, 1]] + [[1] for _ in range(len(valid_files) - 1)]
        input_files_str = [str(f) for f in valid_files]

        XVG.combine_files(
            input_files=input_files_str,
            columns_to_extract=col_cfg,
            output_file=str(self.energy_final),
            xlabel="Time (ps)",
            ylabel="Energy (kJ/mol)",
            title="Total Interaction Energy Components",
        )

        # 计算总和并覆盖保存
        try:
            xvg = XVG(str(self.energy_final))
            if not xvg.df.empty:
                # 添加一列 Total_Sum
                total_col = xvg.df.iloc[:, 1:].sum(axis=1)
                xvg.df["Total_Sum"] = total_col
                xvg.legends.append("Total_Sum")
                xvg.save(str(self.energy_final))
                logger.info(f"最终总能量(含求和)文件生成: {self.energy_final}")
        except Exception as e:
            logger.error(f"计算最终总能量失败: {e}")
