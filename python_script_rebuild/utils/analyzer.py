import subprocess
import logging
import pandas as pd  # 新增引用，用于构建 DataFrame
from pathlib import Path
from dataclasses import dataclass
from typing import List

# 假设 XVG 和 XVG_combiner 类在同一文件中或已正确导入
from xvg_handler import XVG, XVG_combiner

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """分析配置类，用于统一管理文件路径和参数"""

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
        return self.work_dir / self.md_dir_name

    @property
    def data_path(self) -> Path:
        return self.work_dir / self.data_dir_name


class GROMACSBase:
    """GROMACS分析基类"""

    def __init__(self, config: AnalysisConfig):
        self.cfg = config
        self._ensure_dirs()

    def _ensure_dirs(self):
        self.cfg.data_path.mkdir(exist_ok=True, parents=True)

    def run_command(
        self, cmd: List[str], input_text: str = None, description: str = ""
    ):
        """运行GROMACS命令的通用包装器"""
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
            # 某些 GROMACS 版本警告可能写在 stdout
            if e.stdout:
                logger.error(f"STDOUT: {e.stdout}")
            raise RuntimeError(f"GROMACS command failed: {e.stderr}") from e


class PolymerEnergyAnalyzer(GROMACSBase):
    """
    聚合物能量分析器 (适配 Pandas 版 XVG)
    """

    def __init__(
        self, config: AnalysisConfig, edr_file: Path, energy_type: str = "Coul-SR"
    ):
        super().__init__(config)
        self.edr_file = edr_file
        self.energy_type = energy_type
        self.mole_names = [
            f"{self.cfg.mole_prefix}{i}" for i in range(1, self.cfg.mole_count + 1)
        ]

        self.output_dir = self.cfg.data_path / "energy" / energy_type
        self.output_dir.mkdir(parents=True, exist_ok=True)

        file_prefix = f"all_pairs_{energy_type.replace('-', '_')}"
        self.output_file = self.output_dir / f"{file_prefix}.xvg"
        self.summed_output = self.output_dir / f"summed_{file_prefix}.xvg"

    def _generate_pairs(self) -> List[str]:
        terms = []
        n = len(self.mole_names)
        for i in range(n):
            for j in range(i + 1, n):
                terms.append(
                    f"{self.energy_type}:{self.mole_names[i]}-{self.mole_names[j]}"
                )
        return terms

    def analyze(self) -> Path:
        logger.info(f"开始分析 {self.energy_type} ...")
        energy_terms = self._generate_pairs()

        # 1. 提取能量
        cmd = ["gmx", "energy", "-f", str(self.edr_file), "-o", str(self.output_file)]
        input_data = "\n".join(energy_terms) + "\n"

        # 即使文件存在也建议重新运行，或者添加存在性检查逻辑
        self.run_command(
            cmd, input_text=input_data, description=f"Extract {self.energy_type}"
        )

        # 2. 汇总数据
        self._create_summed_file()
        return self.summed_output

    def _create_summed_file(self):
        try:
            # 读取原始 XVG
            original_xvg = XVG(str(self.output_file))

            # --- 修改开始: 适配 Pandas 结构的求和 ---
            if original_xvg.df.empty:
                raise ValueError(f"文件 {self.output_file} 为空，无法汇总。")

            # 假设第0列是 Time，第1列及之后是各项能量
            # 使用 Pandas 直接求和，速度快且代码简洁
            time_series = original_xvg.df.iloc[:, 0]
            # axis=1 表示按行对列求和
            total_energy_series = original_xvg.df.iloc[:, 1:].sum(axis=1)

            # 构建新的 DataFrame
            new_df = pd.DataFrame({0: time_series, 1: total_energy_series})

            # 创建新的 XVG 对象用于保存
            summed_xvg = XVG(str(self.summed_output), is_file=False, new_file=True)
            summed_xvg.df = new_df  # 直接赋值给 df
            # --- 修改结束 ---

            summed_xvg.data_heads = ["Time", f"Total {self.energy_type}"]
            summed_xvg.xlabel = "Time (ps)"
            summed_xvg.ylabel = "Energy (kJ/mol)"
            summed_xvg.legends = [f"Total {self.energy_type}"]

            summed_xvg.save(str(self.summed_output))
            logger.info(f"汇总能量文件已保存: {self.summed_output}")

        except Exception as e:
            logger.error(f"处理XVG文件失败: {e}")
            raise


class TrajectoryAnalyzer(GROMACSBase):
    """
    轨迹分析器 (适配 Pandas 版 XVG)
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
        self.run_command(cmd, input_text="0\n", description="Remove PBC")

    def analyze_structure(self, least_squares_fit, group_for_RMSD, group_for_Rg):
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
            input_text=f"{least_squares_fit}\n{group_for_RMSD}\n",
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
            cmd_rg, input_text=f"{group_for_Rg}\n", description="Calculate Rg"
        )

    def analyze_hbonds(self):
        self.hbond_dir.mkdir(exist_ok=True)
        pairs = []
        n = self.cfg.mole_count

        for i in range(1, n):
            for j in range(i + 1, n + 1):
                pairs.append((i, j))

        logger.info(f"将处理 {len(pairs)} 对聚合物的氢键。")

        hbond_files = []

        for i, j in pairs:
            out_name = self.hbond_dir / f"hbond_{i},{j}.xvg"
            hbond_files.append(out_name)

            if out_name.exists() and out_name.stat().st_size > 0:
                continue

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
                # 根据力场和GROMACS版本，可能不需要显示指定 -de -ae
                # "-de", "O", "N",
                # "-ae", "O", "N",
            ]

            group_selection = (
                f"{self.cfg.mole_prefix}_{i}\n{self.cfg.mole_prefix}_{j}\n"
            )

            try:
                self.run_command(
                    cmd, input_text=group_selection, description=f"Hbond {i}-{j}"
                )
            except Exception:
                logger.warning(f"氢键计算失败: Pair {i}-{j}")

        self._combine_hbonds(hbond_files)

    def _combine_hbonds(self, files: List[Path]):
        valid_files = [f for f in files if f.exists()]
        if not valid_files:
            logger.warning("没有生成有效的氢键文件，跳过合并。")
            return

        combined_out = self.hbond_dir / "Hbond_combine.xvg"
        summed_out = self.hbond_dir / "Hbond_sum.xvg"

        # 配置提取列：第一个文件取 Time(0), Count(1)，后续只取 Count(1)
        col_cfg = [[0, 1]] + [[1] for _ in range(len(valid_files) - 1)]

        try:
            # --- 修改点: 转换 Path 为 str ---
            input_files_str = [str(f) for f in valid_files]

            combiner = XVG_combiner(
                input_files=input_files_str,
                columns_to_extract=col_cfg,
                output_file=str(combined_out),
                xlabel="Time (ps)",
                ylabel="Hbond Count",
            )
            combiner.combine()

            self._sum_hbonds(combined_out, summed_out)
        except Exception as e:
            logger.error(f"合并氢键文件失败: {e}")

    def _sum_hbonds(self, input_file: Path, output_file: Path):
        try:
            xvg = XVG(str(input_file))
            if xvg.df.empty:
                return

            # --- 修改点: 适配 Pandas 结构的求和 ---
            time_col = xvg.df.iloc[:, 0]
            # 第1列及以后是各个对的氢键数
            total_hbonds = xvg.df.iloc[:, 1:].sum(axis=1)

            res = XVG(str(output_file), is_file=False, new_file=True)
            res.df = pd.DataFrame({0: time_col, 1: total_hbonds})
            res.data_heads = ["Time", "Total Hbonds"]
            res.legends = ["Total Hbonds"]
            res.ylabel = "Count"
            res.save(str(output_file))
            logger.info(f"氢键总和已保存: {output_file}")
        except Exception as e:
            logger.error(f"氢键求和失败: {e}")


class EnergyReRunAnalyzer(GROMACSBase):
    """
    能量重运行分析器
    """

    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.rerun_tpr = self.cfg.md_path / "rerun.tpr"
        self.rerun_edr = self.cfg.md_path / "md_rerun.edr"
        self.energy_dir = self.cfg.data_path / "energy"
        self.energy_final = self.energy_dir / "Total_Interaction_Energy.xvg"

    def run(self):
        self._prepare_tpr()
        self._run_rerun()

        coul_recip = self._analyze_coul_recip()
        short_range_files = self._analyze_short_range()

        # 合并列表
        all_files = list(short_range_files)
        if coul_recip and coul_recip.exists():
            all_files.append(coul_recip)

        self._combine_all_energies(all_files)

    def _prepare_tpr(self):
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
        out_file = self.energy_dir / "Coul_recip.xvg"
        # 即使文件存在也覆盖，保证数据最新
        cmd = ["gmx", "energy", "-f", str(self.rerun_edr), "-o", str(out_file)]
        try:
            self.run_command(
                cmd, input_text="Coul.-recip.\n", description="Extract Coul-recip"
            )
            return out_file
        except RuntimeError:
            logger.warning("无法提取 Coul.-recip. (可能不适用于当前力场或设置)")
            return Path("non_existent_file")

    def _analyze_short_range(self) -> List[Path]:
        results = []
        for etype in ["LJ-SR", "Coul-SR"]:
            analyzer = PolymerEnergyAnalyzer(
                self.cfg, self.rerun_edr, energy_type=etype
            )
            # analyze 方法返回的是 summed_output
            res_file = analyzer.analyze()
            results.append(res_file)
        return results

    def _combine_all_energies(self, files: List[Path]):
        valid_files = [f for f in files if f.exists()]
        if not valid_files:
            return

        # 配置列提取：所有输入文件都是两列（Time, Energy），我们只需要合并 Energy
        # 第一个文件取 0, 1，后面取 1
        col_cfg = [[0, 1]] + [[1] for _ in range(len(valid_files) - 1)]

        # --- 修改点: 转换 Path 为 str ---
        input_files_str = [str(f) for f in valid_files]

        combiner = XVG_combiner(
            input_files=input_files_str,
            columns_to_extract=col_cfg,
            output_file=str(self.energy_final),
            xlabel="Time (ps)",
            ylabel="Energy (kJ/mol)",
            title="Total Interaction Energy Components",
        )
        combiner.combine()

        # 可选：计算总相互作用能 (Sum of all components)
        try:
            final_xvg = XVG(str(self.energy_final))
            if not final_xvg.df.empty:
                # 添加一列 Sum
                total_col = final_xvg.df.iloc[:, 1:].sum(axis=1)
                final_xvg.df["Total_Sum"] = total_col
                final_xvg.legends.append("Total_Sum")
                final_xvg.save(str(self.energy_final))  # 覆盖保存包含总和的文件
                logger.info(f"最终总能量(含求和)文件生成: {self.energy_final}")
        except Exception as e:
            logger.error(f"计算最终总能量失败: {e}")
