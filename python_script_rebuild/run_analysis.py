import logging
from pathlib import Path
import sys

# 假设你提供的类保存在 gmx_analysis_tools.py 中
# 如果在同一个文件，可以直接使用，无需 import
try:
    from utils.analyzer import (
        AnalysisConfig,
        TrajectoryAnalyzer,
        EnergyReRunAnalyzer,
    )
except ImportError:
    print("错误: 找不到 gmx_analysis_tools.py 或依赖库 (xvg_handler)。")
    sys.exit(1)

# 1. 设置日志 (非常重要，否则你看不到 logger.info 的输出)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # 输出到控制台
        logging.FileHandler("analysis.log"),  # 输出到文件
    ],
)

logger = logging.getLogger("MainScript")


def main():
    # ============================
    # 1. 配置参数
    # ============================
    # 定义你的工作目录（存放 .tpr, .xtc, .ndx 等文件的父目录）
    work_directory = Path("./simulation_workdir")

    # 初始化配置对象
    # 注意：根据你的实际体系修改 mole_prefix 和 mole_count
    config = AnalysisConfig(
        work_dir=work_directory,
        mole_prefix="poly",  # index文件中聚合物链的前缀 (例如 poly_1, poly_2...)
        mole_count=10,  # 聚合物链的数量
        # 如果你的文件名不是默认的，可以在这里覆盖
        xtc_name="md.xtc",
        tpr_name="md.tpr",
        ndx_name="index.ndx",
    )

    logger.info(f"开始分析项目，工作目录: {config.work_dir}")

    # ============================
    # 2. 轨迹结构分析 (RMSD, Rg, Hbond)
    # ============================
    try:
        trj_analyzer = TrajectoryAnalyzer(config)

        # 2.1 预处理：去除 PBC (生成 md_noPBC.xtc)
        logger.info(">>> 阶段 1: 轨迹预处理")
        trj_analyzer.preprocess_trajectory()

        # 2.2 结构分析：RMSD 和 Rg
        # 注意：这里的参数对应 GROMACS make_ndx 中的组号
        # 通常 "1" 是 Protein/Polymer, "0" 是 System。请根据你的 index.ndx 调整。
        logger.info(">>> 阶段 2: 计算 RMSD 和 Rg")
        trj_analyzer.analyze_structure(
            least_squares_fit="1",  # 用于最小二乘拟合的组
            group_for_RMSD="1",  # 用于计算 RMSD 的组
            group_for_Rg="1",  # 用于计算回转半径的组
        )

        # 2.3 氢键分析
        # 会自动计算聚合物链之间的所有成对氢键并合并
        logger.info(">>> 阶段 3: 计算链间氢键")
        trj_analyzer.analyze_hbonds()

    except Exception as e:
        logger.error(f"轨迹分析过程中发生错误: {e}")
        # 根据需要决定是否继续执行后续步骤

    # ============================
    # 3. 能量重运行分析 (Rerun)
    # ============================
    try:
        logger.info(">>> 阶段 4: 能量重运行分析 (Rerun)")
        energy_analyzer = EnergyReRunAnalyzer(config)

        # 这将执行以下步骤：
        # 1. 生成 rerun.tpr
        # 2. 运行 mdrun -rerun
        # 3. 提取 Coul-SR, LJ-SR, Coul-recip
        # 4. 合并所有能量项并计算总和
        energy_analyzer.run()

    except Exception as e:
        logger.error(f"能量分析过程中发生错误: {e}")

    logger.info("所有分析任务完成。")


if __name__ == "__main__":
    main()
