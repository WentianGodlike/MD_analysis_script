import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List

# 设置模块级日志
logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """模拟配置类：管理文件路径和硬件资源"""

    work_dir: Path

    # === 文件名配置 (默认值) ===
    # 输入文件
    gro_input: str = "minim.gro"  # 初始结构
    top_file: str = "topol.top"
    ndx_file: str = "index.ndx"

    # MDP 文件名
    mdp_em: str = "step4.0_minimization.mdp"
    mdp_eq: str = "step4.1_equilibration.mdp"
    mdp_md: str = "step5_production.mdp"

    # === 硬件与并行计算配置 ===
    # 0 表示自动检测
    ntomp: int = 0  # OpenMP 线程数
    ntmpi: int = 1  # MPI 进程数
    gpu_id: str = "0"  # GPU ID字符串, 例如 "0" 或 "0,1"

    # 任务控制
    use_gpu: bool = True
    max_warn: int = 20

    @property
    def em_dir(self) -> Path:
        return self.work_dir / "em"

    @property
    def eq_dir(self) -> Path:
        return self.work_dir / "equilibration"

    @property
    def md_dir(self) -> Path:
        return self.work_dir / "md"


class GromacsSimulationManager:
    """GROMACS 模拟流程管理器"""

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self._ensure_directories()

    def _ensure_directories(self):
        """确保所有输出目录存在"""
        for d in [self.cfg.em_dir, self.cfg.eq_dir, self.cfg.md_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _run_command(self, cmd: List[str], description: str) -> None:
        """执行 GROMACS 命令的通用包装器"""
        cmd_str = " ".join(str(c) for c in cmd)
        logger.info(f"执行 [{description}]: {cmd_str}")

        try:
            # text=True 自动处理 bytes 解码
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            # 可以根据需要记录 result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"命令失败: {description}")
            logger.error(f"STDERR:\n{e.stderr}")
            raise RuntimeError(f"GROMACS command failed during {description}") from e

    def _grompp(
        self,
        mdp_path: Path,
        gro_path: Path,
        top_path: Path,
        tpr_out: Path,
        desc: str = "grompp",
    ) -> None:
        """通用 grompp 预处理"""
        if not mdp_path.exists():
            raise FileNotFoundError(f"MDP文件未找到: {mdp_path}")

        cmd = [
            "gmx",
            "grompp",
            "-f",
            str(mdp_path),
            "-c",
            str(gro_path),
            "-r",
            str(gro_path),
            "-p",
            str(top_path),
            "-n",
            str(self.cfg.work_dir / self.cfg.ndx_file),
            "-o",
            str(tpr_out),
            "-maxwarn",
            str(self.cfg.max_warn),
        ]
        self._run_command(cmd, f"{desc} (Pre-processing)")

    def _mdrun(self, deffnm: Path, production: bool = False) -> None:
        """通用 mdrun 执行，自动应用硬件配置"""
        cmd = [
            "gmx",
            "mdrun",
            "-deffnm",
            str(deffnm),
            "-v",
            "-ntomp",
            str(self.cfg.ntomp),
            "-ntmpi",
            str(self.cfg.ntmpi),
        ]

        # GPU 设置
        if self.cfg.use_gpu:
            cmd.extend(["-gpu_id", self.cfg.gpu_id])
            # 只有在明确需要高性能且确定支持时才强制指定 -nb gpu
            # 现代 GROMACS 通常能自动优化，但这里保留用户的手动控制
            if production:
                cmd.extend(["-nb", "gpu", "-pme", "gpu", "-update", "gpu"])
            else:
                # EM 阶段通常不支持 pme gpu，Equil 视情况而定
                cmd.extend(["-nb", "gpu"])

        self._run_command(cmd, f"mdrun ({deffnm.name})")

    def run_em(self) -> Path:
        """执行能量最小化"""
        logger.info(">>> 阶段 1: 能量最小化 (EM)")

        mdp = self.cfg.work_dir / self.cfg.mdp_em
        input_gro = self.cfg.work_dir / self.cfg.gro_input
        tpr = self.cfg.em_dir / "em.tpr"
        base_name = self.cfg.em_dir / "em"

        # 如果已经跑完了（检查输出文件），可以选择跳过，这里默认覆盖
        self._grompp(
            mdp, input_gro, self.cfg.work_dir / self.cfg.top_file, tpr, "EM Grompp"
        )
        self._mdrun(base_name, production=False)

        return base_name.with_suffix(".gro")

    def run_equilibration(self, input_gro: Path) -> Path:
        """执行平衡 (NVT/NPT)"""
        logger.info(">>> 阶段 2: 平衡阶段 (Equilibration)")

        mdp = self.cfg.work_dir / self.cfg.mdp_eq
        tpr = self.cfg.eq_dir / "equilibration.tpr"
        base_name = self.cfg.eq_dir / "equilibration"

        self._grompp(
            mdp, input_gro, self.cfg.work_dir / self.cfg.top_file, tpr, "Eq Grompp"
        )
        self._mdrun(base_name, production=False)

        return base_name.with_suffix(".gro")

    def run_production(self, input_gro: Path) -> Path:
        """执行生产 MD"""
        logger.info(">>> 阶段 3: 生产模拟 (Production MD)")

        mdp = self.cfg.work_dir / self.cfg.mdp_md
        tpr = self.cfg.md_dir / "md.tpr"
        base_name = self.cfg.md_dir / "md"

        self._grompp(
            mdp, input_gro, self.cfg.work_dir / self.cfg.top_file, tpr, "MD Grompp"
        )
        self._mdrun(base_name, production=True)

        return base_name.with_suffix(".gro")

    def run_pipeline(self, steps: List[str] = None):
        """
        自动执行指定的模拟管线
        Args:
            steps: 要执行的步骤列表 ["em", "eq", "md"]。如果为 None，则执行全部。
        """
        if steps is None:
            steps = ["em", "eq", "md"]

        current_gro = self.cfg.work_dir / self.cfg.gro_input

        # 1. EM
        if "em" in steps:
            current_gro = self.run_em()
        else:
            # 如果跳过 EM，尝试寻找 EM 的输出作为下一阶段输入
            potential_gro = self.cfg.em_dir / "em.gro"
            if potential_gro.exists():
                current_gro = potential_gro
                logger.info(f"跳过 EM，使用现有结构: {current_gro}")

        # 2. Equilibration
        if "eq" in steps:
            current_gro = self.run_equilibration(current_gro)
        else:
            potential_gro = self.cfg.eq_dir / "equilibration.gro"
            if potential_gro.exists():
                current_gro = potential_gro
                logger.info(f"跳过 Eq，使用现有结构: {current_gro}")

        # 3. MD
        if "md" in steps:
            self.run_production(current_gro)

        logger.info("<<< 所有指定模拟任务完成")
