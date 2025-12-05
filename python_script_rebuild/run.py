import argparse
import logging
import sys
from pathlib import Path
from utils.runner import GromacsSimulationManager, SimulationConfig

# 这里的导入是为了预留接口，如果你想把 analyzer 也合并进来的话
# from utils.analyzer import TrajectoryAnalyzer


def setup_logging(debug: bool = False, log_file: str = "simulation.log"):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a"),  # 'a' append mode
        ],
    )


def handle_simulation(args):
    """处理模拟任务的逻辑"""
    work_dir = Path(args.work_dir).resolve()
    setup_logging(args.debug, log_file=str(work_dir / "gromacs_runner.log"))
    logger = logging.getLogger(__name__)

    logger.info(f"初始化模拟，工作目录: {work_dir}")

    # 1. 构建配置
    config = SimulationConfig(
        work_dir=work_dir,
        gro_input=args.input_gro,
        ntomp=args.ntomp,
        ntmpi=args.ntmpi,
        gpu_id=args.gpu_id,
        use_gpu=not args.cpu_only,
    )

    # 2. 实例化管理器
    manager = GromacsSimulationManager(config)

    # 3. 确定要运行的步骤
    steps_to_run = []
    if args.steps == "all":
        steps_to_run = ["em", "eq", "md"]
    else:
        steps_to_run = args.steps.split(",")  # 例如 "eq,md"

    # 4. 执行管线
    try:
        manager.run_pipeline(steps_to_run)
    except Exception as e:
        logger.critical(f"管线执行中断: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="GROMACS Automation Suite")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # === 子命令: run (用于运行模拟) ===
    parser_run = subparsers.add_parser("run", help="执行 GROMACS 模拟流程")

    # 路径参数
    parser_run.add_argument("--work-dir", "-d", default=".", help="工作目录")
    parser_run.add_argument(
        "--input-gro",
        "-f",
        default="minim.gro",
        help="初始结构文件名 (默认: minim.gro)",
    )

    # 流程控制
    parser_run.add_argument(
        "--steps",
        default="all",
        help="要运行的阶段: 'all' 或用逗号分隔 'em,eq,md' (默认: all)",
    )

    # 硬件资源
    parser_run.add_argument(
        "--ntomp", type=int, default=0, help="OpenMP 线程数 (0=自动, 默认: 0)"
    )
    parser_run.add_argument("--ntmpi", type=int, default=1, help="MPI 进程数 (默认: 1)")
    parser_run.add_argument("--gpu-id", type=str, default="0", help="GPU ID (默认: 0)")
    parser_run.add_argument("--cpu-only", action="store_true", help="强制仅使用 CPU")
    parser_run.add_argument("--debug", action="store_true", help="开启调试日志")

    # === 子命令: analyze (预留接口) ===
    # parser_analyze = subparsers.add_parser("analyze", help="分析轨迹数据")
    # parser_analyze.add_argument(...)

    args = parser.parse_args()

    if args.command == "run":
        handle_simulation(args)
    elif args.command == "analyze":
        print("Analysis module not connected yet.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
