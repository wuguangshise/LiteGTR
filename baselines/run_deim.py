"""
在 DEIM 环境里启动 DEIM 官方的 train.py —— 由 train_baselines.py 调用，不需要手动运行。

    python run_deim.py <DEIM 仓库目录> <train.py 的参数...>

单卡训练，不用 torchrun、不建进程组：DEIM 的 setup_distributed() 初始化失败时会走它自带的
单进程模式（"Not init distributed mode"：不包 DDP，SyncBN 不转换，普通 DataLoader）。单卡时
这和官方 `torchrun --nproc_per_node=1` 在数值上等价（world size 1 的 DDP / SyncBN 本来就不做同步），
而且不依赖 NCCL / gloo / libuv —— Windows 版 PyTorch 没有 NCCL，这正是在 Windows 上容易出错的地方。

DEIM 只有一处没有照顾单进程模式：加载 ImageNet 预训练骨干时（engine/backbone/hgnetv2.py）
直接调用 torch.distributed.get_rank() / barrier()。这里让这两个函数在没有进程组时返回 0 / 什么都不做。
DEIM 仓库里的文件一个都不改。
"""
import os
import runpy
import sys


def main() -> None:
    deim_dir = os.path.abspath(sys.argv[1])

    import torch.distributed as dist

    def init_process_group(*args, **kwargs):
        raise RuntimeError("single-process training: no process group")

    get_rank, barrier = dist.get_rank, dist.barrier
    dist.init_process_group = init_process_group
    dist.get_rank = lambda *a, **k: get_rank(*a, **k) if dist.is_initialized() else 0
    dist.barrier = lambda *a, **k: barrier(*a, **k) if dist.is_initialized() else None

    script = os.path.join(deim_dir, "train.py")
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, deim_dir)
    os.chdir(deim_dir)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
