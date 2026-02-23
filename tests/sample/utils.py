import os
import torch
import torch.distributed as dist


def suppress_output(rank):
    """
    Suppresses output from print statements in distributed training, such that only the root process (rank 0) prints messages, and non-root processes do not print anything unless a 'force' flag is specified.
    
    Args:
        rank (int): The rank of the current process in the distributed training setup. Used to determine whether to suppress or allow output.
    """

    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force:
            builtin_print("rank #%d:" % rank, *args, **kwargs)
        elif rank == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed() -> torch.device:
    """
    Initialize distributed training with PyTorch and NCCL backend.
    
    This function sets up a distributed training environment using PyTorch's distributed module.
    It retrieves the world size and rank from environment variables, initializes the distributed
    process group, sets the current CUDA device to the specified rank, and performs a warm-up
    operation to reduce initial latency in NCCL communication.
    
    Args:
        None
    
    Returns:
        torch.device: A CUDA device object corresponding to the current process rank.
    
    Raises:
        RuntimeError: If the environment variables WORLD_SIZE or RANK are not set or invalid.
        RuntimeError: If NCCL initialization fails or CUDA is not available.
    """

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Warm up NCCL to avoid first-time latency
    if world_size > 1:
        x = torch.ones(1, device=device)
        dist.all_reduce(x)
        torch.cuda.synchronize(device)

    suppress_output(rank)
    return device
