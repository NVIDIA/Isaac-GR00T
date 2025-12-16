# Make training work w/ and w/o distributed training.
import torch


def is_dist_avail_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    if is_dist_avail_and_initialized():
        return torch.distributed.get_rank()
    return 0


def barrier():
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()
