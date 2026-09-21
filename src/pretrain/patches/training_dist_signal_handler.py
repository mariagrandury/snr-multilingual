# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import os
import signal

import torch


def get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    return world_size


def get_device(local_rank=None):
    backend = torch.distributed.get_backend()
    if backend == 'nccl':
        if local_rank is None:
            device = torch.device('cuda')
        else:
            device = torch.device(f'cuda:{local_rank}')
    elif backend == 'gloo':
        device = torch.device('cpu')
    else:
        raise RuntimeError
    return device


def all_gather_item(item, dtype, group=None, async_op=False, local_rank=None):
    if not torch.distributed.is_available() or \
       not torch.distributed.is_initialized():
        return [item]

    device = get_device(local_rank)

    if group is not None:
        group_size = group.size()
    else:
        group_size = get_world_size()

    tensor = torch.tensor([item], device=device, dtype=dtype)
    output_tensors = [
        torch.zeros(1, dtype=tensor.dtype, device=tensor.device)
        for _ in range(group_size)
    ]
    torch.distributed.all_gather(output_tensors, tensor, group, async_op)
    output = [elem.item() for elem in output_tensors]
    return output


class DistributedSignalHandler:
    """Raise a flag when a signal arrives; training.py checks it once per
    iteration and saves a checkpoint before exiting (--exit-signal-handler).

    SIGUSR2 is what SLURM sends before the walltime (`--signal=SIGUSR2@3600`).
    A PREEMPTION is different: it arrives as SIGTERM, straight to the ranks,
    which have no handler for it and die where they stand — on 2026-09-21 a
    1.7B on `preemptable` lost 334 iterations x 21 nodes that way, with the
    batch script's own TERM trap firing far too late to matter.
    MEGATRON_EXIT_ON_SIGTERM=1 adds SIGTERM to the signals caught, so the run
    checkpoints inside the 4 min grace period and the requeued job resumes.

    Opt-in, because it also changes what `scancel` does: the run then saves
    before stopping, and SLURM's KillWait (30 s here, against 240 s of
    preemption grace) may not leave the async save time to land. Only jobs
    submitted to a preemptable partition set it — snr-multilingual's
    launch_trainings.py --partition, through launch_pretraining_cscs.sh.

    (Local patch, NOT upstream: snr-multilingual/src/pretrain/patches/.)
    """

    def __init__(self, sig=signal.SIGUSR2):
        self.sigs = [sig]
        if os.environ.get("MEGATRON_EXIT_ON_SIGTERM", "") not in ("", "0"):
            self.sigs.append(signal.SIGTERM)

    def signals_received(self):
        return self._signal_received

    def __enter__(self):
        self._signal_received = False
        self.released = False
        self.original_handlers = {s: signal.getsignal(s) for s in self.sigs}

        def handler(signum, frame):
            self._signal_received = True

        for sig in self.sigs:
            signal.signal(sig, handler)

        # The one line that says the patch is live in this job: grep the log
        # for it before trusting a preemption to checkpoint.
        if int(os.environ.get("RANK", 0)) == 0:
            names = ", ".join(signal.Signals(s).name for s in self.sigs)
            print(f"[exit-signal-handler] catching {names}", flush=True)

        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False

        for sig, original in self.original_handlers.items():
            signal.signal(sig, original)
        self.released = True
        return True
