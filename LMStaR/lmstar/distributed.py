"""Single-node data parallelism started from a plain ``python`` command.

``launch(fn, args, gpus)`` runs ``fn(env, *args)`` in-process (one GPU), spawns one process per visible GPU with
``torch.multiprocessing`` (no torchrun needed), or joins an existing torchrun group (``WORLD_SIZE`` / ``RANK`` set).
The (small) trainable parameter sets are all-reduced explicitly once per optimizer step; there is no DDP wrapper,
which keeps the two-pass gradient-cache replay exact.
"""
from __future__ import annotations

import datetime
import os
import socket
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistEnv:
    rank: int = 0
    world: int = 1
    device: torch.device = torch.device('cpu')

    @property
    def main(self):
        return self.rank == 0

    @property
    def enabled(self):
        return self.world > 1


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _spawned(rank, world, port, backend, timeout_minutes, fn, args):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world),
                      LOCAL_RANK=str(rank))
    _init_and_run(rank, world, rank, backend, timeout_minutes, fn, args)


def _init_and_run(rank, world, local_rank, backend, timeout_minutes, fn, args):
    if backend == 'nccl':
        device = torch.device('cuda', local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    dist.init_process_group(backend, rank=rank, world_size=world,
                            timeout=datetime.timedelta(minutes=int(timeout_minutes)))
    try:
        return fn(DistEnv(rank, world, device), *args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def launch(fn, args=(), gpus=1, backend=None, timeout_minutes=240):
    """Run ``fn(env, *args)`` on ``gpus`` local GPUs (CPU processes with ``backend='gloo'``, used by the tests)."""
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:                    # already inside torchrun
        world, rank = int(os.environ['WORLD_SIZE']), int(os.environ['RANK'])
        local = int(os.environ.get('LOCAL_RANK', rank))
        return _init_and_run(rank, world, local, backend or 'nccl', timeout_minutes, fn, args)
    gpus = int(gpus)
    if gpus <= 1:
        use_cuda = backend != 'gloo' and torch.cuda.is_available()
        device = torch.device('cuda', 0) if use_cuda else torch.device('cpu')
        if use_cuda:
            torch.cuda.set_device(device)
        return fn(DistEnv(0, 1, device), *args)
    backend = backend or 'nccl'
    if backend == 'nccl' and torch.cuda.device_count() < gpus:
        raise RuntimeError(f'Requested {gpus} GPUs but only {torch.cuda.device_count()} are visible '
                           f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")})')
    import torch.multiprocessing as mp
    mp.spawn(_spawned, args=(gpus, _free_port(), backend, timeout_minutes, fn, args), nprocs=gpus, join=True)


# --------------------------------------------------------------------------------------------- collectives
def barrier(env):
    if not env.enabled:
        return
    if env.device.type == 'cuda':
        dist.barrier(device_ids=[env.device.index])
    else:
        dist.barrier()


def broadcast_object(obj, env):
    if not env.enabled:
        return obj
    box = [obj if env.main else None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def all_gather_object(obj, env):
    if not env.enabled:
        return [obj]
    out = [None] * env.world
    dist.all_gather_object(out, obj)
    return out


def all_reduce_sum(values, env):
    """Sum a list of python floats over the ranks."""
    if not env.enabled:
        return list(values)
    t = torch.tensor(list(values), dtype=torch.float64, device=env.device)
    dist.all_reduce(t)
    return t.tolist()


def broadcast_params(params, env):
    if not env.enabled:
        return
    with torch.no_grad():
        for p in params:
            dist.broadcast(p.data, src=0)


def sync_grads(params, env):
    """Sum the gradients over the ranks (a missing gradient counts as zeros), one bucket per dtype / device."""
    if not env.enabled:
        return
    groups = {}
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        groups.setdefault((p.grad.dtype, p.grad.device), []).append(p.grad)
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    for grads in groups.values():
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        for g, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
            g.copy_(synced)


def gather_rows(indices, values, env):
    """All-gather variable-size shards ``(indices[n], values[n, d])``; returns their concatenation."""
    if not env.enabled:
        return indices, values
    n = torch.tensor([indices.numel()], device=env.device)
    sizes = [torch.zeros_like(n) for _ in range(env.world)]
    dist.all_gather(sizes, n)
    sizes = [int(s.item()) for s in sizes]
    width, dim = max(sizes), values.shape[1]
    padded_idx = torch.full((width,), -1, dtype=torch.long, device=env.device)
    padded_val = torch.zeros((width, dim), dtype=values.dtype, device=env.device)
    padded_idx[:indices.numel()] = indices.to(env.device)
    padded_val[:indices.numel()] = values.to(env.device)
    all_idx = [torch.empty_like(padded_idx) for _ in range(env.world)]
    all_val = [torch.empty_like(padded_val) for _ in range(env.world)]
    dist.all_gather(all_idx, padded_idx)
    dist.all_gather(all_val, padded_val)
    return (torch.cat([g[:s] for g, s in zip(all_idx, sizes)]),
            torch.cat([g[:s] for g, s in zip(all_val, sizes)]))
