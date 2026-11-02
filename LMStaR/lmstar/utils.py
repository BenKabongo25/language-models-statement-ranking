"""Seeding, logging, hashing, atomic file writes and the length-bucketed training order."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

LOG = logging.getLogger('lmstar')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(folder=None, filename='train.log', rank=0):
    """Rank 0 logs INFO to the console (and to ``folder/filename``); other ranks only log warnings."""
    LOG.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    LOG.handlers.clear()
    LOG.propagate = False
    prefix = '' if rank == 0 else f'[rank{rank}] '
    fmt = logging.Formatter(f'%(asctime)s | %(levelname)s | {prefix}%(message)s', '%Y-%m-%d %H:%M:%S')
    handlers = [logging.StreamHandler()]
    if folder is not None and rank == 0:
        Path(folder).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(folder) / filename, encoding='utf-8'))
    for handler in handlers:
        handler.setFormatter(fmt)
        LOG.addHandler(handler)


def digest(obj):
    """Short stable hash of a JSON-serialisable object (cache keys and run ids)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:20]


def atomic_write(path, write):
    """Write ``path`` through a temporary file in the same folder and an atomic rename;
    ``write(binary_file)`` produces the content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.stem + '_', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            write(f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_torch_save(obj, path):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.stem + '_', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding='utf-8')


def _json_default(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return str(x)


def batch_order(examples, contexts, seed, epoch, bucket_size):
    """Training order of one epoch: shuffle, cut into buckets (in shuffled order, so no curriculum), sort each
    bucket by prompt length (less padding)."""
    rng = random.Random(seed + 1009 * epoch)
    order = list(range(len(examples)))
    rng.shuffle(order)
    chunks = [order[i:i + bucket_size] for i in range(0, len(order), bucket_size)]
    for chunk in chunks:
        chunk.sort(key=lambda j: len(contexts[examples[j].interaction_id]))
    return [j for chunk in chunks for j in chunk]
