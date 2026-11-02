"""Training examples and leakage-safe, seed-reproducible negative sampling (uniform or UserPop).

Negatives are drawn for the GLOBAL batch on every rank with the same random stream, so the sampled training data do
not depend on the number of GPUs.
"""
from __future__ import annotations

import logging
import math
import random
from collections import Counter
from dataclasses import dataclass, fields

LOG = logging.getLogger('lmstar')
STRATEGIES = ('uniform', 'uniform_userpop')


def eligible_examples(data, split, doc_index):
    """Training examples with >= 1 positive and >= 1 item-level negative; evaluation examples unchanged."""
    examples = data.examples(split)
    if split != 'train':
        return examples
    valid = [e for e in examples if any(s in doc_index for s in e.positives) and data.negatives(e)]
    LOG.info('Train sampling: %d eligible / %d total; %d skipped (no item-level negatives or positives)',
             len(valid), len(examples), len(examples) - len(valid))
    if not valid:
        raise RuntimeError('No training examples have eligible negatives. Check item-level pools.')
    return valid


@dataclass
class SamplingStats:
    examples: int = 0
    eligible_gt_k: int = 0
    eligible_total: int = 0
    nonzero_total: int = 0
    selected_total: int = 0
    hard_selected: int = 0
    hard_score_total: int = 0
    uniform_selected: int = 0
    uniform_score_total: int = 0
    hard_shortfall: int = 0

    def add(self, other):
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def metrics(self):
        return {
            'eligible_gt_k_rate': self.eligible_gt_k / max(1, self.examples),
            'userpop_nonzero_rate': self.nonzero_total / max(1, self.eligible_total),
            'hard_selected_mean': self.hard_selected / max(1, self.examples),
            'hard_selected_fraction': self.hard_selected / max(1, self.selected_total),
            'hard_shortfall_mean': self.hard_shortfall / max(1, self.examples),
            'hard_score_mean': self.hard_score_total / max(1, self.hard_selected),
            'uniform_score_mean': self.uniform_score_total / max(1, self.uniform_selected),
        }


class NegativeSampler:
    """K negatives among ``HistoryDataset.negatives``: uniform, or ``uniform_userpop`` (a share drawn among the
    statements the user wrote most often, weighted by count, the rest uniform)."""

    def __init__(self, data, num_negatives, config=None, seed=40):
        config = config or {}
        self.data, self.k = data, int(num_negatives)
        self.strategy = config.get('strategy', 'uniform')
        self.ratio = float(config.get('userpop_ratio', 0.25))
        self.multiplier = int(config.get('hard_pool_multiplier', 3))
        if self.strategy not in STRATEGIES:
            raise ValueError(f'negative_sampling.strategy must be one of {STRATEGIES}')
        if not 0 <= self.ratio <= 1 or not math.isfinite(self.ratio):
            raise ValueError('negative_sampling.userpop_ratio must be finite and in [0, 1]')
        if self.multiplier < 1:
            raise ValueError('negative_sampling.hard_pool_multiplier must be >= 1')
        self.seed = int(seed)
        self.rng = random.Random(seed)                 # independent of the shuffling / model random streams
        self.counts = {}
        if self.strategy == 'uniform_userpop':
            # full authorised FIT history (strict past), not the packed prompt
            for ex in data.examples('train'):
                counts = Counter()
                for h in data.history_for(ex.interaction_id):
                    counts.update(h['positives'])
                self.counts[ex.interaction_id] = counts

    def start_epoch(self, epoch):
        self.rng = random.Random(self.seed + 7919 * int(epoch))

    def sample(self, ex, eligible, stats=None):
        """Unique negatives: all of them (shuffled) if <= K, else hard-first (UserPop) then uniform."""
        pool = list(dict.fromkeys(eligible))
        if len(pool) != len(eligible):
            raise ValueError('Eligible negative pool must contain unique statement IDs')
        counts = self.counts.get(ex.interaction_id, {})
        nonzero = sum(counts.get(s, 0) > 0 for s in pool)
        hard, requested = [], 0
        if len(pool) <= self.k:
            chosen = uniform = self.rng.sample(pool, len(pool))
        elif self.strategy == 'uniform':
            chosen = uniform = self.rng.sample(pool, self.k)
        else:
            requested = min(self.k, int(math.floor(self.k * self.ratio + 0.5)))
            scored = [(s, counts.get(s, 0)) for s in pool if counts.get(s, 0) > 0]
            self.rng.shuffle(scored)                  # random tie breaks, including ties at the pool cut-off
            scored.sort(key=lambda pair: pair[1], reverse=True)
            top = scored[:self.multiplier * requested] if requested else []
            for _ in range(min(requested, len(top))):  # weighted draws without replacement
                draw = self.rng.randrange(sum(weight for _, weight in top))
                for j, (sid, weight) in enumerate(top):
                    if draw < weight:
                        hard.append(sid)
                        top.pop(j)
                        break
                    draw -= weight
            hard_set = set(hard)
            uniform = self.rng.sample([s for s in pool if s not in hard_set], self.k - len(hard))
            chosen = hard + uniform
        if stats is not None:
            stats.examples += 1
            stats.eligible_gt_k += int(len(pool) > self.k)
            stats.eligible_total += len(pool)
            stats.nonzero_total += nonzero
            stats.selected_total += len(chosen)
            stats.hard_selected += len(hard)
            stats.hard_score_total += sum(counts.get(s, 0) for s in hard)
            stats.uniform_selected += len(uniform)
            stats.uniform_score_total += sum(counts.get(s, 0) for s in uniform)
            stats.hard_shortfall += requested - len(hard)
        return chosen


def sample_rows(examples, data, index, sampler, stats=None):
    """Positives + sampled negatives (statement indices) of every example of the global batch."""
    rows, npos = [], []
    for ex in examples:
        pos = list(dict.fromkeys(s for s in ex.positives if s in index))
        pool = [s for s in data.negatives(ex) if s in index]
        neg = sampler.sample(ex, pool, stats)
        if not pos or not neg:
            raise RuntimeError(f'Invalid training example {ex.interaction_id}')
        rows.append([index[s] for s in pos + neg])
        npos.append(len(pos))
    return rows, npos


def batch_tensors(rows, npos):
    """(distinct statements in first-appearance order, index [B, M], positive mask, negative mask)."""
    import torch
    unique = list(dict.fromkeys(s for row in rows for s in row))   # each statement encoded once per (micro)batch
    position = {s: j for j, s in enumerate(unique)}
    width = max(map(len, rows))
    idx = torch.zeros((len(rows), width), dtype=torch.long)
    pos_mask = torch.zeros_like(idx, dtype=torch.bool)
    neg_mask = torch.zeros_like(idx, dtype=torch.bool)
    for j, row in enumerate(rows):
        idx[j, :len(row)] = torch.tensor([position[s] for s in row])
        pos_mask[j, :npos[j]] = True
        neg_mask[j, npos[j]:len(row)] = True
    return unique, idx, pos_mask, neg_mask
