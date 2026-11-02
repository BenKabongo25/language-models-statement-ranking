from __future__ import annotations

import random

import numpy as np

from baselines.data import IndexedInteraction
from baselines.models.base import TorchRanker
from baselines.sampling import ItemNegativeSampler


def _dev(name: str):
    import torch

    return torch.device("cuda" if name == "cuda" and torch.cuda.is_available() else "cpu")


class CLER(TorchRanker):
    """Contrastive Learning for Explanation Ranking (CLER).

    We preserve the published ID-based representation, dropout, normalized weighted
    user-item query, normalized explanation embeddings, temperature scaling and
    row-balanced NT-BXent objective.  The benchmark-specific adaptation is deliberate:
    negatives are sampled from the target item's statement pool instead of coming from
    the global in-batch explanation matrix.
    """

    def fit(self, interactions: list[IndexedInteraction], *, validation=None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        cfg = self.cfg
        mcfg = cfg["model"]
        tcfg = cfg["training"]
        dev = _dev(self.device)
        dim = int(mcfg.get("dimension", 768))
        dropout = float(mcfg.get("dropout", 0.88))
        mu = float(mcfg.get("mu", 0.55))
        tau = float(mcfg.get("temperature", 0.2))

        class Net(nn.Module):
            def __init__(self, nu: int, ni: int, ne: int):
                super().__init__()
                self.u = nn.Embedding(nu, dim)
                self.i = nn.Embedding(ni, dim)
                self.e = nn.Embedding(ne, dim)
                self.du = nn.Dropout(dropout)
                self.di = nn.Dropout(dropout)
                # CLER initializes the ID embeddings from a uniform distribution.
                nn.init.uniform_(self.u.weight, -0.05, 0.05)
                nn.init.uniform_(self.i.weight, -0.05, 0.05)
                nn.init.uniform_(self.e.weight, -0.05, 0.05)

            def query(self, u, i, training: bool = True):
                ue = self.u(u)
                ie = self.i(i)
                if training:
                    ue = self.du(ue)
                    ie = self.di(ie)
                return F.normalize(mu * ue + (1.0 - mu) * ie, p=2, dim=-1)

            def statements(self, e):
                return F.normalize(self.e(e), p=2, dim=-1)

        self.net = Net(self.data.n_users, self.data.n_items, self.data.n_statements).to(dev)
        opt = torch.optim.AdamW(
            self.net.parameters(),
            lr=float(tcfg.get("lr", 1e-4)),
            weight_decay=float(tcfg.get("weight_decay", 1e-2)),
        )
        sampler = ItemNegativeSampler(
            self.data,
            interactions,
            cfg["sampling"].get("candidate_source", "fit_only"),
            self.seed,
        )
        rng = random.Random(self.seed)
        batch_size = int(tcfg.get("batch_size", 512))
        epochs = int(tcfg.get("epochs", 10))
        nneg = int(tcfg.get("num_negatives", 64))
        history = []
        eligible = [x for x in interactions if sampler.pool(x)]

        for epoch in range(1, epochs + 1):
            rng.shuffle(eligible)
            total = 0.0
            n_rows = 0
            self.net.train()
            for start in range(0, len(eligible), batch_size):
                batch = eligible[start : start + batch_size]
                rows = []
                for x in batch:
                    neg = sampler.sample(x, nneg, replace=False)
                    if not neg:
                        continue
                    pos = list(x.statements)
                    rows.append((x.user, x.item, pos, neg))
                if not rows:
                    continue

                # Vectorized, padded row-local candidate matrix.  Padding never enters
                # the objective because valid/positive/negative masks are explicit.
                width = max(len(pos) + len(neg) for _, _, pos, neg in rows)
                users = torch.tensor([r[0] for r in rows], dtype=torch.long, device=dev)
                items = torch.tensor([r[1] for r in rows], dtype=torch.long, device=dev)
                cand = torch.zeros((len(rows), width), dtype=torch.long, device=dev)
                valid = torch.zeros((len(rows), width), dtype=torch.bool, device=dev)
                positive = torch.zeros((len(rows), width), dtype=torch.bool, device=dev)
                for row_id, (_, _, pos, neg) in enumerate(rows):
                    ids = pos + neg
                    cand[row_id, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
                    valid[row_id, : len(ids)] = True
                    positive[row_id, : len(pos)] = True

                q = self.net.query(users, items, training=True)  # [B,D]
                e = self.net.statements(cand)                    # [B,C,D]
                logits = torch.einsum("bd,bcd->bc", q, e) / tau
                negative = valid & ~positive

                # NT-BXent balances positives and negatives independently per row.
                pos_loss = F.softplus(-logits) * positive
                neg_loss = F.softplus(logits) * negative
                lp = pos_loss.sum(1) / positive.sum(1).clamp_min(1)
                ln = neg_loss.sum(1) / negative.sum(1).clamp_min(1)
                loss = (lp + ln).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(rows)
                n_rows += len(rows)
            history.append({"epoch": epoch, "loss": total / max(n_rows, 1)})

        return {
            "history": history,
            "eligible_interactions": len(eligible),
            "item_level_negative_adaptation": True,
        }

    def score(self, user, item, candidates):
        import torch

        dev = next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            e = torch.as_tensor(candidates, dtype=torch.long, device=dev)
            u = torch.tensor([user], dtype=torch.long, device=dev)
            i = torch.tensor([item], dtype=torch.long, device=dev)
            return (
                self.net.query(u, i, training=False) @ self.net.statements(e).T
            ).squeeze(0).cpu().numpy()
