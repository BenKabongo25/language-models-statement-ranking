from __future__ import annotations

import random
from collections import Counter

import numpy as np

from baselines.data import IndexedInteraction
from baselines.models.base import TorchRanker
from baselines.sampling import ItemNegativeSampler


def _dev(name):
    import torch
    return torch.device("cuda" if name == "cuda" and torch.cuda.is_available() else "cpu")


def _bipartite_edges(pairs, n_left, n_right, device):
    """Weighted symmetric-normalized bipartite edges used by ExpGCN.

    The official RecBole implementation constructs sparse U-I, U-E and I-E matrices.
    Repeated associations therefore coalesce as edge weights.  We reproduce that behavior
    explicitly and return D_l^-1/2 A D_r^-1/2 values for message passing.
    """
    import torch

    if not pairs:
        z = torch.empty(0, dtype=torch.long, device=device)
        return z, z, torch.empty(0, dtype=torch.float32, device=device)

    counts = pairs if isinstance(pairs, Counter) else Counter((int(a), int(b)) for a, b in pairs)
    keys = sorted(counts)
    a = torch.tensor([x[0] for x in keys], dtype=torch.long, device=device)
    b = torch.tensor([x[1] for x in keys], dtype=torch.long, device=device)
    value = torch.tensor([float(counts[x]) for x in keys], dtype=torch.float32, device=device)
    dl = torch.zeros(n_left, dtype=torch.float32, device=device)
    dr = torch.zeros(n_right, dtype=torch.float32, device=device)
    dl.index_add_(0, a, value)
    dr.index_add_(0, b, value)
    w = value / torch.sqrt(dl[a].clamp_min(1e-12) * dr[b].clamp_min(1e-12))
    return a, b, w


def _propagate(left, right, edges, layers: int):
    """LightGCN propagation with mean of layer 0..L, matching official ExpGCN."""
    import torch

    a, b, w = edges
    ls = [left]
    rs = [right]
    l, r = left, right
    for _ in range(int(layers)):
        nl = torch.zeros_like(l)
        nr = torch.zeros_like(r)
        if len(a):
            nl.index_add_(0, a, r[b] * w[:, None])
            nr.index_add_(0, b, l[a] * w[:, None])
        l, r = nl, nr
        ls.append(l)
        rs.append(r)
    return torch.stack(ls, dim=0).mean(0), torch.stack(rs, dim=0).mean(0)


class ExpGCN(TorchRanker):
    """Task-oriented ExpGCN with an item-level explanation-negative adaptation.

    Fidelity points retained from the authors' implementation:
    * three U-I, U-E and I-E bipartite graphs;
    * LightGCN-style linear propagation and mean layer aggregation;
    * explanation score = U-E branch + I-E branch;
    * recommendation U-I graph starts from user/item embeddings enriched by their
      explanation-branch representations;
    * joint BPR recommendation + tag_weight * BPR explanation objective.

    Only the explanation-negative pool is changed: negatives are sampled from statements
    of the target item, as required by this benchmark. Item recommendation negatives remain
    the original unseen-item negatives.
    """

    def fit(self, interactions: list[IndexedInteraction], *, validation=None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        cfg = self.cfg
        mcfg = cfg["model"]
        tcfg = cfg["training"]
        dev = _dev(self.device)
        dim = int(mcfg.get("dimension", 128))
        n_layers = int(mcfg.get("n_layers", 4))   # U-I branch in official code
        m_layers = int(mcfg.get("m_layers", 2))   # U-E/I-E branches
        tag_weight = float(mcfg.get("tag_weight", 1.0))

        class Net(nn.Module):
            def __init__(self, nu, ni, ne):
                super().__init__()
                self.u = nn.Embedding(nu, dim)
                self.i = nn.Embedding(ni, dim)
                self.e = nn.Embedding(ne, dim)
                nn.init.xavier_uniform_(self.u.weight)
                nn.init.xavier_uniform_(self.i.weight)
                nn.init.xavier_uniform_(self.e.weight)

        self.net = Net(self.data.n_users, self.data.n_items, self.data.n_statements).to(dev)

        ui, ue, ie = Counter(), Counter(), Counter()
        for x in interactions:
            ui[(x.user, x.item)] += 1
            for e in x.statements:
                ue[(x.user, e)] += 1
                ie[(x.item, e)] += 1
        self.ui = _bipartite_edges(ui, self.data.n_users, self.data.n_items, dev)
        self.ue = _bipartite_edges(ue, self.data.n_users, self.data.n_statements, dev)
        self.ie = _bipartite_edges(ie, self.data.n_items, self.data.n_statements, dev)

        opt = torch.optim.Adam(
            self.net.parameters(),
            lr=float(tcfg.get("lr", 1e-3)),
            weight_decay=float(tcfg.get("weight_decay", 1e-6)),
        )
        sampler = ItemNegativeSampler(
            self.data,
            interactions,
            cfg["sampling"].get("candidate_source", "fit_only"),
            self.seed,
        )
        rng = random.Random(self.seed)
        user_items = self.data.observed_items_by_user(interactions)
        all_items = list(range(self.data.n_items))
        batch_size = int(tcfg.get("batch_size", 16384))
        epochs = int(tcfg.get("epochs", 500))
        num_neg = int(tcfg.get("num_negatives", 20))
        history = []
        eligible = [x for x in interactions if sampler.pool(x)]

        def unseen_item(user: int) -> int | None:
            seen = user_items.get(user, set())
            if len(seen) >= self.data.n_items:
                return None
            while True:
                j = rng.choice(all_items)
                if j not in seen:
                    return j

        for epoch in range(1, epochs + 1):
            rng.shuffle(eligible)
            total = 0.0
            count = 0
            self.net.train()
            for start in range(0, len(eligible), batch_size):
                U, I, NI, P, NE = [], [], [], [], []
                for x in eligible[start : start + batch_size]:
                    for _ in range(num_neg):
                        ns = sampler.sample(x, 1)
                        if not ns:
                            continue
                        j = unseen_item(x.user)
                        if j is None:
                            continue
                        U.append(x.user)
                        I.append(x.item)
                        NI.append(j)
                        P.append(rng.choice(x.statements))
                        NE.append(ns[0])
                if not U:
                    continue

                # Explanation branches first (official m_layers).
                ua_u, ua_e = _propagate(self.net.u.weight, self.net.e.weight, self.ue, m_layers)
                ia_i, ia_e = _propagate(self.net.i.weight, self.net.e.weight, self.ie, m_layers)

                # Official ExpGCN enriches the U-I branch's layer-0 representation with
                # the task-specific explanation representations before n_layers propagation.
                ui_u, ui_i = _propagate(
                    self.net.u.weight + ua_u,
                    self.net.i.weight + ia_i,
                    self.ui,
                    n_layers,
                )

                u = torch.tensor(U, dtype=torch.long, device=dev)
                i = torch.tensor(I, dtype=torch.long, device=dev)
                ni = torch.tensor(NI, dtype=torch.long, device=dev)
                p = torch.tensor(P, dtype=torch.long, device=dev)
                ne = torch.tensor(NE, dtype=torch.long, device=dev)

                rec_pos = (ui_u[u] * ui_i[i]).sum(-1)
                rec_neg = (ui_u[u] * ui_i[ni]).sum(-1)
                tag_pos = (ua_u[u] * ua_e[p]).sum(-1) + (ia_i[i] * ia_e[p]).sum(-1)
                tag_neg = (ua_u[u] * ua_e[ne]).sum(-1) + (ia_i[i] * ia_e[ne]).sum(-1)

                rec_loss = -F.logsigmoid(rec_pos - rec_neg).mean()
                tag_loss = -F.logsigmoid(tag_pos - tag_neg).mean()
                loss = rec_loss + tag_weight * tag_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(U)
                count += len(U)

            history.append({"epoch": epoch, "loss": total / max(count, 1)})

        self.net.eval()
        with torch.no_grad():
            self._ua_u, self._ua_e = _propagate(self.net.u.weight, self.net.e.weight, self.ue, m_layers)
            self._ia_i, self._ia_e = _propagate(self.net.i.weight, self.net.e.weight, self.ie, m_layers)

        return {
            "history": history,
            "eligible_interactions": len(eligible),
            "ui_edges": len(self.ui[0]),
            "ue_edges": len(self.ue[0]),
            "ie_edges": len(self.ie[0]),
            "item_level_explanation_negative_adaptation": True,
        }

    def score(self, user, item, candidates):
        import torch

        dev = self._ua_u.device
        with torch.no_grad():
            e = torch.as_tensor(candidates, dtype=torch.long, device=dev)
            return (
                (self._ua_u[user][None, :] * self._ua_e[e]).sum(-1)
                + (self._ia_i[item][None, :] * self._ia_e[e]).sum(-1)
            ).cpu().numpy()
