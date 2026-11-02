from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from baselines.data import IndexedInteraction
from baselines.models.base import TorchRanker
from baselines.sampling import ItemNegativeSampler


def _dev(name):
    import torch
    return torch.device("cuda" if name == "cuda" and torch.cuda.is_available() else "cpu")


def _fixed_item_level_examples(interactions, sampler, nneg: int, seed: int):
    """Flatten every positive statement and sample fixed same-item negatives once.

    The released SE-BPER preprocessing materializes ``num_negs`` rows per positive
    rationale before training, then DataLoader only shuffles these rows.
    """
    rng = random.Random(seed)
    examples = []
    for x in interactions:
        pool = sampler.pool(x)
        if not pool:
            continue
        for positive in x.statements:
            for _ in range(int(nneg)):
                negative = rng.choice(pool)
                examples.append((x.user, x.item, int(positive), int(negative)))
    return examples


def _mean_profiles(emb: np.ndarray, interactions: list[IndexedInteraction], n_users: int, n_items: int):
    """Equations 11/12 in SE-BPER: mean semantic rationale vectors per entity."""
    dim = emb.shape[1]
    users = np.zeros((n_users, dim), dtype=np.float32)
    items = np.zeros((n_items, dim), dtype=np.float32)
    seen_u: dict[int, set[int]] = defaultdict(set)
    seen_i: dict[int, set[int]] = defaultdict(set)
    for x in interactions:
        seen_u[x.user].update(x.statements)
        seen_i[x.item].update(x.statements)
    for u, sids in seen_u.items():
        if sids:
            users[u] = emb[list(sids)].mean(0)
    for i, sids in seen_i.items():
        if sids:
            items[i] = emb[list(sids)].mean(0)
    return users, items


class _SemanticMixin:
    def _load_text_model(self):
        from transformers import AutoModel, AutoTokenizer
        mcfg = self.cfg["model"]
        name = str(mcfg.get("transformer", "albert-base-v2"))
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        model = AutoModel.from_pretrained(name)
        return model, tokenizer

    def _tokenize_ids(self, tokenizer, statement_ids, device):
        ids = [int(x) for x in statement_ids]
        texts = [self.data.statement_texts[x] for x in ids]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(self.cfg["model"].get("max_length", 128)),
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}

    def _encode_all(self, model, tokenizer, dev, batch_size=256):
        """Encode the full statement dictionary without materializing an N×L token matrix."""
        import torch
        model.eval()
        out = []
        with torch.no_grad():
            for start in range(0, self.data.n_statements, batch_size):
                ids = range(start, min(start + batch_size, self.data.n_statements))
                enc = self._tokenize_ids(tokenizer, ids, dev)
                h = model(**enc).last_hidden_state[:, 0, :]
                out.append(h.float().cpu())
        return torch.cat(out, dim=0).numpy().astype(np.float32)

    def _fit_initialized_bper(self, interactions, init_e, init_u, init_i):
        """SE-BPER second stage: BPER initialized by semantic representations."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        tcfg, mcfg = self.cfg["training"], self.cfg["model"]
        dim = int(init_e.shape[1])
        dev = _dev(self.device)

        class Net(nn.Module):
            def __init__(self, nu, ni, ne):
                super().__init__()
                self.u = nn.Embedding(nu, dim)
                self.i = nn.Embedding(ni, dim)
                self.eu = nn.Embedding(ne, dim)
                self.ei = nn.Embedding(ne, dim)
                self.bu = nn.Embedding(ne, 1)
                self.bi = nn.Embedding(ne, 1)

            def us(self, u, e):
                return (self.u(u) * self.eu(e)).sum(-1) + self.bu(e).squeeze(-1)

            def its(self, i, e):
                return (self.i(i) * self.ei(e)).sum(-1) + self.bi(e).squeeze(-1)

        self.net = Net(self.data.n_users, self.data.n_items, self.data.n_statements).to(dev)
        semantic_scale = float(mcfg.get("semantic_scale", 0.01))
        bias_range = float(mcfg.get("bias_init_range", 0.01))
        with torch.no_grad():
            # Official SE-BPER divides the semantic initialization by 100.
            self.net.u.weight.copy_(torch.from_numpy(init_u * semantic_scale).to(dev))
            self.net.i.weight.copy_(torch.from_numpy(init_i * semantic_scale).to(dev))
            self.net.eu.weight.copy_(torch.from_numpy(init_e * semantic_scale).to(dev))
            self.net.ei.weight.copy_(torch.from_numpy(init_e * semantic_scale).to(dev))
            self.net.bu.weight.uniform_(-bias_range, bias_range)
            self.net.bi.weight.uniform_(-bias_range, bias_range)

        # The paper specifies Adam + L2; the released engine's Adam branch omits
        # weight_decay.  Both are reproducible through this explicit switch.
        wd = 0.0 if bool(mcfg.get("official_adam_ignores_weight_decay", True)) else float(tcfg.get("weight_decay", 1e-4))
        opt = torch.optim.Adam(
            self.net.parameters(),
            lr=float(tcfg.get("lr", 1e-3)),
            weight_decay=wd,
        )
        sampler = ItemNegativeSampler(
            self.data,
            interactions,
            self.cfg["sampling"].get("candidate_source", "fit_only"),
            self.seed,
        )
        rng = random.Random(self.seed)
        bs = int(tcfg.get("batch_size", 32))
        epochs = int(tcfg.get("epochs", 50))
        nneg = int(tcfg.get("num_negatives", 3))
        hist = []
        examples = _fixed_item_level_examples(interactions, sampler, nneg, self.seed)

        for ep in range(1, epochs + 1):
            rng.shuffle(examples)
            total = 0.0
            n = 0
            for start in range(0, len(examples), bs):
                batch = examples[start : start + bs]
                if not batch:
                    continue
                U, I, P, N = zip(*batch)
                u = torch.tensor(U, device=dev)
                i = torch.tensor(I, device=dev)
                p = torch.tensor(P, device=dev)
                neg = torch.tensor(N, device=dev)
                user_loss = -F.logsigmoid(self.net.us(u, p) - self.net.us(u, neg)).mean()
                item_loss = -F.logsigmoid(self.net.its(i, p) - self.net.its(i, neg)).mean()
                # Released model concatenates both score-difference vectors and means.
                loss = 0.5 * (user_loss + item_loss)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(batch)
                n += len(batch)
            hist.append({"epoch": ep, "loss": total / max(n, 1)})
        self._semantic_train_examples = len(examples)
        return hist

    def score(self, user, item, candidates):
        import torch
        mu = float(self.cfg["model"].get("mu", 0.7))
        dev = next(self.net.parameters()).device
        with torch.no_grad():
            e = torch.as_tensor(candidates, dtype=torch.long, device=dev)
            u = torch.full_like(e, user)
            i = torch.full_like(e, item)
            return (mu * self.net.us(u, e) + (1 - mu) * self.net.its(i, e)).cpu().numpy()


class SEBPER(_SemanticMixin, TorchRanker):
    """SE-BPER: pretrained ALBERT [CLS] initialization followed by BPER optimization."""

    def fit(self, interactions: list[IndexedInteraction], *, validation=None):
        dev = _dev(self.device)
        model, tokenizer = self._load_text_model()
        model = model.to(dev)
        emb = self._encode_all(
            model,
            tokenizer,
            dev,
            int(self.cfg["model"].get("encode_batch_size", 256)),
        )
        u, i = _mean_profiles(emb, interactions, self.data.n_users, self.data.n_items)
        hist = self._fit_initialized_bper(interactions, emb, u, i)
        del model
        return {"history": hist, "semantic_dimension": int(emb.shape[1])}


class SEBPERPlus(_SemanticMixin, TorchRanker):
    """SE-BPER+: task-specific ALBERT ranking pretrain, then the SE-BPER stage."""

    def fit(self, interactions: list[IndexedInteraction], *, validation=None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        dev = _dev(self.device)
        model, tokenizer = self._load_text_model()
        model = model.to(dev)
        hidden = int(model.config.hidden_size)
        head = nn.Linear(hidden * 3, 1).to(dev)
        nn.init.normal_(head.weight, std=0.02)
        nn.init.zeros_(head.bias)

        pcfg = self.cfg.get("pretrain", {})
        opt = torch.optim.Adam(
            [
                {"params": model.parameters(), "lr": float(pcfg.get("encoder_lr", 5e-6))},
                {"params": head.parameters(), "lr": float(pcfg.get("head_lr", 1e-4))},
            ],
            weight_decay=float(pcfg.get("weight_decay", 0.0)),
        )
        sampler = ItemNegativeSampler(
            self.data,
            interactions,
            self.cfg["sampling"].get("candidate_source", "fit_only"),
            self.seed,
        )
        rng = random.Random(self.seed)
        bs = int(pcfg.get("batch_size", 32))
        epochs = int(pcfg.get("epochs", 3))
        nneg = int(pcfg.get("num_negatives", 3))
        encode_bs = int(self.cfg["model"].get("encode_batch_size", 256))
        pre_hist = []

        # Eq. 11/12 profiles are updated after every ALBERT epoch, exactly as described.
        current = self._encode_all(model, tokenizer, dev, encode_bs)
        up, ip = _mean_profiles(current, interactions, self.data.n_users, self.data.n_items)
        examples = _fixed_item_level_examples(interactions, sampler, nneg, self.seed)

        for ep in range(1, epochs + 1):
            model.train()
            head.train()
            rng.shuffle(examples)
            total = 0.0
            n = 0
            for start in range(0, len(examples), bs):
                batch = examples[start : start + bs]
                if not batch:
                    continue
                U, I, P, N = map(list, zip(*batch))

                # Item-level adaptation of Eq. 13: negatives are from the same target item.
                all_e = P + N
                enc = self._tokenize_ids(tokenizer, all_e, dev)
                h = model(**enc).last_hidden_state[:, 0, :]
                b = len(U)
                pu = torch.from_numpy(up[np.asarray(U)]).to(dev)
                qi = torch.from_numpy(ip[np.asarray(I)]).to(dev)
                sp = head(torch.cat([pu, qi, h[:b]], dim=-1)).squeeze(-1)
                sn = head(torch.cat([pu, qi, h[b:]], dim=-1)).squeeze(-1)
                loss = -F.logsigmoid(sp - sn).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * b
                n += b

            current = self._encode_all(model, tokenizer, dev, encode_bs)
            up, ip = _mean_profiles(current, interactions, self.data.n_users, self.data.n_items)
            pre_hist.append({"epoch": ep, "loss": total / max(n, 1)})

        hist = self._fit_initialized_bper(interactions, current, up, ip)
        del model, head
        return {
            "pretrain_history": pre_hist,
            "history": hist,
            "semantic_dimension": hidden,
            "pretrain_examples": len(examples),
            "item_level_pretrain_negative_adaptation": True,
        }
