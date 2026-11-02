from __future__ import annotations

import random
from typing import Any

import numpy as np

from baselines.data import IndexedInteraction, RankingData
from baselines.models.base import TorchRanker
from baselines.sampling import ItemNegativeSampler


def _dev(device):
    import torch
    return torch.device("cuda" if device=="cuda" and torch.cuda.is_available() else "cpu")


def _paper_bper_biases(interactions, n_users: int, n_items: int, n_statements: int):
    """Original BPER explanation-bias initialization.

    The public NumPy implementation initializes each user-side explanation bias as
    |U_e|/|U| - |D|/(|U||E|), and analogously on the item side.  We preserve that
    initialization even though negative sampling is changed to the item-level pool.
    """
    from collections import defaultdict
    ue = defaultdict(set)
    ie = defaultdict(set)
    for x in interactions:
        for e in x.statements:
            ue[int(e)].add(int(x.user))
            ie[int(e)].add(int(x.item))
    bu = np.zeros(n_statements, dtype=np.float32)
    bi = np.zeros(n_statements, dtype=np.float32)
    if n_statements == 0:
        return bu, bi
    gu = len(interactions) / max(1, n_users * n_statements)
    gi = len(interactions) / max(1, n_items * n_statements)
    for e, users in ue.items():
        bu[e] = len(users) / max(1, n_users) - gu
    for e, items in ie.items():
        bi[e] = len(items) / max(1, n_items) - gi
    return bu, bi


class _BPERModule:
    pass


class BPER(TorchRanker):
    """BPER with the original two user/item BPR terms and score interpolation.

    The only benchmark adaptation is that both explanation negatives are drawn from
    S_i \\ P_ui, rather than the original global user/item complement pools.
    """
    def _build(self, dim: int):
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self,nu,ni,ne):
                super().__init__(); self.u=nn.Embedding(nu,dim);self.i=nn.Embedding(ni,dim);self.eu=nn.Embedding(ne,dim);self.ei=nn.Embedding(ne,dim);self.bu=nn.Embedding(ne,1);self.bi=nn.Embedding(ne,1)
                for m in [self.u,self.i,self.eu,self.ei]: nn.init.uniform_(m.weight,-0.005,0.005)
                nn.init.zeros_(self.bu.weight);nn.init.zeros_(self.bi.weight)
            def user_score(self,u,e): return (self.u(u)*self.eu(e)).sum(-1)+self.bu(e).squeeze(-1)
            def item_score(self,i,e): return (self.i(i)*self.ei(e)).sum(-1)+self.bi(e).squeeze(-1)
        return Net(self.data.n_users,self.data.n_items,self.data.n_statements)

    def fit(self, interactions: list[IndexedInteraction], *, validation=None):
        import torch
        import torch.nn.functional as F
        tcfg,mcfg=self.cfg["training"],self.cfg["model"];dim=int(mcfg.get("dimension",20));dev=_dev(self.device)
        self.net=self._build(dim).to(dev)
        if str(mcfg.get("bias_init", "paper_popularity")).lower() == "paper_popularity":
            bu, bi = _paper_bper_biases(
                interactions, self.data.n_users, self.data.n_items, self.data.n_statements
            )
            with torch.no_grad():
                self.net.bu.weight.copy_(torch.from_numpy(bu[:, None]).to(dev))
                self.net.bi.weight.copy_(torch.from_numpy(bi[:, None]).to(dev))
        opt_name=str(tcfg.get("optimizer","sgd")).lower()
        opt_cls=torch.optim.SGD if opt_name=="sgd" else torch.optim.Adam
        # The public NumPy BPER code regularizes only factors/biases touched by the
        # sampled tuple.  Global optimizer weight decay would shrink every embedding
        # row at each vectorized batch and is not equivalent, so regularization is
        # added explicitly to the sampled objective below.
        opt=opt_cls(self.net.parameters(),lr=float(tcfg.get("lr",1e-3)),weight_decay=0.0)
        reg=float(tcfg.get("weight_decay", tcfg.get("reg_rate", 1e-4)))
        sampler=ItemNegativeSampler(self.data,interactions,self.cfg["sampling"].get("candidate_source","fit_only"),self.seed)
        rng=random.Random(self.seed);bs=int(tcfg.get("batch_size",1024));epochs=int(tcfg.get("epochs",50));nneg=int(tcfg.get("num_negatives",1));hist=[]
        eligible=[x for x in interactions if sampler.pool(x)]
        for ep in range(1,epochs+1):
            rng.shuffle(eligible);tot=0.;num=0
            for st in range(0,len(eligible),bs):
                U=[];I=[];P=[];NU=[];NI=[]
                for x in eligible[st:st+bs]:
                    for _ in range(nneg):
                        negs=sampler.sample(x,2)
                        if len(negs)<2: continue
                        U.append(x.user);I.append(x.item);P.append(rng.choice(x.statements));NU.append(negs[0]);NI.append(negs[1])
                if not U: continue
                u=torch.tensor(U,device=dev);i=torch.tensor(I,device=dev);p=torch.tensor(P,device=dev);nu=torch.tensor(NU,device=dev);ni=torch.tensor(NI,device=dev)
                lu=-F.logsigmoid(self.net.user_score(u,p)-self.net.user_score(u,nu)).mean()
                li=-F.logsigmoid(self.net.item_score(i,p)-self.net.item_score(i,ni)).mean()
                if reg > 0:
                    ru=(
                        self.net.u(u).pow(2).sum(-1)
                        +self.net.eu(p).pow(2).sum(-1)+self.net.eu(nu).pow(2).sum(-1)
                        +self.net.bu(p).squeeze(-1).pow(2)+self.net.bu(nu).squeeze(-1).pow(2)
                    ).mean()
                    ri=(
                        self.net.i(i).pow(2).sum(-1)
                        +self.net.ei(p).pow(2).sum(-1)+self.net.ei(ni).pow(2).sum(-1)
                        +self.net.bi(p).squeeze(-1).pow(2)+self.net.bi(ni).squeeze(-1).pow(2)
                    ).mean()
                    loss=lu+li+0.5*reg*(ru+ri)
                else:
                    loss=lu+li
                opt.zero_grad();loss.backward();opt.step();tot+=float(loss.detach())*len(U);num+=len(U)
            hist.append({"epoch":ep,"loss":tot/max(num,1)})
        return {"history":hist,"eligible_interactions":len(eligible)}

    def score(self,user,item,candidates):
        import torch
        mu=float(self.cfg["model"].get("mu",0.7));dev=next(self.net.parameters()).device
        with torch.no_grad():
            e=torch.as_tensor(candidates,dtype=torch.long,device=dev);u=torch.full_like(e,user);i=torch.full_like(e,item)
            return (mu*self.net.user_score(u,e)+(1-mu)*self.net.item_score(i,e)).cpu().numpy()


class BPERLarge(BPER):
    """BPER-large as implemented for the SE-BPER ablation.

    The SE-BPER repository realizes BPER-large through ``random_init`` in the same
    PyTorch training pipeline used for SE-BPER: 768-D factors, U(-.01,.01) embeddings,
    zero explanation biases, Adam, every positive rationale expanded with ``num_negatives``
    negatives, and the same negative used by user/item BPR terms.
    """
    def _build(self, dim: int):
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self, nu, ni, ne):
                super().__init__()
                self.u=nn.Embedding(nu,dim); self.i=nn.Embedding(ni,dim)
                self.eu=nn.Embedding(ne,dim); self.ei=nn.Embedding(ne,dim)
                self.bu=nn.Embedding(ne,1); self.bi=nn.Embedding(ne,1)
                for emb in [self.u,self.i,self.eu,self.ei]: nn.init.uniform_(emb.weight,-0.01,0.01)
                nn.init.zeros_(self.bu.weight); nn.init.zeros_(self.bi.weight)
            def user_score(self,u,e): return (self.u(u)*self.eu(e)).sum(-1)+self.bu(e).squeeze(-1)
            def item_score(self,i,e): return (self.i(i)*self.ei(e)).sum(-1)+self.bi(e).squeeze(-1)
        return Net(self.data.n_users,self.data.n_items,self.data.n_statements)

    def fit(self, interactions: list[IndexedInteraction], *, validation=None):
        import torch
        import torch.nn.functional as F
        tcfg,mcfg=self.cfg["training"],self.cfg["model"]
        dim=int(mcfg.get("dimension",768)); dev=_dev(self.device)
        self.net=self._build(dim).to(dev)
        wd=0.0 if bool(mcfg.get("official_adam_ignores_weight_decay", True)) else float(tcfg.get("weight_decay",1e-4))
        opt=torch.optim.Adam(self.net.parameters(),lr=float(tcfg.get("lr",1e-3)),weight_decay=wd)
        sampler=ItemNegativeSampler(self.data,interactions,self.cfg["sampling"].get("candidate_source","fit_only"),self.seed)
        rng=random.Random(self.seed);bs=int(tcfg.get("batch_size",32));epochs=int(tcfg.get("epochs",50));nneg=int(tcfg.get("num_negatives",3));hist=[]
        examples=[]
        for x in interactions:
            pool=sampler.pool(x)
            if not pool: continue
            for positive in x.statements:
                for _ in range(nneg): examples.append((x.user,x.item,int(positive),rng.choice(pool)))
        for ep in range(1,epochs+1):
            rng.shuffle(examples);tot=0.;num=0
            for st in range(0,len(examples),bs):
                batch=examples[st:st+bs]
                if not batch: continue
                U,I,P,N=zip(*batch)
                u=torch.tensor(U,device=dev);i=torch.tensor(I,device=dev);p=torch.tensor(P,device=dev);n=torch.tensor(N,device=dev)
                lu=-F.logsigmoid(self.net.user_score(u,p)-self.net.user_score(u,n)).mean()
                li=-F.logsigmoid(self.net.item_score(i,p)-self.net.item_score(i,n)).mean()
                loss=0.5*(lu+li)
                opt.zero_grad(set_to_none=True);loss.backward();opt.step();tot+=float(loss.detach())*len(batch);num+=len(batch)
            hist.append({"epoch":ep,"loss":tot/max(num,1)})
        return {"history":hist,"training_examples":len(examples),"se_bper_random_init_ablation":True}


class BPERPlus(BPER):
    """Official BPER+ architecture: BERT CLS projection multiplied element-wise with rationale factors."""
    def fit(self, interactions: list[IndexedInteraction], *, validation=None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
        tcfg,mcfg=self.cfg["training"],self.cfg["model"];dim=int(mcfg.get("dimension",20));dev=_dev(self.device)
        model_name=str(mcfg.get("transformer","bert-base-uncased"));max_len=int(mcfg.get("max_length",10))
        tok=AutoTokenizer.from_pretrained(model_name,use_fast=True)
        enc=tok(self.data.statement_texts,padding="max_length",truncation=True,max_length=max_len,return_tensors="pt")
        class Net(nn.Module):
            def __init__(self,nu,ni,ne):
                super().__init__();self.u=nn.Embedding(nu,dim);self.i=nn.Embedding(ni,dim);self.eu=nn.Embedding(ne,dim);self.ei=nn.Embedding(ne,dim);self.bu=nn.Embedding(ne,1);self.bi=nn.Embedding(ne,1);self.bert=AutoModel.from_pretrained(model_name);self.proj=nn.Linear(self.bert.config.hidden_size,dim)
                for m in [self.u,self.i,self.eu,self.ei]:nn.init.uniform_(m.weight,-.005,.005)
                nn.init.zeros_(self.bu.weight);nn.init.zeros_(self.bi.weight);nn.init.uniform_(self.proj.weight,-.005,.005);nn.init.zeros_(self.proj.bias)
            def text(self,ids,mask): return self.proj(self.bert(input_ids=ids,attention_mask=mask).last_hidden_state[:,0,:])
            def scores(self,u,i,e,te):
                us=(self.u(u)*(self.eu(e)*te)).sum(-1)+self.bu(e).squeeze(-1); it=(self.i(i)*(self.ei(e)*te)).sum(-1)+self.bi(e).squeeze(-1);return us,it
        self.net=Net(self.data.n_users,self.data.n_items,self.data.n_statements).to(dev)
        if not bool(mcfg.get("transformer_trainable",True)):
            for p in self.net.bert.parameters():p.requires_grad=False
        opt=torch.optim.AdamW([p for p in self.net.parameters() if p.requires_grad],lr=float(tcfg.get("lr",1e-4)),weight_decay=float(tcfg.get("weight_decay",0.0)))
        ids=enc["input_ids"];mask=enc["attention_mask"]
        sampler=ItemNegativeSampler(self.data,interactions,self.cfg["sampling"].get("candidate_source","fit_only"),self.seed);rng=random.Random(self.seed);bs=int(tcfg.get("batch_size",128));epochs=int(tcfg.get("epochs",5));hist=[];eligible=[x for x in interactions if sampler.pool(x)]
        for ep in range(1,epochs+1):
            rng.shuffle(eligible);tot=0.;num=0
            for st in range(0,len(eligible),bs):
                U=[];I=[];P=[];NU=[];NI=[]
                for x in eligible[st:st+bs]:
                    ns=sampler.sample(x,2)
                    if len(ns)<2:continue
                    U.append(x.user);I.append(x.item);P.append(rng.choice(x.statements));NU.append(ns[0]);NI.append(ns[1])
                if not U:continue
                all_e=P+NU+NI; tid=ids[all_e].to(dev);tm=mask[all_e].to(dev);te=self.net.text(tid,tm);b=len(U)
                u=torch.tensor(U,device=dev);i=torch.tensor(I,device=dev);p=torch.tensor(P,device=dev);nu=torch.tensor(NU,device=dev);ni=torch.tensor(NI,device=dev)
                up,ip=self.net.scores(u,i,p,te[:b]); un,_=self.net.scores(u,i,nu,te[b:2*b]); _,inn=self.net.scores(u,i,ni,te[2*b:])
                du=up-un; di=ip-inn
                # The public BPER+ code averages the user-side and item-side BPR
                # examples together (equivalent to 0.5 * (L_u + L_i)).  It also
                # accidentally broadcasts the scalar explanation bias over d latent
                # dimensions during training.  The latter can be reproduced for exact
                # legacy fidelity, while paper-consistent scoring is available by switch.
                if bool(mcfg.get("official_legacy_bias_broadcast", True)):
                    # Remove the once-added bias and re-add it d times, matching the
                    # official code's broadcast-before-sum training computation.
                    bu=(self.net.bu(p)-self.net.bu(nu)).squeeze(-1)
                    bi=(self.net.bi(p)-self.net.bi(ni)).squeeze(-1)
                    du=du + (dim-1)*bu
                    di=di + (dim-1)*bi
                loss=0.5*(-F.logsigmoid(du).mean()-F.logsigmoid(di).mean())
                opt.zero_grad();loss.backward();opt.step();tot+=float(loss.detach())*b;num+=b
            hist.append({"epoch":ep,"loss":tot/max(num,1)})
        self._tokenized=(ids,mask);return {"history":hist,"eligible_interactions":len(eligible)}

    def score(self,user,item,candidates):
        import torch
        mu=float(self.cfg["model"].get("mu",.7));dev=next(self.net.parameters()).device;ids,mask=self._tokenized;chunk=int(self.cfg.get("evaluation",{}).get("score_batch_size",512));out=[]
        self.net.eval()
        with torch.no_grad():
            for st in range(0,len(candidates),chunk):
                c=np.asarray(candidates[st:st+chunk],dtype=np.int64);e=torch.tensor(c,device=dev);u=torch.full_like(e,user);i=torch.full_like(e,item);te=self.net.text(ids[c].to(dev),mask[c].to(dev));us,it=self.net.scores(u,i,e,te);out.append((mu*us+(1-mu)*it).cpu().numpy())
        return np.concatenate(out) if out else np.empty(0)
