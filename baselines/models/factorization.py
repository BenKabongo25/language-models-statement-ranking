from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from baselines.data import IndexedInteraction, RankingData
from baselines.models.base import TorchRanker
from baselines.sampling import ItemNegativeSampler


def _device(device: str):
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _init_uniform(module, r=0.005):
    import torch.nn as nn
    for m in module.modules():
        if isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -r, r)


class CDNet:
    pass


class CD(TorchRanker):
    """Canonical decomposition score sum_f U_uf I_if E_ef; item-conditioned BPR."""
    def fit(self, interactions: list[IndexedInteraction], *, validation=None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        mcfg, tcfg = self.cfg["model"], self.cfg["training"]
        dim = int(mcfg.get("dimension", 20)); dev = _device(self.device)
        class Net(nn.Module):
            def __init__(self, nu, ni, ne):
                super().__init__(); self.u=nn.Embedding(nu,dim); self.i=nn.Embedding(ni,dim); self.e=nn.Embedding(ne,dim); _init_uniform(self)
            def score(self,u,i,e): return (self.u(u)*self.i(i)*self.e(e)).sum(-1)
        self.net=Net(self.data.n_users,self.data.n_items,self.data.n_statements).to(dev)
        opt_name=str(tcfg.get("optimizer","sgd")).lower()
        opt_cls=torch.optim.SGD if opt_name=="sgd" else torch.optim.Adam
        opt=opt_cls(self.net.parameters(),lr=float(tcfg.get("lr",1e-3)),weight_decay=0.0)
        reg=float(tcfg.get("weight_decay", tcfg.get("reg_rate", 1e-4)))
        sampler=ItemNegativeSampler(self.data,interactions,self.cfg["sampling"].get("candidate_source","fit_only"),self.seed)
        rng=random.Random(self.seed); bs=int(tcfg.get("batch_size",1024)); epochs=int(tcfg.get("epochs",50)); history=[]
        eligible=[x for x in interactions if sampler.pool(x)]
        for ep in range(1,epochs+1):
            rng.shuffle(eligible); total=0.; n=0
            for st in range(0,len(eligible),bs):
                rows=eligible[st:st+bs]; us=[]; it=[]; ps=[]; ns=[]
                for x in rows:
                    p=rng.choice(x.statements); neg=sampler.sample(x,1)
                    if neg: us.append(x.user);it.append(x.item);ps.append(p);ns.append(neg[0])
                if not us: continue
                u=torch.tensor(us,device=dev);i=torch.tensor(it,device=dev);p=torch.tensor(ps,device=dev);q=torch.tensor(ns,device=dev)
                loss=-F.logsigmoid(self.net.score(u,i,p)-self.net.score(u,i,q)).mean()
                if reg > 0:
                    reg_loss=(self.net.u(u).pow(2).sum(-1)+self.net.i(i).pow(2).sum(-1)+self.net.e(p).pow(2).sum(-1)+self.net.e(q).pow(2).sum(-1)).mean()
                    loss=loss+0.5*reg*reg_loss
                opt.zero_grad();loss.backward();opt.step(); total+=float(loss.detach())*len(us);n+=len(us)
            history.append({"epoch":ep,"loss":total/max(n,1)})
        return {"history":history,"eligible_interactions":len(eligible)}
    def score(self,user,item,candidates):
        import torch
        dev=next(self.net.parameters()).device
        with torch.no_grad():
            e=torch.as_tensor(candidates,dtype=torch.long,device=dev);u=torch.full_like(e,user);i=torch.full_like(e,item)
            return self.net.score(u,i,e).detach().cpu().numpy()


class PITF(TorchRanker):
    """PITF explanation ranker: U-E + I-E score with pairwise BPR.

    This follows the explanation-ranking baseline in the BPER paper, where PITF uses
    the same pairwise explanation objective as CD. Item-level adaptation changes only
    the explanation-negative pool to S_i \\ P_ui.
    """
    def fit(self, interactions: list[IndexedInteraction], *, validation=None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        mcfg,tcfg=self.cfg["model"],self.cfg["training"]; dim=int(mcfg.get("dimension",20)); dev=_device(self.device)
        class Net(nn.Module):
            def __init__(self,nu,ni,ne):
                super().__init__(); self.u=nn.Embedding(nu,dim);self.i=nn.Embedding(ni,dim);self.eu=nn.Embedding(ne,dim);self.ei=nn.Embedding(ne,dim); _init_uniform(self)
            def exp_score(self,u,i,e): return (self.u(u)*self.eu(e)).sum(-1)+(self.i(i)*self.ei(e)).sum(-1)
        self.net=Net(self.data.n_users,self.data.n_items,self.data.n_statements).to(dev)
        opt_name=str(tcfg.get("optimizer","sgd")).lower()
        opt_cls=torch.optim.SGD if opt_name=="sgd" else torch.optim.Adam
        opt=opt_cls(self.net.parameters(),lr=float(tcfg.get("lr",1e-3)),weight_decay=0.0)
        reg=float(tcfg.get("weight_decay", tcfg.get("reg_rate", 1e-4)))
        sampler=ItemNegativeSampler(self.data,interactions,self.cfg["sampling"].get("candidate_source","fit_only"),self.seed)
        rng=random.Random(self.seed); bs=int(tcfg.get("batch_size",1024));epochs=int(tcfg.get("epochs",50));nneg=int(tcfg.get("num_negatives",1));history=[]
        eligible=[x for x in interactions if sampler.pool(x)]
        for ep in range(1,epochs+1):
            rng.shuffle(eligible); total=0.;n=0
            for st in range(0,len(eligible),bs):
                U=[];I=[];P=[];N=[]
                for x in eligible[st:st+bs]:
                    for _ in range(nneg):
                        neg=sampler.sample(x,1)
                        if not neg: continue
                        U.append(x.user);I.append(x.item);P.append(rng.choice(x.statements));N.append(neg[0])
                if not U: continue
                u=torch.tensor(U,device=dev);i=torch.tensor(I,device=dev);p=torch.tensor(P,device=dev);nexp=torch.tensor(N,device=dev)
                loss=-F.logsigmoid(self.net.exp_score(u,i,p)-self.net.exp_score(u,i,nexp)).mean()
                if reg > 0:
                    reg_loss=(
                        self.net.u(u).pow(2).sum(-1)+self.net.i(i).pow(2).sum(-1)
                        +self.net.eu(p).pow(2).sum(-1)+self.net.eu(nexp).pow(2).sum(-1)
                        +self.net.ei(p).pow(2).sum(-1)+self.net.ei(nexp).pow(2).sum(-1)
                    ).mean()
                    loss=loss+0.5*reg*reg_loss
                opt.zero_grad();loss.backward();opt.step(); total+=float(loss.detach())*len(U);n+=len(U)
            history.append({"epoch":ep,"loss":total/max(n,1)})
        return {"history":history,"eligible_interactions":len(eligible)}
    def score(self,user,item,candidates):
        import torch
        dev=next(self.net.parameters()).device
        with torch.no_grad():
            e=torch.as_tensor(candidates,dtype=torch.long,device=dev);u=torch.full_like(e,user);i=torch.full_like(e,item)
            return self.net.exp_score(u,i,e).cpu().numpy()
