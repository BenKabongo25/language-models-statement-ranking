from __future__ import annotations

from .models.classical import RandomRanker, UserPop, ItemPop, GlobalPop, RUCF, RICF
from .models.factorization import CD, PITF
from .models.bper import BPER, BPERLarge, BPERPlus
from .models.semantic_bper import SEBPER, SEBPERPlus
from .models.expgcn import ExpGCN
from .models.cler import CLER

REGISTRY = {
    "rand": RandomRanker,
    "rucf": RUCF,
    "ricf": RICF,
    "cd": CD,
    "pitf": PITF,
    "bper": BPER,
    "bper_plus": BPERPlus,
    "bper_large": BPERLarge,
    "se_bper": SEBPER,
    "se_bper_plus": SEBPERPlus,
    "expgcn": ExpGCN,
    "userpop": UserPop,
    "itempop": ItemPop,
    "globalpop": GlobalPop,
    "cler": CLER,
}

DETERMINISTIC = {"rucf", "ricf", "userpop", "itempop", "globalpop"}
STOCHASTIC = set(REGISTRY) - DETERMINISTIC
TRANSFORMER = {"bper_plus", "se_bper", "se_bper_plus"}
