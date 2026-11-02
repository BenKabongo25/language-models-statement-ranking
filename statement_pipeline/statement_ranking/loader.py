from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from statement_ranking.utils import decode_ids

StatementType = Literal["user", "item", "match"]


@dataclass(frozen=True)
class RankingInteraction:
    interaction_id: int
    line_index: int
    user_id: str
    item_id: str
    rating: float | None
    timestamp: int
    split: str
    user_sequence_index: int
    user_sequence_length: int
    item_sequence_index: int
    item_sequence_length: int
    global_sequence_index: int
    statement_ids: list[int]
    aligned_statement_ids: list[int]
    original_statement_ids: list[int]


@dataclass(frozen=True)
class OriginalStatement:
    original_statement_id: int
    type: str
    canonical_statement_id: int
    interaction_id: int
    line_index: int
    statement_index: int
    user_id: str
    item_id: str
    source_span: str
    text: str
    text_source: str
    stance: str | None


class StatementRankingDataset:
    """Reader for temporal statement-ranking datasets exported by this module."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.final = self.root / "final"
        manifest_path = self.final / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def interactions(self, typ: StatementType = "match", split: str | None = None) -> Iterator[RankingInteraction]:
        path = self.final / f"interactions_{typ}.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if split is not None and row["split"] != split:
                    continue
                yield RankingInteraction(
                    interaction_id=int(row["interaction_id"]),
                    line_index=int(row["line_index"]),
                    user_id=row["user_id"], item_id=row["item_id"],
                    rating=None if row["rating"] == "" else float(row["rating"]),
                    timestamp=int(row["timestamp"]), split=row["split"],
                    user_sequence_index=int(row["user_sequence_index"]),
                    user_sequence_length=int(row["user_sequence_length"]),
                    item_sequence_index=int(row["item_sequence_index"]),
                    item_sequence_length=int(row["item_sequence_length"]),
                    global_sequence_index=int(row["global_sequence_index"]),
                    statement_ids=decode_ids(row["statement_ids"]),
                    aligned_statement_ids=decode_ids(row["aligned_statement_ids"]),
                    original_statement_ids=decode_ids(row["original_statement_ids"]),
                )

    def id2statement(self, typ: StatementType) -> dict[int, str]:
        value = json.loads((self.final / f"id2statement_{typ}.json").read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in value.items()}

    def id2original(self, typ: StatementType) -> dict[int, str]:
        value = json.loads((self.final / f"id2original_{typ}.json").read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in value.items()}

    def original_statements(self, typ: StatementType) -> Iterator[OriginalStatement]:
        path = self.final / f"original_statements_{typ}.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                yield OriginalStatement(
                    original_statement_id=int(row["original_statement_id"]),
                    type=row["type"], canonical_statement_id=int(row["canonical_statement_id"]),
                    interaction_id=int(row["interaction_id"]), line_index=int(row["line_index"]),
                    statement_index=int(row["statement_index"]), user_id=row["user_id"],
                    item_id=row["item_id"], source_span=row["source_span"], text=row["text"],
                    text_source=row["text_source"], stance=None if row["stance"] == "" else row["stance"],
                )

    def item_candidates(self, typ: StatementType = "match") -> dict[str, list[int]]:
        path = self.final / f"item_candidates_{typ}.csv"
        out: dict[str, list[int]] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                out[row["item_id"]] = decode_ids(row["statement_ids"])
        return out

    def train_index(self) -> list[int]:
        return self._read_index("train.index")

    def test_index(self) -> list[int]:
        return self._read_index("test.index")

    def _read_index(self, name: str) -> list[int]:
        text = (self.final / name).read_text(encoding="utf-8").strip()
        return [] if not text else [int(v) for v in text.split()]
