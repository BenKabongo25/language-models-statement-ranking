from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evidence_pipeline.utils import project_root, sha256_text


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    text: str
    sha256: str


def _load(relative: str) -> str:
    path = project_root() / "evidence_pipeline" / "prompts" / relative
    return path.read_text(encoding="utf-8").strip()


def load_stage1_prompt(has_review_title: bool) -> PromptTemplate:
    common = _load("stage1_common.txt")
    suffix = _load("stage1_with_title.txt" if has_review_title else "stage1_without_title.txt")
    name = "stage1_with_review_title" if has_review_title else "stage1_without_review_title"
    text = f"{common}\n\n{suffix}\n"
    return PromptTemplate(name=name, text=text, sha256=sha256_text(text))


def load_stage2_prompt() -> PromptTemplate:
    text = _load("stage2_projection.txt") + "\n"
    return PromptTemplate(name="stage2_projection", text=text, sha256=sha256_text(text))


def load_stage3_prompt() -> PromptTemplate:
    text = _load("stage3_stance.txt") + "\n"
    return PromptTemplate(name="stage3_stance", text=text, sha256=sha256_text(text))


def render(template: PromptTemplate, values: dict[str, object]) -> str:
    text = template.text
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", "" if value is None else str(value))
    return text
