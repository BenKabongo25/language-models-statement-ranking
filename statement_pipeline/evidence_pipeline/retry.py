from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from evidence_pipeline.model_runtime import GenerationMode, generate_texts
from evidence_pipeline.utils import parse_json_text


@dataclass
class ValidatedGeneration:
    value: Any | None
    errors: list[str]
    raw_answers: list[str]

    @property
    def ok(self) -> bool:
        return self.value is not None and not self.errors


def batch_generate_and_validate(
    *,
    llm,
    spec,
    sampling_params,
    prompts: list[str],
    validators: list[Callable[[Any], tuple[Any | None, list[str]]]],
    retries: int,
    mode: GenerationMode | None = None,
) -> list[ValidatedGeneration]:
    if len(prompts) != len(validators):
        raise ValueError("prompts and validators must have the same length")

    mode = mode or GenerationMode()
    results = [ValidatedGeneration(value=None, errors=[], raw_answers=[]) for _ in prompts]
    pending = list(range(len(prompts)))
    retry_prompts = list(prompts)

    for attempt in range(retries + 1):
        if not pending:
            break
        generations = generate_texts(
            llm,
            [retry_prompts[index] for index in pending],
            sampling_params,
            spec,
            mode=mode,
        )
        next_pending: list[int] = []
        for index, generation in zip(pending, generations, strict=True):
            result = results[index]
            result.raw_answers.append(generation.raw_text)
            try:
                parsed = parse_json_text(generation.answer)
            except Exception as exc:
                result.errors = [f"JSON parsing failed: {exc}"]
            else:
                value, errors = validators[index](parsed)
                result.value = value
                result.errors = errors
                if not errors and value is not None:
                    continue
            if attempt < retries:
                next_pending.append(index)
                retry_prompts[index] = (
                    prompts[index]
                    + "\n\nYour previous answer was invalid. Correct only the output according to the task. "
                    + "Validation errors: "
                    + "; ".join(result.errors)
                )
        pending = next_pending
    return results
