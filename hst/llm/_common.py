"""Shared helpers for the LLM stage: folders, JSONL I/O, schema reading and response validation.

The response contract comes from the JSON schema (``cfg.llm.schema``).  A "category" is any
top-level property whose shape is the per-category decision object ``{label, confidence,
evidence}``; ``frame_verdict`` is the optional two-step gate emitted first (its allowed values
are read from the schema's enum, and the value that does not start with ``not`` counts as the
positive verdict).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ..config import Config

CATEGORY_FIELDS = {"label", "confidence", "evidence"}
MAX_EVIDENCE_WORDS = 20


# ---- folders -------------------------------------------------------------------------------
def llm_dir(cfg: Config) -> Path:
    return cfg.path("work", "work") / "llm"


def inputs_dir(cfg: Config) -> Path:
    return llm_dir(cfg) / "inputs"


def predictions_dir(cfg: Config) -> Path:
    return llm_dir(cfg) / "predictions"


def labels_path(cfg: Config) -> Path:
    return llm_dir(cfg) / "llm_labels.csv"


def shard_name(index: int) -> str:
    return f"shard_{index:02d}"


def input_shards(cfg: Config) -> list[Path]:
    return sorted(inputs_dir(cfg).glob("shard_*.jsonl"))


# ---- jsonl ----------------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- schema -------------------------------------------------------------------------------------
def load_schema(cfg: Config) -> dict[str, Any]:
    path = cfg.llm_schema_path()
    return json.loads(path.read_text(encoding="utf-8"))


def _is_category_shape(spec: dict[str, Any], schema: dict[str, Any]) -> bool:
    if "$ref" in spec:
        ref = spec["$ref"].split("/")[-1]
        spec = (schema.get("$defs") or schema.get("definitions") or {}).get(ref, {})
    props = spec.get("properties") or {}
    return spec.get("type") == "object" and CATEGORY_FIELDS <= set(props)


def schema_categories(schema: dict[str, Any]) -> list[str]:
    return [k for k, v in (schema.get("properties") or {}).items() if isinstance(v, dict) and _is_category_shape(v, schema)]


def verdict_values(schema: dict[str, Any]) -> tuple[str | None, str | None]:
    """(positive_verdict, negative_verdict) from the frame_verdict enum, or (None, None)."""
    spec = (schema.get("properties") or {}).get("frame_verdict")
    if not isinstance(spec, dict) or not spec.get("enum"):
        return None, None
    values = [str(v) for v in spec["enum"]]
    negative = next((v for v in values if v.lower().startswith("not")), None)
    positive = next((v for v in values if v != negative), None)
    return positive, negative


def check_categories(cfg: Config, schema: dict[str, Any]) -> list[str]:
    """The configured LLM categories must match the category objects of the schema."""
    configured = cfg.llm_categories
    in_schema = schema_categories(schema)
    if sorted(configured) != sorted(in_schema):
        raise ValueError(
            f"llm.categories in the config {configured} do not match the schema's category objects {in_schema}"
        )
    return in_schema


# ---- response validation --------------------------------------------------------------------------
def normalized_for_evidence(value: str) -> str:
    return " ".join(value.casefold().split())


def validate_response(raw: str, source_text: str, schema: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse a raw model response and check it against the schema and the evidence rules.

    Returns (parsed_or_None, errors).  A response is valid only when errors is empty.
    Rules (as in the original classifier): every required field present, no unexpected fields,
    frame_verdict in its enum, string fields respect maxLength, each category is an object with
    exactly label (bool) / confidence (number in [0, 1]) / evidence (str); a positive label needs a
    non-empty evidence excerpt of at most 20 words copied verbatim from the post; a negative label
    must have evidence == "".
    """
    errors: list[str] = []
    try:
        parsed = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return None, [f"json_decode_error: {exc}"]
    if not isinstance(parsed, dict):
        return None, ["response_is_not_an_object"]

    props = schema.get("properties") or {}
    required = set(schema.get("required") or props.keys())
    categories = schema_categories(schema)
    missing = required - set(parsed)
    extra = set(parsed) - set(props)
    if missing:
        errors.append("missing_fields:" + ",".join(sorted(missing)))
    if extra and schema.get("additionalProperties") is False:
        errors.append("unexpected_fields:" + ",".join(sorted(extra)))

    positive, negative = verdict_values(schema)
    if "frame_verdict" in parsed and positive is not None:
        if parsed["frame_verdict"] not in (positive, negative):
            errors.append("frame_verdict:invalid_value")
    for name, spec in props.items():
        if name in categories or name not in parsed or not isinstance(spec, dict):
            continue
        if spec.get("type") == "string":
            value = parsed[name]
            if not isinstance(value, str):
                errors.append(f"{name}:not_string")
            elif spec.get("maxLength") and len(value) > int(spec["maxLength"]) * 2:
                # the schema-guided decoder enforces maxLength; allow slack for an unguided model
                errors.append(f"{name}:too_long")

    source_norm = normalized_for_evidence(source_text)
    for cat in categories:
        value = parsed.get(cat)
        if not isinstance(value, dict):
            errors.append(f"{cat}:not_object")
            continue
        if set(value) != CATEGORY_FIELDS:
            errors.append(f"{cat}:invalid_fields")
        label = value.get("label")
        confidence = value.get("confidence")
        evidence = value.get("evidence")
        if not isinstance(label, bool):
            errors.append(f"{cat}:label_not_boolean")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            errors.append(f"{cat}:confidence_not_number")
        elif not 0.0 <= float(confidence) <= 1.0:
            errors.append(f"{cat}:confidence_out_of_range")
        if not isinstance(evidence, str):
            errors.append(f"{cat}:evidence_not_string")
        elif label is True:
            if not evidence.strip():
                errors.append(f"{cat}:positive_without_evidence")
            elif len(evidence.split()) > MAX_EVIDENCE_WORDS:
                errors.append(f"{cat}:evidence_over_{MAX_EVIDENCE_WORDS}_words")
            elif normalized_for_evidence(evidence) not in source_norm:
                errors.append(f"{cat}:evidence_not_exact_excerpt")
        elif evidence:
            errors.append(f"{cat}:negative_with_evidence")
    return parsed, errors


def frame_label_contradiction(parsed: dict[str, Any] | None, schema: dict[str, Any]) -> bool | None:
    """True when the verdict is negative but some category label is true."""
    positive, negative = verdict_values(schema)
    if not isinstance(parsed, dict) or positive is None or parsed.get("frame_verdict") not in (positive, negative):
        return None
    any_true = any(isinstance(parsed.get(c), dict) and parsed[c].get("label") is True for c in schema_categories(schema))
    return parsed["frame_verdict"] == negative and any_true
