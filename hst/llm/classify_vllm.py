"""Label texts with a prompt + JSON-schema classifier (an open LLM served by vLLM).

Inputs   ``<work>/llm/inputs/shard_XX.jsonl`` (from ``hst.llm.prepare_inputs``), the prompt
         template ``cfg.llm.prompt`` and the JSON schema ``cfg.llm.schema``; the configured
         ``cfg.llm.categories`` must match the category objects in the schema.
Outputs  ``<work>/llm/predictions/shard_XX.jsonl`` with one line per text::

             {"text_hash", "raw_response", "parsed" (validated dict or null), "parse_ok",
              "error" (list of validation errors), "generated_at", "backend", "model"}

         and ``shard_XX_summary.json`` (rows, parse_ok, positives per category, verdict counts,
         elapsed).  A shard resumes: texts already predicted with parse_ok are skipped; failed
         ones are retried; ``--force`` re-labels everything.

    python -m hst.llm.classify_vllm --config project.yaml [--shard 0]... [--backend vllm|fake]
                                    [--batch_size 64] [--max_tokens 400] [--force]

Prompt template: a text file with ``<<<SYSTEM>>>`` and ``<<<USER>>>`` sections and the
``{{POST_TEXT_JSON}}`` placeholder (the post text is inserted as a JSON string so that it is
quoted data, not an instruction).  A plain file without those markers is treated as a single
user prompt with a ``{text}`` placeholder.

Backends
  vllm  the real model (GPU).  Deterministic: temperature 0, fixed seed, structured JSON decoding
        constrained by the schema.  vLLM is imported lazily; install it with ``pip install vllm``
        on a GPU machine.
  fake  no model.  A transparent keyword rule (the same one behind the toy dataset's gold labels)
        that returns schema-valid JSON, so the whole pipeline can run on a laptop and in tests.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ..config import Config, add_config_argument, config_from_args
from ._common import (check_categories, frame_label_contradiction, input_shards, load_schema, predictions_dir,
                      read_jsonl, sha256_file, validate_response, verdict_values, write_json)

SYSTEM_MARK, USER_MARK, TEXT_MARK = "<<<SYSTEM>>>", "<<<USER>>>", "{{POST_TEXT_JSON}}"


# ---- prompt ---------------------------------------------------------------------------------------
def parse_prompt_template(path: Path) -> tuple[str, str]:
    """Returns (system, user_template).  The user template contains {{POST_TEXT_JSON}}."""
    text = path.read_text(encoding="utf-8")
    if SYSTEM_MARK in text and USER_MARK in text:
        _, remainder = text.split(SYSTEM_MARK, 1)
        system, user = remainder.split(USER_MARK, 1)
        if TEXT_MARK not in user:
            raise ValueError(f"the <<<USER>>> section of {path} must contain {TEXT_MARK}")
        return system.strip(), user.strip()
    if "{text}" not in text:
        raise ValueError(f"{path} must contain {SYSTEM_MARK}/{USER_MARK} sections or a {{text}} placeholder")
    return "", text.replace("{text}", TEXT_MARK).strip()


def fill_user_prompt(user_template: str, text: str) -> str:
    return user_template.replace(TEXT_MARK, json.dumps(str(text), ensure_ascii=False))


# ---- backends -------------------------------------------------------------------------------------
class FakeBackend:
    """Deterministic stand-in for the model: labels by fixed phrases, no GPU needed.

    Useful for checking the plumbing (shards, resume, merge, attach) on a machine without vLLM.
    The verdict is positive only when a group name appears together with a category phrase, so the
    two-step gate is exercised as well.
    """

    GROUPS = ("zorbs", "blerns")
    PHRASES = {
        "legitimation_political_violence": ["by force", "fight them"],
        "revolutionary_accelerationist": ["burn it all down", "until the whole system collapses"],
        "eliminationist_logic": ["get rid of all", "wipe out every"],
        "totalising_apocalyptic_conspiracy": ["secret council controls everything", "they control everything"],
    }
    name = "fake"

    def __init__(self, schema: dict[str, Any], categories: Sequence[str]):
        self.schema = schema
        self.categories = list(categories)
        self.positive_verdict, self.negative_verdict = verdict_values(schema)
        self.phrases = {c: list(self.PHRASES.get(c, [])) for c in self.categories}

    def generate(self, prompts: Sequence[str], texts: Sequence[str]) -> list[str]:
        out = []
        for text in texts:
            low = text.lower()
            decisions: dict[str, Any] = {}
            any_positive = False
            for cat in self.categories:
                hit = next((p for p in self.phrases.get(cat, []) if p in low), None)
                if hit:
                    any_positive = True
                    start = low.index(hit)
                    decisions[cat] = {"label": True, "confidence": 0.9, "evidence": text[start:start + len(hit)]}
                else:
                    decisions[cat] = {"label": False, "confidence": 0.95, "evidence": ""}
            group_present = any(g in low for g in self.GROUPS)
            response: dict[str, Any] = {}
            if self.positive_verdict is not None:
                supremacist = group_present and any_positive
                response["frame_verdict"] = self.positive_verdict if supremacist else self.negative_verdict
                response["frame_reason"] = "group named with an advocated outcome" if supremacist else "no supremacist outcome"
                if not supremacist:
                    for cat in self.categories:
                        decisions[cat] = {"label": False, "confidence": 0.95, "evidence": ""}
            response.update(decisions)
            response["rationale"] = "Keyword rule of the fake backend."
            out.append(json.dumps(response, ensure_ascii=False))
        return out


class VLLMBackend:
    """The real model through vLLM with schema-constrained JSON decoding."""

    name = "vllm"

    def __init__(self, cfg: Config, schema: dict[str, Any], max_tokens: int):
        try:
            from vllm import LLM  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed. Install it on a GPU machine with `pip install vllm`, "
                "or run with --backend fake to exercise the pipeline without a model."
            ) from exc
        from transformers import AutoTokenizer
        from vllm import LLM

        llm_cfg = cfg.section("llm")
        self.model = str(llm_cfg.get("model", "Qwen/Qwen2.5-14B-Instruct"))
        self.seed = int(llm_cfg.get("seed", 20260810))
        self.max_tokens = int(max_tokens)
        self.schema = _schema_for_decoding(schema)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=False)
        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=int(llm_cfg.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(llm_cfg.get("gpu_memory_utilization", 0.90)),
            max_model_len=int(llm_cfg.get("max_model_len", 4096)),
            max_num_seqs=int(llm_cfg.get("max_num_seqs", 16)),
            dtype=str(llm_cfg.get("dtype", "bfloat16")),
            trust_remote_code=False,
            enable_prefix_caching=True,
            seed=self.seed,
        )

    def _sampling(self) -> tuple[Any, dict[str, Any]]:
        """Fresh SamplingParams per batch (older vLLM versions mutate them), across API versions."""
        from vllm import SamplingParams

        common = {"temperature": 0.0, "top_p": 1.0, "max_tokens": self.max_tokens, "seed": self.seed}
        try:
            from vllm.sampling_params import StructuredOutputsParams

            return SamplingParams(structured_outputs=StructuredOutputsParams(json=self.schema), **common), {}
        except (ImportError, TypeError):
            pass
        try:
            from vllm.sampling_params import GuidedDecodingParams

            return SamplingParams(guided_decoding=GuidedDecodingParams(json=self.schema), **common), {}
        except (ImportError, TypeError):
            return SamplingParams(**common), {"guided_options_request": {"guided_json": self.schema}}

    def generate(self, prompts: Sequence[str], texts: Sequence[str]) -> list[str]:
        chat = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}] if system else
                [{"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)
            for system, user in prompts_split(prompts)
        ]
        params, extra = self._sampling()
        generated = self.llm.generate(chat, sampling_params=params, use_tqdm=False, **extra)
        if len(generated) != len(prompts):
            raise RuntimeError(f"vLLM returned {len(generated)} outputs for {len(prompts)} prompts")
        return [g.outputs[0].text.strip() for g in generated]


def _schema_for_decoding(value: Any) -> Any:
    """Drop boolean additionalProperties (unsupported by some guided-decoding backends);
    the validator still rejects unexpected fields."""
    if isinstance(value, dict):
        return {k: _schema_for_decoding(v) for k, v in value.items() if not (k == "additionalProperties" and isinstance(v, bool))}
    if isinstance(value, list):
        return [_schema_for_decoding(v) for v in value]
    return value


PROMPT_SEP = "\n\n<<<END_SYSTEM>>>\n\n"


def make_prompt(system: str, user: str) -> str:
    """One string per text carrying both sections; backends split it again."""
    return f"{system}{PROMPT_SEP}{user}" if system else user


def prompts_split(prompts: Sequence[str]) -> list[tuple[str, str]]:
    out = []
    for p in prompts:
        if PROMPT_SEP in p:
            system, user = p.split(PROMPT_SEP, 1)
        else:
            system, user = "", p
        out.append((system, user))
    return out


def make_backend(name: str, cfg: Config, schema: dict[str, Any], categories: Sequence[str], max_tokens: int):
    if name == "fake":
        return FakeBackend(schema, categories)
    if name == "vllm":
        return VLLMBackend(cfg, schema, max_tokens)
    raise ValueError(f"unknown backend {name!r}; choose vllm or fake")


# ---- the run ----------------------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def completed_hashes(path: Path) -> dict[str, bool]:
    """text_hash -> parse_ok for every prediction already on disk (last one wins)."""
    done: dict[str, bool] = {}
    for row in read_jsonl(path):
        done[str(row["text_hash"]).lower()] = bool(row.get("parse_ok"))
    return done


def classify_shard(cfg: Config, shard_path: Path, backend, schema: dict[str, Any], categories: list[str],
                   batch_size: int, force: bool, system: str, user_template: str) -> dict[str, Any]:
    out_dir = predictions_dir(cfg)
    out_path = out_dir / shard_path.name
    summary_path = out_dir / f"{shard_path.stem}_summary.json"
    rows = read_jsonl(shard_path)
    if force and out_path.exists():
        out_path.unlink()
    done = completed_hashes(out_path)
    pending = [r for r in rows if not done.get(str(r["text_hash"]).lower(), False)]
    counts: Counter[str] = Counter()
    verdicts: Counter[str] = Counter()
    parse_ok_n = contradictions = 0
    started = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8", newline="\n") as fh:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            texts = [str(r["text"]) for r in batch]
            prompts = [make_prompt(system, fill_user_prompt(user_template, t)) for t in texts]
            raws = backend.generate(prompts, texts)
            for row, text, raw in zip(batch, texts, raws):
                parsed, errors = validate_response(raw, text, schema)
                ok = not errors
                record = {
                    "text_hash": str(row["text_hash"]).lower(), "raw_response": raw, "parsed": parsed if ok else None,
                    "parse_ok": ok, "error": errors, "frame_label_contradiction": frame_label_contradiction(parsed, schema),
                    "generated_at": utc_now(), "backend": backend.name, "model": getattr(backend, "model", backend.name),
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                parse_ok_n += int(ok)
                if ok:
                    for cat in categories:
                        counts[cat] += int(parsed[cat]["label"] is True)
                    if "frame_verdict" in parsed:
                        verdicts[str(parsed["frame_verdict"])] += 1
                    contradictions += int(bool(record["frame_label_contradiction"]))
            fh.flush()
    all_done = completed_hashes(out_path)
    summary = {
        "shard": shard_path.name, "input_rows": len(rows), "already_done": len(rows) - len(pending),
        "new_rows": len(pending), "new_parse_ok": parse_ok_n, "new_parse_failed": len(pending) - parse_ok_n,
        "total_parse_ok": int(sum(all_done.values())), "positives_new": dict(counts),
        "frame_verdicts_new": dict(verdicts), "frame_label_contradictions_new": contradictions,
        "backend": backend.name, "model": getattr(backend, "model", backend.name),
        "elapsed_seconds": round(time.monotonic() - started, 3), "finished_at": utc_now(),
    }
    write_json(summary_path, summary)
    print(f"{shard_path.name}: {len(pending)} new ({parse_ok_n} valid), {len(rows) - len(pending)} already done, "
          f"{summary['elapsed_seconds']}s")
    return summary


def run(cfg: Config, shards: list[int] | None = None, backend_name: str = "vllm", batch_size: int | None = None,
        max_tokens: int | None = None, force: bool = False) -> dict[str, Any]:
    llm_cfg = cfg.section("llm")
    schema = load_schema(cfg)
    categories = check_categories(cfg, schema)
    prompt_path = cfg.llm_prompt_path()
    system, user_template = parse_prompt_template(prompt_path)
    batch_size = int(batch_size or llm_cfg.get("batch_size", 64))
    max_tokens = int(max_tokens or llm_cfg.get("max_tokens", 400))
    paths = input_shards(cfg)
    if shards:
        wanted = {f"shard_{i:02d}.jsonl" for i in shards}
        paths = [p for p in paths if p.name in wanted]
    if not paths:
        raise FileNotFoundError("no input shards found; run `python -m hst.llm.prepare_inputs` first")
    backend = make_backend(backend_name, cfg, schema, categories, max_tokens)
    summaries = [classify_shard(cfg, p, backend, schema, categories, batch_size, force, system, user_template) for p in paths]
    overall = {
        "backend": backend_name, "prompt": str(prompt_path), "prompt_sha256": sha256_file(prompt_path),
        "schema_sha256": sha256_file(cfg.llm_schema_path()), "shards": summaries,
        "new_rows": sum(s["new_rows"] for s in summaries), "new_parse_ok": sum(s["new_parse_ok"] for s in summaries),
    }
    write_json(predictions_dir(cfg) / "run_summary.json", overall)
    return overall


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument("--shard", type=int, action="append", help="Shard index to process (repeatable; default all).")
    parser.add_argument("--backend", choices=["vllm", "fake"], default="vllm")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Re-label texts that already have predictions.")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    summary = run(cfg, args.shard, args.backend, args.batch_size, args.max_tokens, args.force)
    print(f"done: {summary['new_rows']} new predictions, {summary['new_parse_ok']} valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
