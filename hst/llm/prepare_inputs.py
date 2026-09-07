"""Collect the unique texts to label with the LLM and write them as JSONL input shards.

Inputs   the record tables (``<work>/records/<platform>_records.csv.gz``): records inside the
         study period with a non-empty text.  The same text is labelled ONCE even when it
         appears on several platforms or in several records (labels are propagated back by
         text_hash at merge time).
Outputs  ``<work>/llm/inputs/shard_00.jsonl`` ...  one ``{"text_hash", "text", "platforms"}``
         per line, ``shard_size`` texts per shard, plus ``inputs_summary.json``.

    python -m hst.llm.prepare_inputs --config project.yaml [--platform x]... [--shard_size 20000]
                                     [--sample 5000] [--seed 0] [--force]

Existing shards are never overwritten unless ``--force`` is given, so a run that has already
started labelling keeps its input files stable.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config, add_config_argument, config_from_args
from ..schema import load_records
from ._common import input_shards, inputs_dir, shard_name, write_json, write_jsonl


def collect_texts(cfg: Config, platforms: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Unique (text_hash, text, platforms) across the requested platforms."""
    frames = []
    per_platform: dict[str, int] = {}
    for platform in platforms or cfg.platforms:
        df = load_records(cfg, platform, usecols=["platform", "text", "text_hash", "in_study_period"])
        df = df[df["in_study_period"] & df["text"].fillna("").astype(str).str.strip().ne("")]
        df = df.drop_duplicates("text_hash")
        per_platform[platform] = int(len(df))
        frames.append(df[["text_hash", "text", "platform"]])
    if not frames:
        return pd.DataFrame(columns=["text_hash", "text", "platforms"]), per_platform
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["text_hash"] = all_rows["text_hash"].astype(str).str.lower()
    grouped = (
        all_rows.groupby("text_hash", sort=True)
        .agg(text=("text", "first"), platforms=("platform", lambda s: "|".join(sorted(set(s)))))
        .reset_index()
    )
    return grouped, per_platform


def run(cfg: Config, platforms: list[str] | None = None, shard_size: int = 20000, sample: int | None = None,
        seed: int = 0, force: bool = False) -> dict[str, Any]:
    out_dir = inputs_dir(cfg)
    existing = input_shards(cfg)
    if existing and not force:
        print(f"{len(existing)} input shard(s) already exist in {out_dir}; use --force to rebuild")
        return {"status": "exists", "shards": [p.name for p in existing]}
    for old in existing:
        old.unlink()

    texts, per_platform = collect_texts(cfg, platforms)
    n_unique = int(len(texts))
    if sample is not None and sample < n_unique:
        rng = random.Random(seed)
        keep = sorted(rng.sample(range(n_unique), sample))
        texts = texts.iloc[keep].reset_index(drop=True)
    shard_size = max(int(shard_size), 1)
    shards = []
    for index, start in enumerate(range(0, len(texts), shard_size)):
        chunk = texts.iloc[start:start + shard_size]
        path = out_dir / f"{shard_name(index)}.jsonl"
        n = write_jsonl(path, ({"text_hash": r.text_hash, "text": str(r.text), "platforms": r.platforms} for r in chunk.itertuples()))
        shards.append({"shard": index, "file": path.name, "rows": n})
    summary = {
        "status": "written",
        "texts_per_platform": per_platform,
        "unique_texts": n_unique,
        "sampled": int(len(texts)) if sample is not None else None,
        "shard_size": shard_size,
        "shards": shards,
    }
    write_json(out_dir / "inputs_summary.json", summary)
    print(f"llm inputs: {len(texts):,} unique texts in {len(shards)} shard(s) -> {out_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_argument(parser)
    parser.add_argument("--platform", action="append", help="Restrict to one platform (repeatable).")
    parser.add_argument("--shard_size", type=int, default=20000)
    parser.add_argument("--sample", type=int, default=None, help="Random sample of N unique texts (e.g. for a scale check).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Rebuild shards even if they exist.")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    run(cfg, args.platform, args.shard_size, args.sample, args.seed, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
