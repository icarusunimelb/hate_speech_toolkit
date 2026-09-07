"""Project configuration.

One YAML file describes a project (platforms, study period, categories, events, analysis
settings).  Every stage takes ``--config project.yaml``.  Relative paths inside the YAML are
resolved against the YAML file's own folder, so a project folder is self-contained.

    from hst.config import load_config
    cfg = load_config("configs/examples/whitesupremacist.yaml")
    cfg.platforms                 -> ["x", "telegram", "instagram"]
    cfg.path("work")              -> absolute Path of the work folder
    cfg.categories                -> {"anti_zorb": {...}, ...}   (supervised categories)
    cfg.llm_categories            -> [...]                        (LLM categories)
    cfg.events()                  -> DataFrame(id, date, label)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PLATFORMS = ("x", "telegram", "instagram")
PLATFORM_LABEL = {"x": "X", "telegram": "Telegram", "instagram": "Instagram"}


class Config:
    def __init__(self, data: dict[str, Any], base_dir: Path):
        self._data = data
        self.base_dir = base_dir

    # ---- raw access -------------------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def section(self, name: str) -> dict[str, Any]:
        """A sub-dictionary (empty dict when absent)."""
        value = self._data.get(name) or {}
        if not isinstance(value, dict):
            raise ValueError(f"config section {name!r} must be a mapping")
        return value

    # ---- paths ------------------------------------------------------------------------
    def path(self, key: str, default: str | None = None) -> Path:
        value = self.section("paths").get(key, default)
        if value is None:
            raise KeyError(f"paths.{key} is not set in the config")
        p = Path(str(value))
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def resolve(self, value: str | Path) -> Path:
        p = Path(str(value))
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    # ---- LLM prompt files (bundled defaults) ------------------------------------------------
    def _llm_file(self, key: str, default_name: str) -> Path:
        """llm.<key> from the YAML if set, else the copy bundled with the package."""
        value = self.section("llm").get(key)
        if value:
            return self.resolve(value)
        return Path(__file__).resolve().parent / "llm" / "prompts" / default_name

    def llm_prompt_path(self) -> Path:
        return self._llm_file("prompt", "extremist_discourse_v5_1.txt")

    def llm_schema_path(self) -> Path:
        return self._llm_file("schema", "extremist_schema_v2.json")

    # ---- project facts ------------------------------------------------------------------
    @property
    def project(self) -> str:
        return str(self._data.get("project", "project"))

    @property
    def platforms(self) -> list[str]:
        plats = [str(p).lower() for p in self._data.get("platforms", ["x"])]
        bad = [p for p in plats if p not in PLATFORMS]
        if bad:
            raise ValueError(f"unknown platforms {bad}; supported: {list(PLATFORMS)}")
        return plats

    @property
    def timezone(self) -> str:
        return str(self._data.get("timezone", "Australia/Sydney"))

    @property
    def study_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.section("study_period")["start"])

    @property
    def study_end(self) -> pd.Timestamp:
        return pd.Timestamp(self.section("study_period")["end"])

    @property
    def categories(self) -> dict[str, dict[str, Any]]:
        """Supervised (RoBERTa) categories: name -> {label, target_group, threshold, model}."""
        cats = self.section("categories")
        out: dict[str, dict[str, Any]] = {}
        for name, spec in cats.items():
            spec = dict(spec or {})
            spec.setdefault("label", name.replace("_", " ").capitalize())
            spec.setdefault("threshold", 0.5)
            out[str(name)] = spec
        return out

    @property
    def category_names(self) -> list[str]:
        return list(self.categories)

    def category_label(self, name: str) -> str:
        if name in self.categories:
            return str(self.categories[name]["label"])
        llm = self.section("llm").get("categories") or {}
        if isinstance(llm, dict) and name in llm:
            return str((llm[name] or {}).get("label", name))
        return {"any_hate": "Any hate category", "any_extremist": "Any extremist category",
                "all": "All content"}.get(name, name.replace("_", " ").capitalize())

    def threshold(self, name: str) -> float:
        return float(self.categories[name]["threshold"])

    def model_dir(self, name: str) -> Path:
        spec = self.categories[name]
        model = spec.get("model") or f"{self.section('paths').get('models', 'models')}/{name}"
        return self.resolve(model)

    @property
    def llm_categories(self) -> list[str]:
        cats = self.section("llm").get("categories") or {}
        return [str(c) for c in (cats.keys() if isinstance(cats, dict) else cats)]

    def events(self) -> pd.DataFrame:
        rows = self._data.get("events") or []
        if not rows:
            return pd.DataFrame(columns=["id", "date", "label"])
        df = pd.DataFrame(rows)[["id", "date", "label"]]
        df["date"] = pd.to_datetime(df["date"])
        return df

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return Config(data, path.parent)


def add_config_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="Project YAML (see configs/project_template.yaml).")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return load_config(args.config)
