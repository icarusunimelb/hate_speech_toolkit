"""Plot style, colours and output folders shared by the analysis modules.

Tables go to  <work>/analysis/<area>/[<platform>/]  and figures to  <work>/figures/<area>/[<platform>/].
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from ..config import PLATFORM_LABEL, Config  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
           "#7f5539", "#0a9396", "#bb3e03", "#6a4c93"]
PLATFORM_COLOR = {"x": "#2a78d6", "telegram": "#eb6834", "instagram": "#1baf7a"}
NEUTRAL = "#8a8984"
INK = "#0b0b0b"
INK2 = "#52514e"
SEQ_CMAP = "Blues"
ROLL = 7


def platform_label(platform: str) -> str:
    return PLATFORM_LABEL.get(platform, platform)


def category_colors(cfg: Config) -> dict[str, str]:
    """A fixed colour per category name (supervised first, then LLM, then the aggregates)."""
    names = list(cfg.category_names) + list(cfg.llm_categories)
    colours = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(names)}
    colours.update({"any_hate": INK2, "any_extremist": INK2, "all": NEUTRAL, "records": NEUTRAL})
    return colours


def category_label(cfg: Config, name: str) -> str:
    return cfg.category_label(name)


def table_dir(cfg: Config, area: str, platform: str | None = None) -> Path:
    p = cfg.path("work", "work") / "analysis" / area
    if platform:
        p = p / platform
    p.mkdir(parents=True, exist_ok=True)
    return p


def fig_dir(cfg: Config, area: str, platform: str | None = None) -> Path:
    p = cfg.path("work", "work") / "figures" / area
    if platform:
        p = p / platform
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def style_axes(ax, title: str | None = None, ylabel: str | None = None, xlabel: str | None = None) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(axis="y", color="#e5e4df", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=8)


def add_event_markers(ax, cfg: Config, label: bool = True, y_frac: float = 0.97) -> None:
    for e in cfg.events().itertuples():
        x = pd.Timestamp(e.date)
        ax.axvline(x, color="#b5b4ad", linestyle=(0, (3, 3)), linewidth=0.8, zorder=1)
        if label:
            ax.annotate(str(e.id), (x, y_frac), xycoords=("data", "axes fraction"), fontsize=6,
                        color=INK2, ha="center", va="top")


def xaxis_dates(ax, span_days: int | None = None) -> None:
    """Month ticks for long series, day ticks for short windows."""
    if span_days is not None and span_days <= 60:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, span_days // 12)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(axis="x", labelsize=7, rotation=45)
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.tick_params(axis="x", labelsize=7)


def rolling(s: pd.Series, window: int = ROLL) -> pd.Series:
    return s.rolling(window, min_periods=max(2, window // 2)).mean()


def save_fig(fig, path: Path, dpi: int = 170) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
