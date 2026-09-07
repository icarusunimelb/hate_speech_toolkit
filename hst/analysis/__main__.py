"""Run the analysis stages in order:  python -m hst.analysis --config project.yaml [--only ...] [--platform ...]

Stages: daily, spikes, events, network, geography.  Tables are written under <work>/analysis/ and
figures under <work>/figures/.
"""

from __future__ import annotations

import argparse

from ..config import add_config_argument, config_from_args
from . import daily, events, geography, network, spikes

STAGES = ["daily", "spikes", "events", "network", "geography"]


def main(argv=None) -> int:
    parser = add_config_argument(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--only", nargs="+", choices=STAGES, default=STAGES, help="stages to run (default: all)")
    parser.add_argument("--platform", action="append", help="restrict to a platform (repeatable)")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    for stage in STAGES:
        if stage not in args.only:
            continue
        print(f"== {stage}")
        if stage == "daily":
            daily.run(cfg, args.platform)
        elif stage == "spikes":
            spikes.run(cfg, args.platform)
        elif stage == "events":
            events.run(cfg, args.platform)
        elif stage == "network":
            network.run(cfg, args.platform)
        elif stage == "geography":
            geography.run(cfg)
    print(f"tables: {cfg.path('work', 'work') / 'analysis'}")
    print(f"figures: {cfg.path('work', 'work') / 'figures'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
