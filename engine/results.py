"""What `python main.py run` writes to data/out, so the app never retrains."""
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TABLES = ("predictions", "importance", "market", "coverage")


@dataclass
class RunResults:
    predictions: pd.DataFrame
    importance: pd.DataFrame
    market: pd.DataFrame
    coverage: pd.DataFrame
    run_info: dict


def save_results(results: RunResults, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        getattr(results, name).to_parquet(out_dir / f"{name}.parquet")
    (out_dir / "run_info.json").write_text(json.dumps(results.run_info, indent=2, default=str))


def load_results(out_dir: Path) -> RunResults:
    tables = {name: pd.read_parquet(out_dir / f"{name}.parquet") for name in TABLES}
    return RunResults(**tables, run_info=json.loads((out_dir / "run_info.json").read_text()))
