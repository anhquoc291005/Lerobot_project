#!/usr/bin/env python
"""Create a degree-unit copy of a local LeRobot dataset.

The SO101 MuJoCo collector stores joint observations and actions in radians.
This script copies a dataset root, then converts only these numeric features:

- observation.state
- action

It also scales the corresponding global and per-episode stats.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


RAD_TO_DEG = 180.0 / math.pi
FEATURES_TO_CONVERT = ("observation.state", "action")
STATS_TO_CONVERT = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")


def _scale_value(value: Any, factor: float) -> Any:
    if value is None:
        return None
    if isinstance(value, list):
        return [_scale_value(item, factor) for item in value]
    return float(value) * factor


def _scale_column(table: pa.Table, column_name: str, factor: float) -> pa.Table:
    column_index = table.schema.get_field_index(column_name)
    if column_index < 0:
        return table

    old_column = table[column_name]
    scaled_values = [_scale_value(value, factor) for value in old_column.to_pylist()]
    new_column = pa.array(scaled_values, type=old_column.type)
    return table.set_column(column_index, table.field(column_index), new_column)


def _rewrite_parquet(path: Path, column_names: tuple[str, ...], factor: float) -> bool:
    table = pq.read_table(path)
    changed = False

    for column_name in column_names:
        if table.schema.get_field_index(column_name) >= 0:
            table = _scale_column(table, column_name, factor)
            changed = True

    if changed:
        pq.write_table(table, path)

    return changed


def _convert_data_files(root: Path, factor: float) -> int:
    converted = 0
    for parquet_path in sorted((root / "data").rglob("*.parquet")):
        if _rewrite_parquet(parquet_path, FEATURES_TO_CONVERT, factor):
            converted += 1
    return converted


def _convert_stats_json(root: Path, factor: float) -> None:
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    for feature_name in FEATURES_TO_CONVERT:
        feature_stats = stats.get(feature_name)
        if not isinstance(feature_stats, dict):
            continue
        for stat_name in STATS_TO_CONVERT:
            if stat_name in feature_stats:
                feature_stats[stat_name] = _scale_value(feature_stats[stat_name], factor)

    stats_path.write_text(json.dumps(stats, indent=4), encoding="utf-8")


def _episode_stats_columns(table: pa.Table) -> tuple[str, ...]:
    columns: list[str] = []
    prefixes = tuple(f"stats/{feature_name}/" for feature_name in FEATURES_TO_CONVERT)

    for column_name in table.column_names:
        if not column_name.startswith(prefixes):
            continue
        stat_name = column_name.rsplit("/", maxsplit=1)[-1]
        if stat_name in STATS_TO_CONVERT:
            columns.append(column_name)

    return tuple(columns)


def _convert_episode_stats(root: Path, factor: float) -> int:
    converted = 0
    episodes_dir = root / "meta" / "episodes"
    for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
        table = pq.read_table(parquet_path)
        column_names = _episode_stats_columns(table)
        if not column_names:
            continue

        for column_name in column_names:
            table = _scale_column(table, column_name, factor)

        pq.write_table(table, parquet_path)
        converted += 1

    return converted


def _validate_roots(input_root: Path, output_root: Path, overwrite: bool) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset root does not exist: {input_root}")
    if output_root == input_root:
        raise ValueError("Output root must be different from input root.")
    if output_root.is_relative_to(input_root):
        raise ValueError("Output root must not be inside input root.")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset root already exists: {output_root}. "
                "Pass --overwrite if you intentionally want to replace it."
            )


def _make_temp_root(output_root: Path) -> Path:
    return output_root.with_name(f"{output_root.name}.tmp_rad_to_deg")


def _create_converted_copy(input_root: Path, output_root: Path, overwrite: bool) -> tuple[int, int]:
    _validate_roots(input_root, output_root, overwrite)
    temp_root = _make_temp_root(output_root)

    if temp_root.exists():
        shutil.rmtree(temp_root)

    try:
        shutil.copytree(input_root, temp_root)
        data_files = _convert_data_files(temp_root, RAD_TO_DEG)
        _convert_stats_json(temp_root, RAD_TO_DEG)
        episode_stats_files = _convert_episode_stats(temp_root, RAD_TO_DEG)

        if output_root.exists():
            shutil.rmtree(output_root)
        temp_root.rename(output_root)
        return data_files, episode_stats_files
    except Exception:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a local LeRobot dataset and convert action/state radians to degrees."
    )
    parser.add_argument("--input-root", required=True, type=Path, help="Existing local dataset root.")
    parser.add_argument("--output-root", required=True, type=Path, help="New converted dataset root.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output-root if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    data_files, episode_stats_files = _create_converted_copy(input_root, output_root, args.overwrite)

    print(f"Created degree-unit dataset copy: {output_root}")
    print(f"Converted data parquet files: {data_files}")
    print(f"Converted episode stats parquet files: {episode_stats_files}")
    print("Converted features: observation.state, action")


if __name__ == "__main__":
    main()
