#!/usr/bin/env python
"""Create a grid-balanced LeRobot dataset from paired sim/real runs.

The source datasets are expected to follow the same episode setup table.
By default (--episodes-per-domain) the script draws the same number of
episodes from both domains for a 50/50 mix. Pass --sim-episodes/
--real-episodes together for an asymmetric ratio (e.g. 72 sim + 128 real):
each domain is then stratified independently by object cell and relation,
so the object-cell/relation proportions inside each domain's subset match
the overall sim:real ratio.

Selection is balanced across object cells and relations. For the default
200-row table and 100 episodes per domain, each object cell contributes five
or six episodes from sim and the same five or six episodes from real.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import random
import shutil
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import (
    _copy_and_reindex_data,
    _copy_and_reindex_episodes_metadata,
    merge_datasets,
    split_dataset,
)
from lerobot.datasets.io_utils import write_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_SIM_REPO_ID = "ngocthuong2212/so101_mujoco_follower_7_31"
DEFAULT_REAL_REPO_ID = "vasco281204/so101_green_block"
DEFAULT_OUTPUT_REPO_ID = "vasco281204/so101_sim_real_mix_50_50_200"
DEFAULT_TABLE = Path("matrix_5x9_real_rule_selection_table.html")
DEFAULT_WORK_DIR = Path("datasets/so101_sim_real_mix_50_50_200_work")
DEFAULT_OUTPUT_ROOT = Path("datasets/so101_sim_real_mix_50_50_200")


@dataclass(frozen=True)
class SetupRule:
    episode_index: int
    object_cell: str
    container_cell: str
    relation: str
    distance_cells: float
    row_delta: int
    col_delta: int


class _SelectionTableParser(HTMLParser):
    """Read rows from the HTML table whose id is ``selection-table``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_selection_table = False
        self.in_cell = False
        self.current_cell: list[str] = []
        self.current_row: list[str] | None = None
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "table" and attrs_dict.get("id") == "selection-table":
            self.in_selection_table = True
        elif self.in_selection_table and tag == "tr":
            self.current_row = []
        elif self.in_selection_table and tag == "td" and self.current_row is not None:
            self.in_cell = True
            self.current_cell = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self.in_cell and self.current_row is not None:
            self.current_row.append("".join(self.current_cell).strip())
            self.current_cell = []
            self.in_cell = False
        elif tag == "tr" and self.in_selection_table and self.current_row is not None:
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = None
        elif tag == "table" and self.in_selection_table:
            self.in_selection_table = False


def parse_selection_table(path: Path) -> list[SetupRule]:
    parser = _SelectionTableParser()
    with path.open(encoding="utf-8") as table_file:
        parser.feed(table_file.read())

    rules: list[SetupRule] = []
    for row_number, row in enumerate(parser.rows, start=1):
        if len(row) != 7:
            raise ValueError(f"Malformed selection-table row {row_number}: expected 7 cells, got {row!r}")
        try:
            rules.append(
                SetupRule(
                    episode_index=int(row[0]),
                    object_cell=row[1].strip().lower(),
                    container_cell=row[2].strip().lower(),
                    relation=row[3].strip().lower(),
                    distance_cells=float(row[4]),
                    row_delta=int(row[5]),
                    col_delta=int(row[6]),
                )
            )
        except ValueError as exc:
            raise ValueError(f"Invalid value in selection-table row {row_number}: {row!r}") from exc

    if not rules:
        raise ValueError(f"No rows found in #selection-table from {path}")

    episode_indices = [rule.episode_index for rule in rules]
    if len(episode_indices) != len(set(episode_indices)):
        duplicates = sorted(index for index, count in Counter(episode_indices).items() if count > 1)
        raise ValueError(f"Duplicate episode indices in selection table: {duplicates}")
    if min(episode_indices) < 0:
        raise ValueError("Episode indices must be non-negative")

    return sorted(rules, key=lambda rule: rule.episode_index)


def _largest_remainder_quotas(
    counts: Counter[str],
    target_total: int,
    rng: random.Random,
) -> tuple[dict[str, int], list[str], int]:
    """Allocate integer quotas proportionally and expose tied cutoff groups.

    The returned list contains groups tied at the cutoff. Callers can try
    different combinations from that list when cross-stratum constraints make
    one largest-remainder allocation infeasible.
    """
    source_total = sum(counts.values())
    if not 0 < target_total <= source_total:
        raise ValueError(f"target_total must be in [1, {source_total}], got {target_total}")

    exact = {key: value * target_total / source_total for key, value in counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = target_total - sum(quotas.values())
    if remaining == 0:
        return quotas, [], 0

    remainders = {key: exact[key] - quotas[key] for key in counts}
    levels = sorted(set(remainders.values()), reverse=True)
    for level in levels:
        tied = sorted(key for key, remainder in remainders.items() if remainder == level)
        rng.shuffle(tied)
        if remaining >= len(tied):
            for key in tied:
                quotas[key] += 1
            remaining -= len(tied)
            if remaining == 0:
                return quotas, [], 0
        else:
            return quotas, tied, remaining

    raise AssertionError("Largest-remainder allocation did not reach target total")


def _add_capacity(
    capacities: dict[Any, dict[Any, int]],
    adjacency: dict[Any, set[Any]],
    source: Any,
    target: Any,
    capacity: int,
) -> None:
    capacities[source][target] = capacity
    capacities[target].setdefault(source, 0)
    adjacency[source].add(target)
    adjacency[target].add(source)


def _object_relation_flow(
    rules: list[SetupRule],
    object_quotas: dict[str, int],
    relation_quotas: dict[str, int],
) -> dict[tuple[str, str], int] | None:
    """Solve exact object/relation quotas as a small integer max-flow."""
    available = Counter((rule.object_cell, rule.relation) for rule in rules)
    capacities: dict[Any, dict[Any, int]] = defaultdict(dict)
    adjacency: dict[Any, set[Any]] = defaultdict(set)
    source = ("source",)
    sink = ("sink",)

    for cell, quota in object_quotas.items():
        _add_capacity(capacities, adjacency, source, ("object", cell), quota)
    for relation, quota in relation_quotas.items():
        _add_capacity(capacities, adjacency, ("relation", relation), sink, quota)
    for (cell, relation), count in available.items():
        _add_capacity(
            capacities,
            adjacency,
            ("object", cell),
            ("relation", relation),
            count,
        )

    original = {node: dict(edges) for node, edges in capacities.items()}
    flow = 0
    target_flow = sum(object_quotas.values())
    while True:
        parent: dict[Any, Any | None] = {source: None}
        queue: deque[Any] = deque([source])
        while queue and sink not in parent:
            node = queue.popleft()
            for neighbor in sorted(adjacency[node], key=str):
                if neighbor not in parent and capacities[node].get(neighbor, 0) > 0:
                    parent[neighbor] = node
                    queue.append(neighbor)
        if sink not in parent:
            break

        increment = target_flow
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            increment = min(increment, capacities[previous][node])
            node = previous
        node = sink
        while parent[node] is not None:
            previous = parent[node]
            capacities[previous][node] -= increment
            capacities[node][previous] = capacities[node].get(previous, 0) + increment
            node = previous
        flow += increment

    if flow != target_flow:
        return None

    group_flow: dict[tuple[str, str], int] = {}
    for cell in object_quotas:
        object_node = ("object", cell)
        for relation in relation_quotas:
            relation_node = ("relation", relation)
            initial_capacity = original.get(object_node, {}).get(relation_node, 0)
            if initial_capacity:
                selected = initial_capacity - capacities[object_node][relation_node]
                if selected:
                    group_flow[(cell, relation)] = selected
    return group_flow


def _choose_object_quotas(
    rules: list[SetupRule],
    target_total: int,
    relation_quotas: dict[str, int],
    seed: int,
) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    counts = Counter(rule.object_cell for rule in rules)
    rng = random.Random(seed)
    base_quotas, tied_cells, tied_slots = _largest_remainder_quotas(counts, target_total, rng)

    if not tied_cells:
        flow = _object_relation_flow(rules, base_quotas, relation_quotas)
        if flow is None:
            raise RuntimeError("Object and relation quotas are not jointly feasible")
        return base_quotas, flow

    combinations = list(itertools.combinations(sorted(tied_cells), tied_slots))
    rng.shuffle(combinations)
    for rounded_up_cells in combinations:
        quotas = dict(base_quotas)
        for cell in rounded_up_cells:
            quotas[cell] += 1
        flow = _object_relation_flow(rules, quotas, relation_quotas)
        if flow is not None:
            return quotas, flow

    raise RuntimeError(
        "Could not find a jointly feasible grid/relation allocation. "
        "The selection table may be too sparse for the requested count."
    )


def select_balanced_rules(rules: list[SetupRule], count: int, seed: int) -> list[SetupRule]:
    if not 0 < count <= len(rules):
        raise ValueError(f"count must be in [1, {len(rules)}], got {count}")

    relation_counts = Counter(rule.relation for rule in rules)
    relation_rng = random.Random(seed + 1)
    relation_quotas, tied_relations, tied_slots = _largest_remainder_quotas(
        relation_counts, count, relation_rng
    )
    if tied_relations:
        # Relation quotas in the 200-row rule table are all even for a 100-row
        # selection. Keep generic behavior deterministic for other tables.
        for relation in tied_relations[:tied_slots]:
            relation_quotas[relation] += 1

    object_quotas, group_flow = _choose_object_quotas(rules, count, relation_quotas, seed)

    grouped: dict[tuple[str, str], list[SetupRule]] = defaultdict(list)
    for rule in rules:
        grouped[(rule.object_cell, rule.relation)].append(rule)

    rng = random.Random(seed + 2)
    selected: list[SetupRule] = []
    for group, group_count in sorted(group_flow.items()):
        candidates = list(grouped[group])
        rng.shuffle(candidates)
        container_counts: Counter[str] = Counter()
        chosen: list[SetupRule] = []
        while len(chosen) < group_count:
            candidate = min(
                candidates,
                key=lambda rule: (container_counts[rule.container_cell], rng.random()),
            )
            candidates.remove(candidate)
            chosen.append(candidate)
            container_counts[candidate.container_cell] += 1
        selected.extend(chosen)

    selected.sort(key=lambda rule: rule.episode_index)
    _validate_selection(selected, count, object_quotas, relation_quotas)
    return selected


def _validate_selection(
    selected: list[SetupRule],
    expected_count: int,
    object_quotas: dict[str, int],
    relation_quotas: dict[str, int],
) -> None:
    if len(selected) != expected_count:
        raise AssertionError(f"Selected {len(selected)} rows, expected {expected_count}")
    if len({rule.episode_index for rule in selected}) != expected_count:
        raise AssertionError("Selected episode indices are not unique")
    if Counter(rule.object_cell for rule in selected) != Counter(object_quotas):
        raise AssertionError("Selected object-cell distribution does not match quotas")
    if Counter(rule.relation for rule in selected) != Counter(relation_quotas):
        raise AssertionError("Selected relation distribution does not match quotas")


def _reject_viewer_repo_id(repo_id: str, argument_name: str) -> None:
    if "visualize_dataset" in repo_id or repo_id.startswith("spaces/"):
        raise ValueError(
            f"{argument_name}={repo_id!r} is a dataset viewer/Space, not a dataset repo id. "
            "Pass the real source shown in the viewer, for example "
            "'vasco281204/so101_green_block'."
        )
    if repo_id.count("/") != 1:
        raise ValueError(f"{argument_name} must look like 'owner/dataset_name', got {repo_id!r}")


def _canonicalize_features(features: dict[str, dict]) -> dict[str, dict]:
    """Normalize equivalent LeRobot video metadata spellings."""
    canonical = copy.deepcopy(features)
    for feature in canonical.values():
        info = feature.get("info")
        if not isinstance(info, dict) or "video.is_depth_map" not in info:
            continue
        value = info.pop("video.is_depth_map")
        if "is_depth_map" in info and info["is_depth_map"] != value:
            raise ValueError(
                "Conflicting video depth-map metadata: "
                f"video.is_depth_map={value!r}, is_depth_map={info['is_depth_map']!r}"
            )
        info["is_depth_map"] = value
    return canonical


def _validate_source_compatibility(sim: LeRobotDataset, real: LeRobotDataset) -> dict[str, dict]:
    problems: list[str] = []
    if sim.meta.fps != real.meta.fps:
        problems.append(f"fps: sim={sim.meta.fps}, real={real.meta.fps}")
    if sim.meta.robot_type != real.meta.robot_type:
        problems.append(f"robot_type: sim={sim.meta.robot_type!r}, real={real.meta.robot_type!r}")
    sim_features = _canonicalize_features(sim.meta.features)
    real_features = _canonicalize_features(real.meta.features)
    if sim_features != real_features:
        sim_keys = set(sim.meta.features)
        real_keys = set(real.meta.features)
        problems.append(
            "features differ "
            f"(sim_only={sorted(sim_keys - real_keys)}, real_only={sorted(real_keys - sim_keys)})"
        )
    if problems:
        raise ValueError("Source datasets cannot be merged directly:\n- " + "\n- ".join(problems))
    return real_features


def _write_canonical_features(dataset: LeRobotDataset, features: dict[str, dict]) -> None:
    """Make generated subset metadata byte-for-byte compatible for aggregation."""
    dataset.meta.info.features = copy.deepcopy(features)
    write_info(dataset.meta.info, dataset.root)


def _ensure_available_episodes(dataset: LeRobotDataset, selected_indices: list[int], domain: str) -> None:
    missing = [index for index in selected_indices if index >= dataset.meta.total_episodes]
    if missing:
        raise ValueError(
            f"{domain} dataset has {dataset.meta.total_episodes} episodes but selection needs "
            f"indices up to {max(selected_indices)}; missing={missing}"
        )


def _validate_sim_matrix_log(sim: LeRobotDataset, selected: list[SetupRule]) -> None:
    log_path = sim.root / "meta" / "matrix_random_placements.jsonl"
    if not log_path.exists():
        logging.warning("Sim matrix log not found; cannot verify setup-table alignment: %s", log_path)
        return

    records: dict[int, dict[str, Any]] = {}
    with log_path.open(encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                records[int(record["episode_index"])] = record
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid sim matrix log line {line_number}: {log_path}") from exc

    mismatches: list[str] = []
    for rule in selected:
        record = records.get(rule.episode_index)
        if record is None:
            mismatches.append(f"episode {rule.episode_index}: missing log record")
            continue
        placements = record.get("matrix_random_placements", {})
        object_cell = placements.get("pick_box", {}).get("cell")
        container_cell = placements.get("container_box", {}).get("cell")
        if object_cell != rule.object_cell or container_cell != rule.container_cell:
            mismatches.append(
                f"episode {rule.episode_index}: log=({object_cell},{container_cell}) "
                f"table=({rule.object_cell},{rule.container_cell})"
            )
    if mismatches:
        preview = "\n- ".join(mismatches[:10])
        raise ValueError(f"Sim matrix log does not match the HTML setup table:\n- {preview}")
    logging.info("Verified %d selected sim episodes against %s", len(selected), log_path)


def _manifest_records(
    sim_selected: list[SetupRule],
    real_selected: list[SetupRule],
    sim_repo_id: str,
    real_repo_id: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    for domain, repo_id, selected in (
        ("sim", sim_repo_id, sim_selected),
        ("real", real_repo_id, real_selected),
    ):
        ordered = sorted(selected, key=lambda rule: rule.episode_index)
        for subset_episode_index, rule in enumerate(ordered):
            records.append(
                {
                    "new_episode_index": offset + subset_episode_index,
                    "domain": domain,
                    "source_repo_id": repo_id,
                    "source_episode_index": rule.episode_index,
                    "object_cell": rule.object_cell,
                    "container_cell": rule.container_cell,
                    "relation": rule.relation,
                    "distance_cells": rule.distance_cells,
                    "row_delta": rule.row_delta,
                    "col_delta": rule.col_delta,
                }
            )
        offset += len(ordered)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _build_summary(
    rules: list[SetupRule],
    sim_selected: list[SetupRule],
    real_selected: list[SetupRule],
    args: argparse.Namespace,
) -> dict[str, Any]:
    def distribution(selected: list[SetupRule], key: Any) -> dict[str, int]:
        return dict(sorted(Counter(key(rule) for rule in selected).items()))

    sim_object = distribution(sim_selected, lambda rule: rule.object_cell)
    real_object = distribution(real_selected, lambda rule: rule.object_cell)
    sim_relation = distribution(sim_selected, lambda rule: rule.relation)
    real_relation = distribution(real_selected, lambda rule: rule.relation)
    mixed_object = dict(sorted((Counter(sim_object) + Counter(real_object)).items()))
    mixed_relation = dict(sorted((Counter(sim_relation) + Counter(real_relation)).items()))

    return {
        "sim_repo_id": args.sim_repo_id,
        "real_repo_id": args.real_repo_id,
        "sim_root": str(args.sim_root.resolve()) if args.sim_root else None,
        "real_root": str(args.real_root.resolve()) if args.real_root else None,
        "output_repo_id": args.output_repo_id,
        "selection_table": str(args.selection_table.resolve()),
        "seed": args.seed,
        "source_rule_count": len(rules),
        "sim_episodes": len(sim_selected),
        "real_episodes": len(real_selected),
        "total_output_episodes": len(sim_selected) + len(real_selected),
        "selected_source_episode_indices": {
            "sim": [rule.episode_index for rule in sim_selected],
            "real": [rule.episode_index for rule in real_selected],
        },
        "object_cell_counts_per_domain": {"sim": sim_object, "real": real_object},
        "object_cell_counts_mixed": mixed_object,
        "relation_counts_per_domain": {"sim": sim_relation, "real": real_relation},
        "relation_counts_mixed": mixed_relation,
        "ordering": "sim episodes first, then real episodes; see mix_manifest.jsonl for provenance",
        "video_subset_mode": args.video_subset_mode,
        "assumption": (
            "Both sources follow the same HTML rule table with source episode_index aligned to its row."
        ),
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")


def _add_output_frame_summary(
    summary: dict[str, Any],
    mixed: LeRobotDataset,
    sim_episode_count: int,
    real_episode_count: int,
) -> None:
    episodes = mixed.meta.episodes
    total_episodes = sim_episode_count + real_episode_count
    if episodes is None or len(episodes) != total_episodes:
        raise RuntimeError("Merged episode metadata is unavailable or has an unexpected length")
    sim_frames = sum(int(episodes[index]["length"]) for index in range(sim_episode_count))
    real_frames = sum(
        int(episodes[index]["length"]) for index in range(sim_episode_count, total_episodes)
    )
    total_frames = sim_frames + real_frames
    summary["frames"] = {
        "sim": sim_frames,
        "real": real_frames,
        "total": total_frames,
        "sim_ratio": sim_frames / total_frames,
        "real_ratio": real_frames / total_frames,
    }
    expected_sim_ratio = sim_episode_count / total_episodes
    if total_frames and abs(sim_frames / total_frames - expected_sim_ratio) > 0.05:
        logging.warning(
            "The mix targets sim=%.1f%%/real=%.1f%% by episode, but frames are sim=%.1f%% and "
            "real=%.1f%%. Frame-based training will not match the intended domain ratio unless a "
            "balanced sampler is used.",
            100 * expected_sim_ratio,
            100 * (1 - expected_sim_ratio),
            100 * sim_frames / total_frames,
            100 * real_frames / total_frames,
        )


def _assert_fresh_output(path: Path, description: str) -> None:
    if path.exists():
        raise FileExistsError(
            f"{description} already exists: {path}. Choose a new path or move the existing directory first."
        )


def _link_or_copy(src: Path, dst: Path) -> str:
    """Hard-link a video when possible, otherwise copy it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src.resolve(strict=True), dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def _copy_selected_video_files_without_trimming(
    src_dataset: LeRobotDataset,
    dst_meta: LeRobotDatasetMetadata,
    episode_mapping: dict[int, int],
) -> dict[int, dict[str, Any]]:
    """Copy whole source video files and retain the original episode timestamps.

    This is much faster than LeRobot's regular split operation because no AV1
    frames are decoded or encoded. A copied file may contain unreferenced video
    segments, but the subset metadata only exposes the selected episodes.
    """
    if src_dataset.meta.episodes is None:
        raise RuntimeError("Source episode metadata is unavailable")

    video_metadata: dict[int, dict[str, Any]] = {
        new_idx: {} for new_idx in episode_mapping.values()
    }
    copied_files: set[Path] = set()
    link_count = 0
    copy_count = 0

    for video_key in src_dataset.meta.video_keys:
        for old_idx, new_idx in sorted(episode_mapping.items(), key=lambda item: item[1]):
            relative_path = src_dataset.meta.get_video_file_path(old_idx, video_key)
            if relative_path not in copied_files:
                result = _link_or_copy(src_dataset.root / relative_path, dst_meta.root / relative_path)
                link_count += result == "linked"
                copy_count += result == "copied"
                copied_files.add(relative_path)

            src_episode = src_dataset.meta.episodes[old_idx]
            for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
                key = f"videos/{video_key}/{field}"
                video_metadata[new_idx][key] = src_episode[key]

    logging.info(
        "Fast video subset retained %d whole source files (%d linked, %d copied)",
        len(copied_files),
        link_count,
        copy_count,
    )
    return video_metadata


def _split_dataset_copying_full_videos(
    dataset: LeRobotDataset,
    selected_indices: list[int],
    output_root: Path,
) -> LeRobotDataset:
    """Create a selected subset without trimming or re-encoding video files."""
    split_repo_id = f"{dataset.repo_id}_selected"
    episode_mapping = {
        old_idx: new_idx for new_idx, old_idx in enumerate(sorted(selected_indices))
    }
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=split_repo_id,
        fps=dataset.meta.fps,
        features=dataset.meta.features,
        robot_type=dataset.meta.robot_type,
        root=output_root,
        use_videos=bool(dataset.meta.video_keys),
        chunks_size=dataset.meta.chunks_size,
        data_files_size_in_mb=dataset.meta.data_files_size_in_mb,
        video_files_size_in_mb=dataset.meta.video_files_size_in_mb,
    )

    video_metadata = None
    if dataset.meta.video_keys:
        video_metadata = _copy_selected_video_files_without_trimming(
            dataset,
            new_meta,
            episode_mapping,
        )
    data_metadata = _copy_and_reindex_data(dataset, new_meta, episode_mapping)
    _copy_and_reindex_episodes_metadata(
        dataset,
        new_meta,
        episode_mapping,
        data_metadata,
        video_metadata,
    )
    return LeRobotDataset(
        repo_id=split_repo_id,
        root=output_root,
        image_transforms=dataset.image_transforms,
        delta_timestamps=dataset.delta_timestamps,
        tolerance_s=dataset.tolerance_s,
    )


def _load_completed_dataset(
    root: Path,
    repo_id: str,
    expected_episodes: int,
) -> LeRobotDataset | None:
    """Return a locally completed dataset, without treating partial output as valid."""
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        with info_path.open(encoding="utf-8") as info_file:
            info = json.load(info_file)
    except (OSError, json.JSONDecodeError):
        return None
    if info.get("total_episodes") != expected_episodes:
        return None

    required_patterns = (
        "data/*/*.parquet",
        "meta/episodes/*/*.parquet",
        "meta/tasks.parquet",
        "meta/stats.json",
    )
    if any(not list(root.glob(pattern)) for pattern in required_patterns):
        return None

    try:
        dataset = LeRobotDataset(repo_id=repo_id, root=root, download_videos=False)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return None
    for episode_index in range(expected_episodes):
        if not (root / dataset.meta.get_data_file_path(episode_index)).is_file():
            return None
        for video_key in dataset.meta.video_keys:
            if not (root / dataset.meta.get_video_file_path(episode_index, video_key)).is_file():
                return None
    return dataset


def _move_partial_output_aside(path: Path) -> Path:
    """Preserve an incomplete run under a deterministic backup name."""
    suffix = 1
    while True:
        backup = path.with_name(f"{path.name}.incomplete-{suffix}")
        if not backup.exists():
            path.rename(backup)
            return backup
        suffix += 1


def _materialize_or_resume_subset(
    dataset: LeRobotDataset,
    selected_indices: list[int],
    subset_parent: Path,
    video_subset_mode: str,
    resume: bool,
    domain: str,
) -> LeRobotDataset:
    subset_root = subset_parent / "selected"
    split_repo_id = f"{dataset.repo_id}_selected"
    if subset_root.exists() and resume:
        completed = _load_completed_dataset(subset_root, split_repo_id, len(selected_indices))
        if completed is not None:
            logging.info("Reusing completed %s subset at %s", domain, subset_root)
            return completed
        backup = _move_partial_output_aside(subset_root)
        logging.warning("Moved incomplete %s subset to %s", domain, backup)
    else:
        _assert_fresh_output(subset_root, f"{domain.capitalize()} subset")

    logging.info("Materializing selected %s subset (video mode: %s)", domain, video_subset_mode)
    if video_subset_mode == "trim":
        return split_dataset(
            dataset,
            {"selected": selected_indices},
            output_dir=subset_parent,
        )["selected"]
    return _split_dataset_copying_full_videos(dataset, selected_indices, subset_root)


def create_mixed_dataset(
    args: argparse.Namespace,
    sim_selected: list[SetupRule],
    real_selected: list[SetupRule],
) -> LeRobotDataset:
    sim_indices = sorted(rule.episode_index for rule in sim_selected)
    real_indices = sorted(rule.episode_index for rule in real_selected)
    total_episodes = len(sim_indices) + len(real_indices)
    sim_subset_parent = args.work_dir / "sim_subset"
    real_subset_parent = args.work_dir / "real_subset"
    if args.output_root.exists() and args.resume:
        completed = _load_completed_dataset(
            args.output_root,
            args.output_repo_id,
            total_episodes,
        )
        if completed is not None:
            logging.info("Reusing completed mixed dataset at %s", args.output_root)
            return completed
        backup = _move_partial_output_aside(args.output_root)
        logging.warning("Moved incomplete mixed output to %s", backup)
    else:
        _assert_fresh_output(args.output_root, "Mixed dataset output")

    logging.info("Loading %d selected sim episodes from %s", len(sim_indices), args.sim_repo_id)
    sim = LeRobotDataset(
        args.sim_repo_id,
        root=args.sim_root,
        episodes=sim_indices,
        download_videos=True,
    )
    logging.info("Loading %d selected real episodes from %s", len(real_indices), args.real_repo_id)
    real = LeRobotDataset(
        args.real_repo_id,
        root=args.real_root,
        episodes=real_indices,
        download_videos=True,
    )
    _ensure_available_episodes(sim, sim_indices, "sim")
    _ensure_available_episodes(real, real_indices, "real")
    _validate_sim_matrix_log(sim, sim_selected)
    canonical_features = _validate_source_compatibility(sim, real)

    sim_subset = _materialize_or_resume_subset(
        sim,
        sim_indices,
        sim_subset_parent,
        args.video_subset_mode,
        args.resume,
        "sim",
    )
    real_subset = _materialize_or_resume_subset(
        real,
        real_indices,
        real_subset_parent,
        args.video_subset_mode,
        args.resume,
        "real",
    )
    _write_canonical_features(sim_subset, canonical_features)
    _write_canonical_features(real_subset, canonical_features)

    logging.info("Merging subsets into %s", args.output_root)
    mixed = merge_datasets(
        [sim_subset, real_subset],
        output_repo_id=args.output_repo_id,
        output_dir=args.output_root,
    )
    if mixed.meta.total_episodes != total_episodes:
        raise RuntimeError(
            f"Merged dataset has {mixed.meta.total_episodes} episodes; expected {total_episodes}"
        )
    return mixed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a grid-balanced sim-real LeRobot dataset using the 5x9 HTML setup-rule table. "
            "Defaults to a symmetric 50/50 mix; pass --sim-episodes/--real-episodes for an "
            "asymmetric ratio."
        )
    )
    parser.add_argument("--selection-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--sim-repo-id", default=DEFAULT_SIM_REPO_ID)
    parser.add_argument("--real-repo-id", default=DEFAULT_REAL_REPO_ID)
    parser.add_argument("--sim-root", type=Path, help="Optional local root for the sim source dataset.")
    parser.add_argument("--real-root", type=Path, help="Optional local root for the real source dataset.")
    parser.add_argument("--output-repo-id", default=DEFAULT_OUTPUT_REPO_ID)
    parser.add_argument(
        "--episodes-per-domain",
        type=int,
        default=100,
        help="Episode count per domain for a symmetric mix. Ignored if --sim-episodes/--real-episodes are set.",
    )
    parser.add_argument(
        "--sim-episodes",
        type=int,
        default=None,
        help="Episode count to draw from the sim domain for an asymmetric mix. Requires --real-episodes.",
    )
    parser.add_argument(
        "--real-episodes",
        type=int,
        default=None,
        help="Episode count to draw from the real domain for an asymmetric mix. Requires --sim-episodes.",
    )
    parser.add_argument("--seed", type=int, default=2212)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--video-subset-mode",
        choices=("copy-full", "trim"),
        default="copy-full",
        help=(
            "copy-full avoids AV1 re-encoding by retaining whole source video files (fast, larger); "
            "trim keeps only selected frames (slow, compact)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse completed subsets/output from an earlier run. Incomplete directories are preserved "
            "with an .incomplete-N suffix before rebuilding."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only parse/select/write manifests; do not download or merge datasets.",
    )
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--upload-large-folder",
        action="store_true",
        help="Use Hugging Face's resumable large-folder uploader.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
        force=True,
    )
    _reject_viewer_repo_id(args.sim_repo_id, "--sim-repo-id")
    _reject_viewer_repo_id(args.real_repo_id, "--real-repo-id")
    _reject_viewer_repo_id(args.output_repo_id, "--output-repo-id")

    if (args.sim_episodes is None) != (args.real_episodes is None):
        raise ValueError("--sim-episodes and --real-episodes must be provided together")
    sim_count, real_count = (
        (args.sim_episodes, args.real_episodes)
        if args.sim_episodes is not None
        else (args.episodes_per_domain, args.episodes_per_domain)
    )

    rules = parse_selection_table(args.selection_table)
    sim_selected = select_balanced_rules(rules, sim_count, args.seed)
    real_selected = select_balanced_rules(rules, real_count, args.seed + 10_000)
    summary = _build_summary(rules, sim_selected, real_selected, args)
    manifest = _manifest_records(sim_selected, real_selected, args.sim_repo_id, args.real_repo_id)

    plan_dir = args.work_dir / "plan"
    _write_jsonl(plan_dir / "mix_manifest.jsonl", manifest)
    _write_summary(plan_dir / "mix_summary.json", summary)
    logging.info("Selected source episodes: %s", summary["selected_source_episode_indices"])
    logging.info("Object cells per domain: %s", summary["object_cell_counts_per_domain"])
    logging.info("Relations per domain: %s", summary["relation_counts_per_domain"])
    logging.info("Wrote selection plan to %s", plan_dir)

    if args.plan_only:
        return

    mixed = create_mixed_dataset(args, sim_selected, real_selected)
    _add_output_frame_summary(summary, mixed, len(sim_selected), len(real_selected))
    _write_summary(plan_dir / "mix_summary.json", summary)
    output_meta = mixed.root / "meta"
    _write_jsonl(output_meta / "mix_manifest.jsonl", manifest)
    _write_summary(output_meta / "mix_summary.json", summary)
    logging.info(
        "Created mixed dataset: repo_id=%s episodes=%d frames=%d root=%s",
        mixed.repo_id,
        mixed.meta.total_episodes,
        mixed.meta.total_frames,
        mixed.root,
    )

    if args.push_to_hub:
        logging.info("Pushing %s to Hugging Face Hub", mixed.repo_id)
        ratio_tag = "50-50" if len(sim_selected) == len(real_selected) else "mixed-ratio"
        mixed.push_to_hub(
            tags=["lerobot", "so101", "sim2real", "grid-balanced", ratio_tag],
            private=args.private,
            upload_large_folder=args.upload_large_folder,
        )
        logging.info("Push completed")
    else:
        logging.info("Push skipped; pass --push-to-hub after reviewing the local dataset")


if __name__ == "__main__":
    main()
