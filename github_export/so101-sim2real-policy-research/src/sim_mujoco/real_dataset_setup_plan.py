from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_NUM_EPISODES = 200
GRID_ROWS = ("a", "b", "c", "d", "e")
GRID_COLS = tuple(range(1, 10))

REACHABLE_BOUNDARY_BY_COL = {
    1: "a",
    2: "b",
    3: "c",
    4: "d",
    5: "d",
    6: "d",
    7: "c",
    8: "b",
    9: "a",
}
ROBOT_BASE_CELLS = {"a4", "a5"}
OBJECT_REST_CELLS = {"a3", "a4", "a5", "a6", "b4", "b5"}
EXCLUDED_CELLS = ROBOT_BASE_CELLS | OBJECT_REST_CELLS
REACHABLE_CELLS = tuple(
    f"{row}{col}"
    for col in GRID_COLS
    for row in GRID_ROWS[: GRID_ROWS.index(REACHABLE_BOUNDARY_BY_COL[col]) + 1]
    if f"{row}{col}" not in EXCLUDED_CELLS
)

OBJECT_CELLS = REACHABLE_CELLS
CONTAINER_CELLS = REACHABLE_CELLS

RELATION_QUOTAS = {
    "canh_sat": 20,
    "sat": 16,
    "gan": 20,
    "vua": 20,
    "xa": 20,
    "trai": 18,
    "phai": 18,
    "tren": 18,
    "duoi": 18,
    "cheo_gan": 12,
    "cheo_vua": 10,
    "cheo_xa": 6,
    "random": 4,
}


@dataclass(frozen=True)
class SetupEpisode:
    episode_index: int
    object_cell: str
    container_cell: str
    relation: str
    distance_cells: float
    row_delta: int
    col_delta: int


def cell_to_row_col(cell: str) -> tuple[int, int]:
    cell = cell.strip().lower()
    if len(cell) < 2:
        raise ValueError(f"Invalid cell: {cell!r}")

    row_name = cell[0]
    col = int(cell[1:])
    if row_name not in GRID_ROWS or col not in GRID_COLS:
        raise ValueError(f"Invalid cell: {cell!r}")
    return GRID_ROWS.index(row_name), col - 1


def row_col_to_cell(row: int, col: int) -> str:
    return f"{GRID_ROWS[row]}{col + 1}"


def relation_between(object_cell: str, container_cell: str) -> tuple[float, int, int]:
    obj_row, obj_col = cell_to_row_col(object_cell)
    box_row, box_col = cell_to_row_col(container_cell)
    row_delta = box_row - obj_row
    col_delta = box_col - obj_col
    return math.hypot(row_delta, col_delta), row_delta, col_delta


def generate_setup_plan(
    num_episodes: int = DEFAULT_NUM_EPISODES,
    seed: int = 2212,
) -> list[SetupEpisode]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0.")

    rng = random.Random(seed)
    relation_sequence = _make_relation_sequence(num_episodes, rng)
    object_counts: dict[str, int] = {}
    container_counts: dict[str, int] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    episodes: list[SetupEpisode] = []

    for episode_index, relation in enumerate(relation_sequence):
        candidates = _relation_candidates(relation)
        if not candidates:
            raise RuntimeError(f"No valid object/container candidates for relation {relation!r}.")

        fresh_candidates = [pair for pair in candidates if pair_counts.get(pair, 0) == 0]
        if fresh_candidates:
            candidates = fresh_candidates

        rng.shuffle(candidates)
        object_cell, container_cell = min(
            candidates,
            key=lambda pair: (
                pair_counts.get(pair, 0),
                object_counts.get(pair[0], 0),
                container_counts.get(pair[1], 0),
                object_counts.get(pair[0], 0) + container_counts.get(pair[1], 0),
                rng.random(),
            ),
        )
        object_counts[object_cell] = object_counts.get(object_cell, 0) + 1
        container_counts[container_cell] = container_counts.get(container_cell, 0) + 1
        pair_counts[(object_cell, container_cell)] = pair_counts.get((object_cell, container_cell), 0) + 1
        distance, row_delta, col_delta = relation_between(object_cell, container_cell)
        episodes.append(
            SetupEpisode(
                episode_index=episode_index,
                object_cell=object_cell,
                container_cell=container_cell,
                relation=relation,
                distance_cells=round(distance, 3),
                row_delta=row_delta,
                col_delta=col_delta,
            )
        )

    return episodes


def write_setup_plan_jsonl(plan: list[SetupEpisode], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for episode in plan:
            f.write(json.dumps(asdict(episode), ensure_ascii=False, sort_keys=True) + "\n")


def load_setup_plan_jsonl(path: Path) -> list[SetupEpisode]:
    with path.open("r", encoding="utf-8") as f:
        return [SetupEpisode(**json.loads(line)) for line in f if line.strip()]


def _make_relation_sequence(num_episodes: int, rng: random.Random) -> list[str]:
    sequence: list[str] = []
    for relation, count in RELATION_QUOTAS.items():
        sequence.extend([relation] * count)

    if len(sequence) < num_episodes:
        fill_relations = tuple(RELATION_QUOTAS)
        while len(sequence) < num_episodes:
            sequence.append(rng.choice(fill_relations))
    elif len(sequence) > num_episodes:
        sequence = sequence[:num_episodes]

    rng.shuffle(sequence)
    return sequence


def _relation_candidates(relation: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for object_cell in OBJECT_CELLS:
        for container_cell in CONTAINER_CELLS:
            if object_cell == container_cell:
                continue
            if _matches_relation(object_cell, container_cell, relation):
                candidates.append((object_cell, container_cell))
    return candidates


def _matches_relation(object_cell: str, container_cell: str, relation: str) -> bool:
    distance, row_delta, col_delta = relation_between(object_cell, container_cell)
    abs_row = abs(row_delta)
    abs_col = abs(col_delta)
    is_diagonal = abs_row > 0 and abs_col > 0
    is_side_adjacent = abs_row + abs_col == 1

    if relation == "canh_sat":
        return is_side_adjacent
    if relation == "sat":
        return 0.0 < distance <= 1.45 and not is_side_adjacent
    if relation == "gan":
        return 1.45 < distance <= 2.4
    if relation == "vua":
        return 2.4 < distance <= 4.2
    if relation == "xa":
        return distance > 4.2
    if relation == "trai":
        return col_delta <= -2 and abs_row <= 1
    if relation == "phai":
        return col_delta >= 2 and abs_row <= 1
    if relation == "tren":
        return row_delta <= -1 and abs_col <= 1
    if relation == "duoi":
        return row_delta >= 1 and abs_col <= 1
    if relation == "cheo_gan":
        return is_diagonal and distance <= 2.4
    if relation == "cheo_vua":
        return is_diagonal and 2.4 < distance <= 4.2
    if relation == "cheo_xa":
        return is_diagonal and distance > 4.2
    if relation == "random":
        return True

    raise ValueError(f"Unsupported relation: {relation}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a real dataset object/container setup plan.")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--seed", type=int, default=2212)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/real_setup_plan_reachable_5x9_200_canh_sat.jsonl"),
    )
    args = parser.parse_args()

    plan = generate_setup_plan(num_episodes=args.num_episodes, seed=args.seed)
    write_setup_plan_jsonl(plan, args.output)
    print(f"Wrote {len(plan)} setup episodes to {args.output}")
    print(f"Object cells: {len(OBJECT_CELLS)} allowed")
    print(f"Container cells: {len(CONTAINER_CELLS)} allowed")


if __name__ == "__main__":
    main()
