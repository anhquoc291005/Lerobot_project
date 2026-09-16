from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .real_dataset_setup_plan import REACHABLE_CELLS
    from .top_camera_grid_overlay import (
        DEFAULT_GRID_COLS,
        DEFAULT_GRID_ROWS,
        DEFAULT_MARGIN_BOTTOM,
        DEFAULT_MARGIN_LEFT,
        DEFAULT_MARGIN_RIGHT,
        DEFAULT_MARGIN_TOP,
        draw_top_camera_grid,
        make_image_grid,
    )
except ImportError:
    from real_dataset_setup_plan import REACHABLE_CELLS
    from top_camera_grid_overlay import (
        DEFAULT_GRID_COLS,
        DEFAULT_GRID_ROWS,
        DEFAULT_MARGIN_BOTTOM,
        DEFAULT_MARGIN_LEFT,
        DEFAULT_MARGIN_RIGHT,
        DEFAULT_MARGIN_TOP,
        draw_top_camera_grid,
        make_image_grid,
    )


DEFAULT_REPEATS_PER_CELL = 2
DEFAULT_OUTPUT = Path("datasets/rollout_object_cell_plan_5x9_36.jsonl")
OBJECT_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 255)
SHADOW_COLOR = (0, 0, 0)


@dataclass(frozen=True)
class RolloutObjectCellEpisode:
    episode_index: int
    object_cell: str
    repeat_index: int
    repeats_per_cell: int


def _row_major_cells(cells: tuple[str, ...] = REACHABLE_CELLS) -> list[str]:
    return sorted(cells, key=lambda cell: (cell[0], int(cell[1:])))


def generate_rollout_object_cell_plan(
    repeats_per_cell: int = DEFAULT_REPEATS_PER_CELL,
    cells: tuple[str, ...] = REACHABLE_CELLS,
) -> list[RolloutObjectCellEpisode]:
    if repeats_per_cell <= 0:
        raise ValueError("repeats_per_cell must be > 0.")

    plan: list[RolloutObjectCellEpisode] = []
    for object_cell in _row_major_cells(cells):
        for repeat_index in range(1, repeats_per_cell + 1):
            plan.append(
                RolloutObjectCellEpisode(
                    episode_index=len(plan),
                    object_cell=object_cell,
                    repeat_index=repeat_index,
                    repeats_per_cell=repeats_per_cell,
                )
            )
    return plan


def write_rollout_object_cell_plan_jsonl(
    plan: list[RolloutObjectCellEpisode],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for episode in plan:
            f.write(json.dumps(asdict(episode), ensure_ascii=False, sort_keys=True) + "\n")


def load_rollout_object_cell_plan_jsonl(path: Path) -> list[RolloutObjectCellEpisode]:
    with path.open("r", encoding="utf-8") as f:
        return [RolloutObjectCellEpisode(**json.loads(line)) for line in f if line.strip()]


def draw_rollout_object_cell_overlay(
    cv2: Any,
    image: Any,
    setup: RolloutObjectCellEpisode,
    total_episodes: int,
    margin_left: int = DEFAULT_MARGIN_LEFT,
    margin_top: int = DEFAULT_MARGIN_TOP,
    margin_right: int = DEFAULT_MARGIN_RIGHT,
    margin_bottom: int = DEFAULT_MARGIN_BOTTOM,
) -> None:
    draw_top_camera_grid(
        cv2=cv2,
        image=image,
        rows=DEFAULT_GRID_ROWS,
        cols=DEFAULT_GRID_COLS,
        labels=True,
        margin_left=margin_left,
        margin_top=margin_top,
        margin_right=margin_right,
        margin_bottom=margin_bottom,
    )

    cells = make_image_grid(
        image.shape[1],
        image.shape[0],
        rows=DEFAULT_GRID_ROWS,
        cols=DEFAULT_GRID_COLS,
        margin_left=margin_left,
        margin_top=margin_top,
        margin_right=margin_right,
        margin_bottom=margin_bottom,
    )
    _highlight_cell(cv2, image, cells[setup.object_cell], OBJECT_COLOR, "OBJECT")

    title = (
        f"Rollout {setup.episode_index + 1}/{total_episodes} | "
        f"OBJECT {setup.object_cell.upper()} | "
        f"trial {setup.repeat_index}/{setup.repeats_per_cell}"
    )
    help_text = "ENTER/SPACE: start rollout | N: skip | P: previous | Q/ESC: quit"
    _draw_text(cv2, image, title, (12, 24), 0.58, TEXT_COLOR, 2)
    _draw_text(cv2, image, help_text, (12, image.shape[0] - 12), 0.45, TEXT_COLOR, 1)


def draw_rollout_label_overlay(
    cv2: Any,
    image: Any,
    setup: RolloutObjectCellEpisode,
    total_episodes: int,
    margin_left: int = DEFAULT_MARGIN_LEFT,
    margin_top: int = DEFAULT_MARGIN_TOP,
    margin_right: int = DEFAULT_MARGIN_RIGHT,
    margin_bottom: int = DEFAULT_MARGIN_BOTTOM,
) -> None:
    draw_rollout_object_cell_overlay(
        cv2=cv2,
        image=image,
        setup=setup,
        total_episodes=total_episodes,
        margin_left=margin_left,
        margin_top=margin_top,
        margin_right=margin_right,
        margin_bottom=margin_bottom,
    )
    prompt = "S: success | F: failure | R: rerun | Q/ESC: quit"
    _draw_text(cv2, image, prompt, (12, 52), 0.62, TEXT_COLOR, 2)


def append_rollout_label_log(
    path: Path,
    setup: RolloutObjectCellEpisode,
    dataset_episode_index: int,
    result: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **asdict(setup),
        "dataset_episode_index": dataset_episode_index,
        "result": result,
        "confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _highlight_cell(cv2: Any, image: Any, cell: Any, color: tuple[int, int, int], title: str) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (cell.x_min, cell.y_min), (cell.x_max - 1, cell.y_max - 1), color, -1)
    cv2.addWeighted(overlay, 0.22, image, 0.78, 0, dst=image)
    cv2.rectangle(image, (cell.x_min, cell.y_min), (cell.x_max - 1, cell.y_max - 1), color, 3)

    cx, cy = cell.center
    _draw_text(cv2, image, title, (cell.x_min + 8, max(cell.y_min + 24, cy - 8)), 0.55, color, 2)
    _draw_text(cv2, image, cell.label.upper(), (cx - 18, cy + 18), 0.75, color, 2)


def _draw_text(
    cv2: Any,
    image: Any,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, origin, font, scale, SHADOW_COLOR, thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rollout object-cell setup plan.")
    parser.add_argument("--repeats-per-cell", type=int, default=DEFAULT_REPEATS_PER_CELL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    plan = generate_rollout_object_cell_plan(repeats_per_cell=args.repeats_per_cell)
    write_rollout_object_cell_plan_jsonl(plan, args.output)
    print(f"Wrote {len(plan)} rollout object-cell episodes to {args.output}")
    print(f"Object cells: {len(_row_major_cells())} allowed")
    print(f"Repeats per cell: {args.repeats_per_cell}")


if __name__ == "__main__":
    main()
