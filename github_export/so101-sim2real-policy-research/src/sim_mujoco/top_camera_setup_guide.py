from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from .real_dataset_setup_plan import (
        DEFAULT_NUM_EPISODES,
        SetupEpisode,
        generate_setup_plan,
        load_setup_plan_jsonl,
        write_setup_plan_jsonl,
    )
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
    from real_dataset_setup_plan import (
        DEFAULT_NUM_EPISODES,
        SetupEpisode,
        generate_setup_plan,
        load_setup_plan_jsonl,
        write_setup_plan_jsonl,
    )
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


OBJECT_COLOR = (0, 255, 0)
CONTAINER_COLOR = (255, 80, 0)
TEXT_COLOR = (0, 255, 255)
SHADOW_COLOR = (0, 0, 0)


def wait_for_episode_setup(
    cv2: Any,
    cap: Any,
    setup: SetupEpisode,
    total_episodes: int,
    width: int = 640,
    height: int = 480,
    margin_left: int = DEFAULT_MARGIN_LEFT,
    margin_top: int = DEFAULT_MARGIN_TOP,
    margin_right: int = DEFAULT_MARGIN_RIGHT,
    margin_bottom: int = DEFAULT_MARGIN_BOTTOM,
    window_name: str = "real_setup_top_camera",
) -> str:
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera frame read failed.")

        frame = _resize_for_preview(cv2, frame, width=width, height=height)
        draw_setup_overlay(
            cv2=cv2,
            image=frame,
            setup=setup,
            total_episodes=total_episodes,
            margin_left=margin_left,
            margin_top=margin_top,
            margin_right=margin_right,
            margin_bottom=margin_bottom,
        )
        cv2.imshow(window_name, frame)
        key = getattr(cv2, "waitKeyEx", cv2.waitKey)(1)
        event = _key_to_event(key)
        if event != "continue":
            return event


def draw_setup_overlay(
    cv2: Any,
    image: Any,
    setup: SetupEpisode,
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
    _highlight_cell(cv2, image, cells[setup.container_cell], CONTAINER_COLOR, "BOX")

    title = (
        f"Episode {setup.episode_index + 1}/{total_episodes} | "
        f"OBJECT {setup.object_cell.upper()} | BOX {setup.container_cell.upper()} | "
        f"{setup.relation}"
    )
    help_text = "ENTER/SPACE: start episode | N: skip | P: previous | Q/ESC: quit"
    _draw_text(cv2, image, title, (12, 24), 0.58, TEXT_COLOR, 2)
    _draw_text(cv2, image, help_text, (12, image.shape[0] - 12), 0.45, TEXT_COLOR, 1)


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


def _resize_for_preview(cv2: Any, image: Any, width: int | None, height: int | None) -> Any:
    if not width or not height:
        return image

    h, w = image.shape[:2]
    if w == width and h == height:
        return image

    interpolation = cv2.INTER_AREA if width < w or height < h else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _parse_camera(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _key_to_event(key: int) -> str:
    if key < 0:
        return "continue"
    if key in (13, 10, 32):
        return "start"
    if key in (ord("n"), ord("N")):
        return "skip"
    if key in (ord("p"), ord("P")):
        return "previous"
    if key in (27, ord("q"), ord("Q")):
        return "quit"
    return "continue"


def _load_or_create_plan(plan_path: Path, num_episodes: int, seed: int) -> list[SetupEpisode]:
    if plan_path.exists():
        return load_setup_plan_jsonl(plan_path)

    plan = generate_setup_plan(num_episodes=num_episodes, seed=seed)
    write_setup_plan_jsonl(plan, plan_path)
    return plan


def _append_confirm_log(path: Path, setup: SetupEpisode, event: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **asdict(setup),
        "event": event,
        "confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guide real dataset setup with top-camera grid cells.")
    parser.add_argument("--camera", default="0", help="OpenCV camera index or video path.")
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--seed", type=int, default=2212)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("datasets/real_setup_plan_reachable_5x9_200_canh_sat.jsonl"),
    )
    parser.add_argument("--confirm-log", type=Path, default=Path("datasets/real_setup_confirmed.jsonl"))
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--margin-left", type=int, default=DEFAULT_MARGIN_LEFT)
    parser.add_argument("--margin-top", type=int, default=DEFAULT_MARGIN_TOP)
    parser.add_argument("--margin-right", type=int, default=DEFAULT_MARGIN_RIGHT)
    parser.add_argument("--margin-bottom", type=int, default=DEFAULT_MARGIN_BOTTOM)
    parser.add_argument("--window-name", default="real_setup_top_camera")
    args = parser.parse_args()

    import cv2

    plan = _load_or_create_plan(args.plan, args.num_episodes, args.seed)
    if not plan:
        raise RuntimeError("Setup plan is empty.")

    cap = cv2.VideoCapture(_parse_camera(args.camera))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    episode_index = max(0, min(args.start_episode, len(plan) - 1))
    try:
        while episode_index < len(plan):
            event = wait_for_episode_setup(
                cv2=cv2,
                cap=cap,
                setup=plan[episode_index],
                total_episodes=len(plan),
                width=args.width,
                height=args.height,
                margin_left=args.margin_left,
                margin_top=args.margin_top,
                margin_right=args.margin_right,
                margin_bottom=args.margin_bottom,
                window_name=args.window_name,
            )
            if event == "quit":
                break
            if event == "previous":
                episode_index = max(0, episode_index - 1)
                continue
            if event in {"start", "skip"}:
                _append_confirm_log(args.confirm_log, plan[episode_index], event)
                print(
                    f"{event}: episode={episode_index} "
                    f"object={plan[episode_index].object_cell} "
                    f"box={plan[episode_index].container_cell} "
                    f"relation={plan[episode_index].relation}",
                    flush=True,
                )
                episode_index += 1
    finally:
        cap.release()
        cv2.destroyWindow(args.window_name)


if __name__ == "__main__":
    main()
