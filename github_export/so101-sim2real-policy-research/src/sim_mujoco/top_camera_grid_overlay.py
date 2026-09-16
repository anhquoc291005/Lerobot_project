from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_GRID_ROWS = 5
DEFAULT_GRID_COLS = 9
DEFAULT_MARGIN_LEFT = 16
DEFAULT_MARGIN_TOP = 32
DEFAULT_MARGIN_RIGHT = 16
DEFAULT_MARGIN_BOTTOM = 0


@dataclass(frozen=True)
class ImageGridCell:
    label: str
    row: int
    col: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2)


@dataclass(frozen=True)
class ImageGridBounds:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


def make_image_grid(
    width: int,
    height: int,
    rows: int = DEFAULT_GRID_ROWS,
    cols: int = DEFAULT_GRID_COLS,
    margin_left: int = DEFAULT_MARGIN_LEFT,
    margin_top: int = DEFAULT_MARGIN_TOP,
    margin_right: int = DEFAULT_MARGIN_RIGHT,
    margin_bottom: int = DEFAULT_MARGIN_BOTTOM,
) -> dict[str, ImageGridCell]:
    """Return labeled image-space grid cells with exclusive max bounds."""
    _validate_grid(rows, cols)
    bounds = _make_grid_bounds(
        width=width,
        height=height,
        margin_left=margin_left,
        margin_top=margin_top,
        margin_right=margin_right,
        margin_bottom=margin_bottom,
    )

    cells: dict[str, ImageGridCell] = {}
    for row in range(rows):
        y_min = bounds.y_min + round(row * bounds.height / rows)
        y_max = bounds.y_min + round((row + 1) * bounds.height / rows)
        for col in range(cols):
            x_min = bounds.x_min + round(col * bounds.width / cols)
            x_max = bounds.x_min + round((col + 1) * bounds.width / cols)
            label = top_grid_label(row, col)
            cells[label] = ImageGridCell(
                label=label,
                row=row,
                col=col,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )
    return cells


def draw_top_camera_grid(
    cv2: Any,
    image: Any,
    rows: int = DEFAULT_GRID_ROWS,
    cols: int = DEFAULT_GRID_COLS,
    labels: bool = True,
    margin_left: int = DEFAULT_MARGIN_LEFT,
    margin_top: int = DEFAULT_MARGIN_TOP,
    margin_right: int = DEFAULT_MARGIN_RIGHT,
    margin_bottom: int = DEFAULT_MARGIN_BOTTOM,
) -> None:
    """Draw a 5x9-style grid in-place on an OpenCV BGR image."""
    _validate_grid(rows, cols)
    h, w = image.shape[:2]
    bounds = _make_grid_bounds(
        width=w,
        height=h,
        margin_left=margin_left,
        margin_top=margin_top,
        margin_right=margin_right,
        margin_bottom=margin_bottom,
    )
    line_color = (0, 255, 255)
    text_color = (0, 255, 255)
    shadow_color = (0, 0, 0)
    thickness = max(1, round(min(h, w) / 360))

    cv2.rectangle(
        image,
        (bounds.x_min, bounds.y_min),
        (bounds.x_max - 1, bounds.y_max - 1),
        line_color,
        thickness,
        cv2.LINE_AA,
    )
    for col in range(1, cols):
        x = bounds.x_min + round(col * bounds.width / cols)
        cv2.line(image, (x, bounds.y_min), (x, bounds.y_max - 1), line_color, thickness, cv2.LINE_AA)
    for row in range(1, rows):
        y = bounds.y_min + round(row * bounds.height / rows)
        cv2.line(image, (bounds.x_min, y), (bounds.x_max - 1, y), line_color, thickness, cv2.LINE_AA)

    if not labels:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(w / 640.0, h / 480.0) * 0.55)
    text_thickness = max(1, round(thickness))
    for cell in make_image_grid(
        w,
        h,
        rows=rows,
        cols=cols,
        margin_left=margin_left,
        margin_top=margin_top,
        margin_right=margin_right,
        margin_bottom=margin_bottom,
    ).values():
        label = cell.label
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        cx, cy = cell.center
        origin = (cx - tw // 2, cy + (th - baseline) // 2)
        cv2.putText(
            image,
            label,
            origin,
            font,
            font_scale,
            shadow_color,
            text_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            origin,
            font,
            font_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )


def top_grid_label(row: int, col: int) -> str:
    row_name = chr(ord("a") + row) if row < 26 else f"r{row + 1}"
    return f"{row_name}{col + 1}"


def _validate_grid(rows: int, cols: int) -> None:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be > 0.")


def _make_grid_bounds(
    width: int,
    height: int,
    margin_left: int = 0,
    margin_top: int = 0,
    margin_right: int = 0,
    margin_bottom: int = 0,
) -> ImageGridBounds:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0.")
    if min(margin_left, margin_top, margin_right, margin_bottom) < 0:
        raise ValueError("grid margins must be >= 0.")

    x_min = int(margin_left)
    y_min = int(margin_top)
    x_max = width - int(margin_right)
    y_max = height - int(margin_bottom)
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("grid margins leave no drawable area.")

    return ImageGridBounds(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)


def _parse_camera(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _preview_image(cv2: Any, args: argparse.Namespace) -> None:
    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    image = _resize_for_preview(cv2, image, width=args.width, height=args.height)
    draw_top_camera_grid(
        cv2,
        image,
        rows=args.rows,
        cols=args.cols,
        labels=not args.no_labels,
        margin_left=args.margin_left,
        margin_top=args.margin_top,
        margin_right=args.margin_right,
        margin_bottom=args.margin_bottom,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), image)
    cv2.imshow(args.window_name, image)
    cv2.waitKey(0)


def _preview_camera(cv2: Any, args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(_parse_camera(args.camera))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {args.camera}")

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera frame read failed.")

            frame = _resize_for_preview(cv2, frame, width=args.width, height=args.height)
            draw_top_camera_grid(
                cv2,
                frame,
                rows=args.rows,
                cols=args.cols,
                labels=not args.no_labels,
                margin_left=args.margin_left,
                margin_top=args.margin_top,
                margin_right=args.margin_right,
                margin_bottom=args.margin_bottom,
            )
            cv2.imshow(args.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyWindow(args.window_name)


def _resize_for_preview(cv2: Any, image: Any, width: int | None, height: int | None) -> Any:
    if not width or not height:
        return image

    h, w = image.shape[:2]
    if w == width and h == height:
        return image

    interpolation = cv2.INTER_AREA if width < w or height < h else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a 5x9 grid over the top camera.")
    parser.add_argument("--camera", default="0", help="OpenCV camera index or video path.")
    parser.add_argument("--image", type=Path, help="Optional still image to preview instead of live camera.")
    parser.add_argument("--output", type=Path, help="Optional output path for image preview.")
    parser.add_argument("--rows", type=int, default=DEFAULT_GRID_ROWS)
    parser.add_argument("--cols", type=int, default=DEFAULT_GRID_COLS)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--margin-left", type=int, default=DEFAULT_MARGIN_LEFT)
    parser.add_argument("--margin-top", type=int, default=DEFAULT_MARGIN_TOP)
    parser.add_argument("--margin-right", type=int, default=DEFAULT_MARGIN_RIGHT)
    parser.add_argument("--margin-bottom", type=int, default=DEFAULT_MARGIN_BOTTOM)
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--window-name", default="top_camera_grid_5x9")
    args = parser.parse_args()

    import cv2

    if args.image:
        _preview_image(cv2, args)
    else:
        _preview_camera(cv2, args)


if __name__ == "__main__":
    main()
