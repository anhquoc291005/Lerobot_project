#!/usr/bin/env python3
"""Preview or adjust a camera-space workspace polygon without connecting any robot."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.utils.workspace_boundary import WorkspaceBoundaryOverlay


def draw_boundary(frame, points, color, *, controls=False):
    preview = frame.copy()
    if len(points) >= 3:
        cv2.polylines(preview, [np.rint(points).astype(np.int32)], True, color, 2, cv2.LINE_AA)
    for index, point in enumerate(points if controls else []):
        position = tuple(np.rint(point).astype(int))
        cv2.circle(preview, position, 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            preview, str(index + 1), (position[0] + 7, position[1] + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    if controls and len(points) >= 3:
        x, y = np.min(points, axis=0).astype(int)
        cv2.putText(
            preview, "WORKSPACE", (x + 10, max(y - 12, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    if controls:
        cv2.rectangle(preview, (0, frame.shape[0] - 28), (frame.shape[1], frame.shape[0]), (25, 25, 25), -1)
        cv2.putText(
            preview, "Drag / click | R: redraw | U: undo | S: save | Q: exit",
            (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="/dev/video4")
    parser.add_argument("--name", default="top", help="Camera key in --robot.cameras")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("workspace_boundary.json"))
    parser.add_argument("--image", type=Path, help="Use a saved image instead of opening a camera")
    parser.add_argument("--export", type=Path, help="Save a preview PNG and exit without opening a window")
    args = parser.parse_args()
    overlay = WorkspaceBoundaryOverlay(args.config)
    if args.name not in overlay.boundaries:
        parser.error(f"Camera '{args.name}' is not in {args.config}")
    boundary = overlay.boundaries[args.name]
    color = tuple(int(value) for value in boundary["color"][::-1])
    cap = None
    try:
        if args.image:
            frame = cv2.imread(str(args.image))
        else:
            cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            frame = None
            for _ in range(8):
                ok, current = cap.read()
                if ok:
                    frame = current
        if frame is None:
            raise RuntimeError("Cannot read the image. Close teleop/record before opening the same camera.")
        height, width = frame.shape[:2]
        points = overlay.points_for_image(args.name, width, height).tolist()
        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.export), draw_boundary(frame, points, color)):
                raise RuntimeError(f"Could not save {args.export}")
            print(f"Preview saved: {args.export}")
            return

        window = f"Workspace - {args.name}"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        dragging = None

        def on_mouse(event, x, y, flags, userdata):
            nonlocal dragging
            point = [float(np.clip(x, 0, width - 1)), float(np.clip(y, 0, height - 1))]
            if event == cv2.EVENT_LBUTTONDOWN:
                distances = np.linalg.norm(np.asarray(points) - point, axis=1) if points else []
                if len(distances) and min(distances) <= 15:
                    dragging = int(np.argmin(distances))
                else:
                    points.append(point)
                    dragging = len(points) - 1
            elif event == cv2.EVENT_MOUSEMOVE and dragging is not None:
                points[dragging] = point
            elif event == cv2.EVENT_LBUTTONUP:
                dragging = None

        cv2.setMouseCallback(window, on_mouse)
        while True:
            if cap is not None:
                ok, current = cap.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames.")
                if current.shape != frame.shape:
                    raise RuntimeError("Camera resolution changed. Restart the editor.")
                frame = current
            cv2.imshow(window, draw_boundary(frame, points, color, controls=True))
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("r"):
                points.clear()
                dragging = None
            elif key in (ord("u"), 8, 127) and points:
                points.pop()
                dragging = None
            elif key == ord("s"):
                if len(points) < 3:
                    print("Choose at least 3 vertices before saving.")
                    continue
                config = json.loads(args.config.read_text())
                config["cameras"][args.name]["points"] = (
                    np.asarray(points) / [width - 1, height - 1]
                ).tolist()
                temporary = args.config.with_suffix(".tmp.json")
                try:
                    temporary.write_text(json.dumps(config, indent=2) + "\n")
                    WorkspaceBoundaryOverlay(temporary)
                    temporary.replace(args.config)
                    print(f"Saved: {args.config}. Restart teleop/record to apply.")
                except ValueError as error:
                    print(f"Cannot save: {error}")
                finally:
                    temporary.unlink(missing_ok=True)
    finally:
        if cap is not None:
            cap.release()
        if not args.export:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
