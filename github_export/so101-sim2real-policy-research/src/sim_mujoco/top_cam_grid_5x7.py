from __future__ import annotations

from dataclasses import dataclass


Point3D = tuple[float, float, float]


@dataclass(frozen=True)
class GridRegion:
    name: str
    row: str
    col: int
    top_left: Point3D
    top_right: Point3D
    bottom_left: Point3D
    bottom_right: Point3D

    @property
    def center(self) -> Point3D:
        return _lerp_point(
            _lerp_point(self.top_left, self.top_right, 0.5),
            _lerp_point(self.bottom_left, self.bottom_right, 0.5),
            0.5,
        )

    @property
    def x_min(self) -> float:
        return min(p[0] for p in self.corners)

    @property
    def x_max(self) -> float:
        return max(p[0] for p in self.corners)

    @property
    def y_min(self) -> float:
        return min(p[1] for p in self.corners)

    @property
    def y_max(self) -> float:
        return max(p[1] for p in self.corners)

    @property
    def z(self) -> float:
        return self.center[2]

    @property
    def corners(self) -> tuple[Point3D, Point3D, Point3D, Point3D]:
        return (self.top_left, self.top_right, self.bottom_left, self.bottom_right)

    def point_at(self, row_t: float, col_t: float) -> Point3D:
        top = _lerp_point(self.top_left, self.top_right, col_t)
        bottom = _lerp_point(self.bottom_left, self.bottom_right, col_t)
        return _lerp_point(top, bottom, row_t)


# Change only these four points when you recalibrate the visible top_cam area.
# They are on the floor plane z=0 and use image-style naming:
#   TOP_LEFT -> TOP_RIGHT
#      |            |
#   BOTTOM_LEFT -> BOTTOM_RIGHT
TOP_LEFT: Point3D = (0.6, 0.38, 0.0)
TOP_RIGHT: Point3D = (0.6, -0.38, 0.0)
BOTTOM_LEFT: Point3D = (0.025, 0.38, 0.0)
BOTTOM_RIGHT: Point3D = (0.025, -0.38, 0.0)

ROWS = ("a", "b", "c", "d", "e")
COLS = (1, 2, 3, 4, 5, 6, 7)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_point(a: Point3D, b: Point3D, t: float) -> Point3D:
    return (
        _lerp(a[0], b[0], t),
        _lerp(a[1], b[1], t),
        _lerp(a[2], b[2], t),
    )


def point_at(row_t: float, col_t: float) -> Point3D:
    top = _lerp_point(TOP_LEFT, TOP_RIGHT, col_t)
    bottom = _lerp_point(BOTTOM_LEFT, BOTTOM_RIGHT, col_t)
    return _lerp_point(top, bottom, row_t)


def make_region(row_index: int, col_index: int) -> GridRegion:
    row = ROWS[row_index]
    col = COLS[col_index]
    row_top = row_index / len(ROWS)
    row_bottom = (row_index + 1) / len(ROWS)
    col_left = col_index / len(COLS)
    col_right = (col_index + 1) / len(COLS)
    return GridRegion(
        name=f"{row}{col}",
        row=row,
        col=col,
        top_left=point_at(row_top, col_left),
        top_right=point_at(row_top, col_right),
        bottom_left=point_at(row_bottom, col_left),
        bottom_right=point_at(row_bottom, col_right),
    )


GRID_MATRIX = tuple(
    tuple(make_region(row_index, col_index) for col_index in range(len(COLS)))
    for row_index in range(len(ROWS))
)

GRID_CELLS = {region.name: region for row in GRID_MATRIX for region in row}

# Keep direct variable access, e.g. `from top_cam_grid_5x7 import c4`.
globals().update(GRID_CELLS)


def get_region(cell: str) -> GridRegion:
    return GRID_CELLS[cell.strip().lower()]


def get_center(cell: str) -> Point3D:
    return get_region(cell).center
