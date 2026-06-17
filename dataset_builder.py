# -----------------------------------------------------------------------------
#  Copyright (c) Bentley Systems, Incorporated. All rights reserved.
#  See COPYRIGHT.md in the repository root for full copyright notice.
# -----------------------------------------------------------------------------

"""
dataset_builder.py
------------------------------
Self-contained script that builds a polygon dataset from the SketchGraphs .npy
archive and writes it to a local ``sketch_graphs_dataset/`` folder.

Designed to run from any location without any project-local dependencies.
All required logic (geometry helpers, data models, binary parsing) is inlined.

Usage
-----
    pip install numpy scipy pydantic lz4 tqdm matplotlib
    python dataset_builder_standalone.py

Outputs (created next to the script's working directory):
    sketch_graphs_dataset/
        sg_all.npy                      raw dataset (auto-downloaded)
        sketch_polygons_annotated.json  filtered polygon annotations
        sketch_polygons_preview.png     polygon grid + score distribution

Pipeline
--------
1.  Download ``sg_all.npy`` if absent.
2.  Parse the nested binary format without the SketchGraphs library.
3.  Shuffle sketch indices for reproducibility.
4.  Per sketch: extract line/arc segments → reconstruct closed polygon loop →
    reject self-intersecting, convex, out-of-score-range, and undersized shapes.
5.  Collect until TARGET_COUNT valid polygons are found, then save.
"""

import enum
import hashlib
import io
import json
import math
import pickle
import random
import urllib.request

from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import lz4.frame
import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon as MplPolygon
from pydantic import BaseModel, Field, field_validator
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_URL = "https://sketchgraphs.cs.princeton.edu/sequence/sg_all.npy"

# Output folder created next to wherever this script is executed from.
OUTPUT_DIR = Path.cwd() / "sketch_graphs_dataset"

NPY_FILE = OUTPUT_DIR / "sg_all.npy"
OUTPUT_JSON = OUTPUT_DIR / "sketch_polygons_annotated.json"
VIZ_PATH = OUTPUT_DIR / "sketch_polygons_preview.png"

TARGET_COUNT = 10_000

# Convex-hull defect ratio filter: [0 = convex, higher = more concave].
CONCAVITY_MIN = 0.15
CONCAVITY_MAX = 0.35

# Area filter: polygon must cover at least this fraction of the [-1, 1]² box.
MIN_AREA_FRACTION = 0.20
MIN_AREA = 4.0 * MIN_AREA_FRACTION

# Arc endpoints are snapped to this many decimal places before topology matching.
SNAP_DECIMALS = 4

SEED = 1200
SOURCE_NAME = "SketchGraphs"


# ---------------------------------------------------------------------------
# SketchGraphs binary-format constants
# ---------------------------------------------------------------------------

# Magic numbers from sketchgraphs/data/flat_array.py
_DICT_MAGIC = 87374267
_SEQ_MAGIC = -2356038617
_I64 = np.dtype("<i8")


# ---------------------------------------------------------------------------
# SketchGraphs enum / struct mirrors
#
# These replicate the integer values of the SketchGraphs library so that
# pickled sketch objects can be reconstructed without having the library
# installed.  Values must match sketchgraphs.data._entity and ._constraint.
# ---------------------------------------------------------------------------


class EntityType(enum.IntEnum):
    Point = 0
    Line = 1
    Circle = 2
    Ellipse = 3
    Spline = 4
    Conic = 5
    Arc = 6
    External = 7
    Stop = 8
    Unknown = 9


class SubnodeType(enum.IntEnum):
    SN_Start = 101
    SN_End = 102
    SN_Center = 103


class NodeOp(NamedTuple):
    label: EntityType
    parameters: dict = {}


class ConstraintType(enum.IntEnum):
    Coincident = 0
    Projected = 1
    Mirror = 2
    Distance = 3
    Horizontal = 4
    Parallel = 5
    Vertical = 6
    Tangent = 7
    Length = 8
    Perpendicular = 9
    Midpoint = 10
    Equal = 11
    Diameter = 12
    Offset = 13
    Radius = 14
    Concentric = 15
    Fix = 16
    Angle = 17
    Circular_Pattern = 18
    Pierce = 19
    Linear_Pattern = 20
    Centerline_Dimension = 21
    Intersected = 22
    Silhoutted = 23
    Quadrant = 24
    Normal = 25
    Minor_Diameter = 26
    Major_Diameter = 27
    Rho = 28
    Unknown = 29
    Subnode = 101


class EdgeOp(NamedTuple):
    label: ConstraintType
    references: tuple = ()
    parameters: dict = {}


class _Stub:
    """Absorbs any SketchGraphs class we don't need (constraint parameters, etc.)."""

    def __init__(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Custom unpickler
# ---------------------------------------------------------------------------


class _SketchGraphsUnpickler(pickle.Unpickler):
    """Remaps SketchGraphs module paths to local mirrors during unpickling."""

    _MAP: dict = {
        ("sketchgraphs.data._entity", "EntityType"): EntityType,
        ("sketchgraphs.data._entity", "SubnodeType"): SubnodeType,
        ("sketchgraphs.data.sequence", "NodeOp"): NodeOp,
        ("sketchgraphs.data.sequence", "EdgeOp"): EdgeOp,
        ("sketchgraphs.data._constraint", "ConstraintType"): ConstraintType,
        ("sketchgraphs.data._constraint", "LocalReferenceParameter"): _Stub,
        ("sketchgraphs.data._constraint", "ExternalReferenceParameter"): _Stub,
        ("sketchgraphs.data._constraint", "QuantityParameter"): _Stub,
        ("sketchgraphs.data._constraint", "EnumParameter"): _Stub,
        ("sketchgraphs.data._constraint", "BooleanParameter"): _Stub,
    }

    def find_class(self, module: str, name: str):
        key = (module, name)
        if key in self._MAP:
            return self._MAP[key]
        if module.startswith("sketchgraphs."):
            return _Stub
        return super().find_class(module, name)


# ---------------------------------------------------------------------------
# Binary parser for the SketchGraphs .npy flat-dictionary format
#
# sg_all.npy has a two-level nested structure:
#   Outer: dictionary-flat  (magic = 87374267, version 1)
#   Inner: flat-array       (magic = -2356038617, version 2)
#   Each element: lz4.frame.decompress → pickle.load → sketch sequence
# ---------------------------------------------------------------------------


def _ri64(data: np.ndarray, off: int) -> tuple[int, int]:
    return int(np.frombuffer(data[off : off + 8], dtype=_I64).item()), off + 8


def _ri64s(data: np.ndarray, off: int, n: int) -> tuple[np.ndarray, int]:
    nbytes = 8 * n
    return np.frombuffer(data[off : off + nbytes], dtype=_I64).copy(), off + nbytes


class _FlatSeqView:
    """Memory-efficient lazy view into the SketchGraphs sequence data.

    Decompresses and unpickles each sketch on demand so that the full dataset
    never needs to be materialised in memory at once.
    """

    __slots__ = ("_data", "_base", "_offsets")

    def __init__(self, data: np.ndarray, base_offset: int, offsets: np.ndarray) -> None:
        self._data = data
        self._base = base_offset
        self._offsets = offsets

    def __len__(self) -> int:
        return len(self._offsets) - 1

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        start = int(self._offsets[idx])
        end = int(self._offsets[idx + 1])
        raw = lz4.frame.decompress(bytes(self._data[self._base + start : self._base + end]))
        return _SketchGraphsUnpickler(io.BytesIO(raw)).load()


def _load_sequences(path: Path) -> _FlatSeqView:
    """Parse the SketchGraphs .npy binary and return a lazy sequence view."""
    data: np.ndarray = np.load(path, mmap_mode="r", allow_pickle=False)

    off = 0
    magic, off = _ri64(data, off)
    if magic != _DICT_MAGIC:
        raise ValueError(f"Unexpected dictionary magic {magic}")
    version, off = _ri64(data, off)
    if version != 1:
        raise ValueError(f"Unexpected dictionary version {version}")
    header_len, off = _ri64(data, off)
    header_dict: dict = pickle.loads(data[off : off + header_len].tobytes())
    base_offset = off + int(header_len)

    seq_offset, seq_len, seq_type = header_dict["sequences"]
    if seq_type != 2:
        raise ValueError(f"Expected data_type=2 for sequences, got {seq_type}")
    seq_start = base_offset + int(seq_offset)
    seq_data = data[seq_start : seq_start + int(seq_len)]

    ioff = 0
    magic2, ioff = _ri64(seq_data, ioff)
    if magic2 != _SEQ_MAGIC:
        raise ValueError(f"Unexpected flat-array magic {magic2}")
    version2, ioff = _ri64(seq_data, ioff)
    if version2 != 2:
        raise ValueError(f"Unexpected flat-array version {version2}")
    num_items, ioff = _ri64(seq_data, ioff)
    offsets, ioff = _ri64s(seq_data, ioff, num_items + 1)

    return _FlatSeqView(seq_data, ioff, offsets)


# ---------------------------------------------------------------------------
# Data models (minimal inline equivalents of utils/polygonStorage.py)
# ---------------------------------------------------------------------------


class _PolygonAnnotated(BaseModel):
    id: str
    coordinates: list[list[float]]
    source: str
    concavity_score: float = Field(..., ge=0.0, le=1.0)

    @field_validator("coordinates")
    @classmethod
    def _validate(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) < 3:
            raise ValueError("Polygon must have at least 3 vertices.")
        if v[0] != v[-1]:
            raise ValueError("Polygon is not closed (first != last point).")
        return [[round(x, 6) for x in pt] for pt in v]


class _PolygonJsonAnnotated(BaseModel):
    polygons: dict[str, _PolygonAnnotated]
    date_annotated: str
    description: str | None = None

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=4), encoding="utf-8")


# ---------------------------------------------------------------------------
# Geometry helpers (inlined from utils/geometry.py)
# ---------------------------------------------------------------------------


def _polygon_area(vertices: np.ndarray) -> float:
    """Shoelace formula — works on open or closed vertex arrays."""
    n = len(vertices)
    acc = 0.0
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        acc += x0 * y1 - x1 * y0
    return abs(acc) / 2.0


def _convex_hull_defect_ratio(vertices: np.ndarray) -> float:
    """(hull_area - poly_area) / hull_area — higher means more concave."""
    if len(vertices) < 3:
        return 0.0
    hull_area = ConvexHull(vertices).volume
    return 0.0 if hull_area == 0.0 else (hull_area - _polygon_area(vertices)) / hull_area


def _is_concave(coords: list[list[float]]) -> bool:
    """True when the polygon has at least one reflex angle (cross-product sign flip)."""
    pts = coords[:-1] if coords[0] == coords[-1] else coords
    n = len(pts)
    if n < 4:
        return False
    sign = None
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        cx, cy = pts[(i + 2) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        if cross != 0:
            s = cross > 0
            if sign is None:
                sign = s
            elif sign != s:
                return True
    return False


def _is_self_intersecting(coords: list[list[float]]) -> bool:
    """True if any two non-adjacent edges of the polygon properly cross.

    Uses an O(n²) test; acceptable for the small polygons in this dataset
    (typically 4–60 vertices).
    """
    verts = coords[:-1]
    n = len(verts)
    if n < 4:
        return False

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _proper(p1, p2, p3, p4) -> bool:
        d1, d2 = _cross(p3, p4, p1), _cross(p3, p4, p2)
        d3, d4 = _cross(p1, p2, p3), _cross(p1, p2, p4)
        return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))

    for i in range(n):
        p1, p2 = verts[i], verts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _proper(p1, p2, verts[j], verts[(j + 1) % n]):
                return True
    return False


# ---------------------------------------------------------------------------
# Sketch extraction helpers
# ---------------------------------------------------------------------------


def _snap(x: float, y: float) -> tuple[float, float]:
    return round(x, SNAP_DECIMALS), round(y, SNAP_DECIMALS)


def _extract_segments(sketch, arc_samples: int = 8) -> list[dict]:
    """Extract Line and Arc entities as topology-keyed segments.

    Each segment dict has:
        start / end  : snapped (x, y) tuples used for loop reconstruction
        points       : dense [x, y] list (arcs are sampled into polyline points)
        is_arc       : bool
    """
    segments = []
    for op in sketch:
        if not isinstance(op, NodeOp):
            continue
        p = op.parameters

        if op.label == EntityType.Line:
            try:
                x0, y0 = float(p["pntX"]), float(p["pntY"])
                dx, dy = float(p["dirX"]), float(p["dirY"])
                t1, t2 = float(p["startParam"]), float(p["endParam"])
                start = _snap(x0 + dx * t1, y0 + dy * t1)
                end = _snap(x0 + dx * t2, y0 + dy * t2)
                if start != end:
                    segments.append({"start": start, "end": end, "points": [list(start), list(end)], "is_arc": False})
            except (KeyError, TypeError, ValueError):
                continue

        elif op.label == EntityType.Arc:
            try:
                cx, cy = float(p["xCenter"]), float(p["yCenter"])
                radius = float(p["radius"])
                base_ang = math.atan2(float(p["yDir"]), float(p["xDir"]))
                t0, t1 = float(p["startParam"]), float(p["endParam"])
                clockwise = bool(p["clockwise"])
                sa = base_ang - t0 if clockwise else base_ang + t0
                ea = base_ang - t1 if clockwise else base_ang + t1
                start = _snap(cx + radius * math.cos(sa), cy + radius * math.sin(sa))
                end = _snap(cx + radius * math.cos(ea), cy + radius * math.sin(ea))
                if start != end:
                    angles = np.linspace(sa, ea, arc_samples + 2)
                    pts = [[cx + radius * math.cos(a), cy + radius * math.sin(a)] for a in angles]
                    pts[0], pts[-1] = list(start), list(end)
                    segments.append({"start": start, "end": end, "points": pts, "is_arc": True})
            except (KeyError, TypeError, ValueError):
                continue

    return segments


def _reconstruct_polygon(segments: list[dict]) -> tuple[list[list[float]], bool] | None:
    """Walk the segment graph to reconstruct a single closed polygon loop.

    Returns (coords, has_arcs) where coords is a closed [[x, y], …] list,
    or None if the segments do not form exactly one simple cycle.
    """
    if len(segments) < 3:
        return None

    adj: dict = {}
    for idx, seg in enumerate(segments):
        for pt, nb in ((seg["start"], seg["end"]), (seg["end"], seg["start"])):
            adj.setdefault(pt, []).append((nb, idx))

    if any(len(v) != 2 for v in adj.values()):
        return None

    start_pt = segments[0]["start"]
    walk: list = []
    prev_pt = None
    cur_pt = start_pt

    while True:
        (nb0, i0), (nb1, i1) = adj[cur_pt]
        next_pt, seg_idx = (nb1, i1) if nb0 == prev_pt else (nb0, i0)
        walk.append((seg_idx, segments[seg_idx]["start"] == next_pt))
        prev_pt, cur_pt = cur_pt, next_pt
        if cur_pt == start_pt:
            break
        if len(walk) > len(segments):
            return None

    if len(walk) != len(segments):
        return None

    has_arcs = False
    coords: list = []
    for seg_idx, rev in walk:
        seg = segments[seg_idx]
        if seg["is_arc"]:
            has_arcs = True
        pts = seg["points"] if not rev else list(reversed(seg["points"]))
        coords.extend(pts[:-1])

    coords.append(coords[0])
    return coords, has_arcs


def _normalize(coords: list[list[float]]) -> list[list[float]]:
    """Scale and centre coordinates into [-1, 1] with the bounding box centred at origin."""
    xs, ys = [p[0] for p in coords], [p[1] for p in coords]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half_span = max((max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2) or 1.0
    return [[(x - cx) / half_span, (y - cy) / half_span] for x, y in coords]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a progress bar, using an atomic rename."""
    import sys

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    print(f"Downloading {url}")
    chunk = 1 << 20  # 1 MiB
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        total = int(resp.headers.get("Content-Length", 0)) or None
        with (
            open(tmp, "wb") as fh,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
                dynamic_ncols=True,  # adapts to terminal width — prevents line wrapping
                mininterval=0.5,  # refresh at most every 0.5 s → single in-place line
                file=sys.stderr,  # stderr is unbuffered; avoids stdout interleaving
            ) as bar,
        ):
            while buf := resp.read(chunk):
                fh.write(buf)
                bar.update(len(buf))
    tmp.rename(dest)
    print(f"Saved → {dest}")


# ---------------------------------------------------------------------------
# Visualization (inlined from utils/benchmark.py)
# ---------------------------------------------------------------------------


def _normalize_for_plot(pts: list) -> list:
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1
    return [((x - min(xs)) / span, (y - min(ys)) / span) for x, y in pts]


def _visualize(
    polygons: list[_PolygonAnnotated],
    output_path: Path,
    concavity_min: float,
    concavity_max: float,
    rng: random.Random,
) -> None:
    """Save a PNG with a 10×10 polygon grid and a concavity score distribution."""
    sample = rng.sample(polygons, min(100, len(polygons)))
    scores = [p.concavity_score for p in polygons]
    fp_hash = hashlib.md5(b"".join(p.id.encode() for p in sorted(polygons, key=lambda x: x.id))).hexdigest()[:8]

    fig = plt.figure(figsize=(20, 26))
    gs = GridSpec(12, 10, figure=fig, hspace=0.5, wspace=0.3)

    for idx, poly in enumerate(sample):
        row, col = divmod(idx, 10)
        ax = fig.add_subplot(gs[row, col])
        pts = _normalize_for_plot(poly.coordinates)
        ax.add_patch(MplPolygon(pts, closed=True, facecolor="steelblue", edgecolor="black", linewidth=0.5, alpha=0.75))
        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xticks=[], yticks=[])
        ax.set_aspect("equal")
        ax.set_title(f"{poly.concavity_score:.3f}", fontsize=6, pad=1)

    ax_d = fig.add_subplot(gs[10:, :])
    ax_d.hist(scores, bins=50, density=True, alpha=0.4, color="steelblue", label="histogram")
    if len(scores) > 1:
        x_r = np.linspace(concavity_min, concavity_max, 300)
        ax_d.plot(x_r, gaussian_kde(scores)(x_r), color="steelblue", linewidth=2, label="KDE")
    ax_d.axvline(concavity_min, color="red", linestyle="--", linewidth=1, label=f"min = {concavity_min}")
    ax_d.axvline(concavity_max, color="orange", linestyle="--", linewidth=1, label=f"max = {concavity_max}")
    ax_d.set(
        xlabel="Concavity score (convex-hull defect ratio)",
        ylabel="Density",
        title=f"Concavity distribution — {len(polygons)} polygons",
    )
    ax_d.legend()

    fig.suptitle(f"SketchGraphs · {len(polygons)} polygons · concavity [{concavity_min}, {concavity_max}]", fontsize=13)
    fig.text(
        0.20,
        0.95,
        f"Hash: {fp_hash}",
        fontsize=20,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.3"),
    )

    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not NPY_FILE.exists():
        _download(DATASET_URL, NPY_FILE)

    print(f"Loading {NPY_FILE} ...")
    sequences = _load_sequences(NPY_FILE)
    print(f"Loaded {len(sequences)} sketches")

    rng = random.Random(SEED)
    indices = np.random.default_rng(SEED).permutation(len(sequences))

    lines_pool: dict[str, _PolygonAnnotated] = {}
    curves_pool: dict[str, _PolygonAnnotated] = {}

    pbar = tqdm(total=TARGET_COUNT, desc="Collecting polygons", unit="poly")
    scanned = 0

    for i in indices:
        if len(lines_pool) + len(curves_pool) >= TARGET_COUNT:
            break

        scanned += 1
        lines_needed = TARGET_COUNT - len(lines_pool) - len(curves_pool)

        segments = _extract_segments(sequences[i])
        if not segments:
            continue

        result = _reconstruct_polygon(segments)
        if result is None:
            continue

        coords, has_arcs = result

        if _is_self_intersecting(coords):
            continue
        if not _is_concave(coords):
            continue

        norm = _normalize(coords)

        try:
            score = float(_convex_hull_defect_ratio(np.array(norm)))
            score = max(0.0, min(1.0, score))
        except Exception:
            continue

        if not (CONCAVITY_MIN <= score <= CONCAVITY_MAX):
            continue
        if _polygon_area(np.array(norm)) < MIN_AREA:
            continue

        poly_id = f"sg_{'curves' if has_arcs else 'lines'}_{i:07d}"

        try:
            poly = _PolygonAnnotated(
                id=poly_id,
                coordinates=norm,
                source=SOURCE_NAME,
                concavity_score=round(score, 6),
            )
        except Exception:
            continue

        if has_arcs:
            curves_pool[poly_id] = poly
            pbar.update(1)
        elif lines_needed > 0:
            lines_pool[poly_id] = poly
            pbar.update(1)

    pbar.close()

    polygons = {**curves_pool, **lines_pool}
    curves_count = len(curves_pool)
    curve_pct = 100.0 * curves_count / len(polygons) if polygons else 0.0

    print(f"\nScanned {scanned} sketches, collected {len(polygons)} polygons")
    print(f"  With curves : {curves_count} ({curve_pct:.1f}%)")
    print(f"  Lines-only  : {len(polygons) - curves_count} ({100 - curve_pct:.1f}%)")

    output = _PolygonJsonAnnotated(
        polygons=polygons,
        date_annotated=datetime.now().isoformat(),
        description=(
            f"SketchGraphs subset: {len(polygons)} concave closed-loop polygons "
            f"({curves_count} with curves / {len(polygons) - curves_count} lines-only). "
            f"Concavity range [{CONCAVITY_MIN}, {CONCAVITY_MAX}]. "
            f"Coordinates normalised to [-1, 1]. "
            f"Generated by dataset_builder_standalone.py."
        ),
    )
    output.to_json(OUTPUT_JSON)
    print(f"Saved → {OUTPUT_JSON}")

    if polygons:
        _visualize(list(polygons.values()), VIZ_PATH, CONCAVITY_MIN, CONCAVITY_MAX, rng)


if __name__ == "__main__":
    main()
