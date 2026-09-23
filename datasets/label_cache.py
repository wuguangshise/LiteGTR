"""One-time label cache for XML-annotated datasets.

Why cache rather than convert
-----------------------------
Parsing XML per sample per epoch is genuinely slow: DroneVehicle is ~28k images,
and at 300 epochs with several workers the DataLoader becomes the bottleneck.
But converting to a flat label file on disk throws information away permanently
(the rotation angle, most importantly) and creates a derived artefact that can
silently drift out of sync with the annotations.

The cache is the middle path: the XML stays the single source of truth, parsing
happens exactly once, and what is cached is the *parsed OBB*, not a reduced HBB.
Changing the OBB->HBB policy, or adding an oriented head later, is then a code
change rather than a data-regeneration job.

Layout (single .npz):
    names      (N,)    image stem
    counts     (N,)    objects per image
    polys      (M, 8)  four corners, x1 y1 x2 y2 x3 y3 x4 y4, in ORIGINAL coords
    labels     (M,)    contiguous class id
    offsets    (N, 2)  border (dx, dy) to subtract -- the white-margin crop
    sizes      (N, 2)  content (h, w) after the border crop
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CACHE_VERSION = 2


def cache_path(ann_dir: Path, tag: str) -> Path:
    key = hashlib.md5(f"{ann_dir.resolve()}|{tag}|{CACHE_VERSION}".encode()).hexdigest()[:10]
    return ann_dir.parent / f".litegtr_cache_{tag}_{key}.npz"


def save(path: Path, names, counts, polys, labels, offsets, sizes) -> None:
    np.savez_compressed(
        path,
        version=np.asarray([CACHE_VERSION]),
        names=np.asarray(names, dtype=object),
        counts=np.asarray(counts, dtype=np.int32),
        polys=np.asarray(polys, dtype=np.float32).reshape(-1, 8),
        labels=np.asarray(labels, dtype=np.int64).reshape(-1),
        offsets=np.asarray(offsets, dtype=np.int32).reshape(-1, 2),
        sizes=np.asarray(sizes, dtype=np.int32).reshape(-1, 2),
    )


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=True)
        if int(z["version"][0]) != CACHE_VERSION:
            return None
        counts = z["counts"]
        starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return {
            "names": [str(n) for n in z["names"]],
            "counts": counts,
            "starts": starts,
            "polys": z["polys"],
            "labels": z["labels"],
            "offsets": z["offsets"],
            "sizes": z["sizes"],
        }
    except Exception:
        return None


def poly_to_hbb(polys: np.ndarray) -> np.ndarray:
    """(M, 8) corners -> (M, 4) circumscribed xyxy."""
    p = np.asarray(polys, dtype=np.float32).reshape(-1, 4, 2)
    if not len(p):
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate([p.min(1), p.max(1)], axis=1)
