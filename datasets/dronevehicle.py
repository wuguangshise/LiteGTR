"""DroneVehicle (RGB or IR), HBB protocol, read NATIVELY from the XML.

No offline image rewrite, no derived label files
------------------------------------------------
The earlier pipeline cropped the 100-px white margin by decoding and re-encoding
every image. On a JPEG source that is a lossy round-trip, and the artefacts land
at exactly the scale a 12-px vehicle occupies -- a bad trade in a paper about
small objects. The crop is a numpy slice, so it is free and lossless at load
time; there is no reason to rewrite 28k files.

Annotations stay in their original XML. A one-time cache
(``datasets/label_cache.py``) removes the per-epoch parse cost while keeping the
*oriented* boxes, so the OBB->HBB policy is a code decision rather than baked
into generated files.

Protocol note (docs/DESIGN.md P0-5 / P1-10): published DroneVehicle results are
OBB mAP and are NOT comparable to the HBB numbers this produces. The
circumscribed box of a tilted vehicle is strictly larger than the vehicle, so
IoU is looser. State the protocol and retrain every baseline under it.

Still run ``tools/visualize_labels.py`` before training -- a border offset that
was never applied does not raise, it just trains a wrong model.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from datasets import label_cache
from datasets.base import DetectionDataset, imread

CLASSES = ["car", "truck", "bus", "van", "freight_car"]
DEFAULT_THRESHOLDS = (60.0, 110.0)   # mean V: < 60 -> dark, < 110 -> night, else day

# The released annotations use several spellings for the same class.
ALIASES = {
    "feright_car": "freight_car", "feright car": "freight_car", "freight car": "freight_car",
    "truvk": "truck", "lorry": "truck",
}

# Padded size -> white margin per side. DroneVehicle ships 840x712 with 640x512
# of content; the margin exists so rotated boxes near the edge stay drawable.
KNOWN_PADDING = {(840, 712): 100}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
# common directory spellings across redistributions
ANN_CANDIDATES = ["{m}_xml", "{m}label", "{m}labels", "{m}_label", "annotations", "labels", "xml"]


def norm_class(name: str) -> str | None:
    n = (name or "").strip().lower().replace("-", "_")
    n = ALIASES.get(n, n)
    return n if n in CLASSES else None


def parse_obj(obj: ET.Element) -> np.ndarray | None:
    """Return the object's four corners as (4, 2), whatever form it is stored in."""
    poly = obj.find("polygon")
    if poly is not None:
        try:
            pts = [(float(poly.find(f"x{i}").text), float(poly.find(f"y{i}").text))
                   for i in range(1, 5)]
            return np.asarray(pts, dtype=np.float32)
        except (AttributeError, TypeError, ValueError):
            return None
    rb = obj.find("robndbox")
    if rb is not None:
        try:
            cx, cy = float(rb.find("cx").text), float(rb.find("cy").text)
            w, h = float(rb.find("w").text), float(rb.find("h").text)
            ang = float(rb.find("angle").text)
        except (AttributeError, TypeError, ValueError):
            return None
        c, s = np.cos(ang), np.sin(ang)
        dx, dy = w / 2.0, h / 2.0
        corners = np.asarray([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float32)
        rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
        return corners @ rot.T + np.asarray([cx, cy], dtype=np.float32)
    bb = obj.find("bndbox")
    if bb is not None:
        try:
            x1, y1 = float(bb.find("xmin").text), float(bb.find("ymin").text)
            x2, y2 = float(bb.find("xmax").text), float(bb.find("ymax").text)
        except (AttributeError, TypeError, ValueError):
            return None
        return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return None


class DroneVehicleDataset(DetectionDataset):
    classes = CLASSES

    def __init__(self, root: str, split: str = "train", img_size: int = 640,
                 train: bool = True, mosaic_prob: float = 0.5, modality: str = "rgb",
                 ann_dirname: str | None = None,
                 thresholds: tuple[float, float] = DEFAULT_THRESHOLDS,
                 conditions_file: str | None = None, rebuild_cache: bool = False,
                 border: int = -1):
        super().__init__(img_size, train, mosaic_prob)
        assert modality in ("rgb", "ir"), "modality must be 'rgb' or 'ir'"
        self.root = Path(root)
        self.modality = modality
        self.thresholds = thresholds
        self.border_override = border

        base = self.root / split
        self.img_dir = self._find_img_dir(base, modality)
        self.ann_dir = self._find_ann_dir(base, modality, ann_dirname)

        self.cache = self._load_or_build_cache(split, rebuild_cache)
        self.names = self.cache["names"]
        self._img_paths = self._resolve_image_paths()

        cf = Path(conditions_file) if conditions_file else base / "conditions.txt"
        self.conditions: dict[str, str] = {}
        if cf.exists():
            for line in cf.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    self.conditions[parts[0]] = parts[1]
            print(f"[DroneVehicle] conditions from {cf} ({len(self.conditions)} entries)")
        else:
            print(f"[DroneVehicle] {cf} not found -- tagging day/night/dark by luminance "
                  f"(thresholds={thresholds}). Run datasets.prepare.make_conditions to fix "
                  f"the split explicitly.")

    # ------------------------------------------------------------ discovery
    @staticmethod
    def _find_img_dir(base: Path, modality: str) -> Path:
        for name in (modality, f"{modality}img", f"{modality}_images", "images"):
            d = base / name
            if d.is_dir():
                return d
        raise FileNotFoundError(
            f"no image directory under {base} (tried {modality}/, {modality}img/, images/)")

    @staticmethod
    def _find_ann_dir(base: Path, modality: str, explicit: str | None) -> Path:
        if explicit:
            d = base / explicit
            if not d.is_dir():
                raise FileNotFoundError(f"ann_dirname={explicit!r} not found under {base}")
            return d
        for pat in ANN_CANDIDATES:
            d = base / pat.format(m=modality)
            if d.is_dir() and any(d.glob("*.xml")):
                return d
        raise FileNotFoundError(
            f"no XML annotation directory under {base}. Set data.ann_dirname in the config.")

    def _resolve_image_paths(self) -> list[Path]:
        by_stem = {p.stem: p for p in self.img_dir.iterdir() if p.suffix.lower() in IMG_EXT}
        paths, missing = [], 0
        for n in self.names:
            p = by_stem.get(n)
            if p is None:
                missing += 1
                paths.append(None)
            else:
                paths.append(p)
        if missing:
            keep = [i for i, p in enumerate(paths) if p is not None]
            print(f"[DroneVehicle] {missing} annotated images not found in {self.img_dir} -- skipped")
            self._filter_cache(keep)
            paths = [paths[i] for i in keep]
        return paths

    def _filter_cache(self, keep: list[int]) -> None:
        c = self.cache
        starts, counts = c["starts"], c["counts"]
        obj_idx = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in keep]) \
            if keep else np.zeros(0, dtype=np.int64)
        c["names"] = [c["names"][i] for i in keep]
        c["counts"] = counts[keep]
        c["starts"] = np.concatenate([[0], np.cumsum(c["counts"])]).astype(np.int64)
        c["polys"] = c["polys"][obj_idx]
        c["labels"] = c["labels"][obj_idx]
        c["offsets"] = c["offsets"][keep]
        c["sizes"] = c["sizes"][keep]
        self.names = c["names"]

    # ---------------------------------------------------------------- cache
    def _load_or_build_cache(self, split: str, rebuild: bool) -> dict:
        path = label_cache.cache_path(self.ann_dir, f"{split}_{self.modality}")
        if not rebuild:
            cached = label_cache.load(path)
            if cached is not None:
                print(f"[DroneVehicle] label cache: {path.name} "
                      f"({len(cached['names'])} images, {len(cached['labels'])} objects)")
                return cached
        return self._build_cache(path)

    def _build_cache(self, path: Path) -> dict:
        xmls = sorted(self.ann_dir.glob("*.xml"))
        if not xmls:
            raise FileNotFoundError(f"no .xml under {self.ann_dir}")
        print(f"[DroneVehicle] building label cache from {len(xmls)} XML files "
              f"(one-time; XML stays the source of truth)")

        names, counts, polys, labels, offsets, sizes = [], [], [], [], [], []
        dropped = 0
        for xml in xmls:
            try:
                root = ET.parse(xml).getroot()
            except ET.ParseError:
                dropped += 1
                continue
            w, h = self._xml_size(root)
            b = self._border_for(w, h, xml.stem)
            cw, ch = (w - 2 * b, h - 2 * b) if (w and h) else (0, 0)

            n = 0
            for obj in root.iter("object"):
                nm = obj.find("name")
                cname = norm_class(nm.text if nm is not None else "")
                pts = parse_obj(obj)
                if cname is None or pts is None:
                    dropped += 1
                    continue
                pts = pts - b                                   # lossless border shift
                if cw and ch:
                    pts[:, 0] = pts[:, 0].clip(0, cw)
                    pts[:, 1] = pts[:, 1].clip(0, ch)
                mn, mx = pts.min(0), pts.max(0)
                if mx[0] - mn[0] < 2 or mx[1] - mn[1] < 2:
                    dropped += 1
                    continue
                polys.append(pts.reshape(8))
                labels.append(CLASSES.index(cname))
                n += 1
            names.append(xml.stem)
            counts.append(n)
            offsets.append((b, b))
            sizes.append((ch, cw))

        label_cache.save(path, names, counts, polys, labels, offsets, sizes)
        total = int(sum(counts))
        print(f"[DroneVehicle] cached {len(names)} images / {total} objects "
              f"({dropped} dropped) -> {path.name}")
        if total:
            hist = np.bincount(np.asarray(labels), minlength=len(CLASSES))
            print("[DroneVehicle] class balance (drives seed variance -- 3 seeds are mandatory):")
            for i in np.argsort(-hist):
                print(f"    {CLASSES[i]:14s} {hist[i]:8d}  {100.0 * hist[i] / total:5.1f}%")
        return label_cache.load(path)

    @staticmethod
    def _xml_size(root: ET.Element) -> tuple[int, int]:
        size = root.find("size")
        if size is None:
            return 0, 0
        try:
            return int(float(size.find("width").text)), int(float(size.find("height").text))
        except (AttributeError, TypeError, ValueError):
            return 0, 0

    def _border_for(self, w: int, h: int, stem: str) -> int:
        """Border from the declared size -- no image decode needed for the known case."""
        if self.border_override >= 0:
            return self.border_override
        b = KNOWN_PADDING.get((w, h))
        if b is not None:
            return b
        if (w, h) == (640, 512):
            return 0                      # already cropped by a previous pipeline
        p = self.img_dir / f"{stem}.jpg"
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                top = 0
                while top < img.shape[0] // 3 and img[top].mean() > 240:
                    top += 1
                return top
        return 0

    # ----------------------------------------------------------------- data
    def __len__(self) -> int:
        return len(self._img_paths)

    def _condition(self, stem: str, img: np.ndarray) -> str:
        if stem in self.conditions:
            return self.conditions[stem]
        v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2].mean()
        dark_t, night_t = self.thresholds
        return "dark" if v < dark_t else ("night" if v < night_t else "day")

    def load_raw(self, index: int):
        p = self._img_paths[index]
        img = imread(p)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        dx, dy = self.cache["offsets"][index]
        if dx or dy:
            h, w = img.shape[:2]
            img = img[dy:h - dy, dx:w - dx]          # lossless slice, no re-encode
        img = np.ascontiguousarray(img)

        s, e = self.cache["starts"][index], self.cache["starts"][index + 1]
        boxes = label_cache.poly_to_hbb(self.cache["polys"][s:e])
        labels = self.cache["labels"][s:e].copy()
        if len(boxes):
            h, w = img.shape[:2]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
            keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
            boxes, labels = boxes[keep], labels[keep]

        return img, boxes.astype(np.float32), labels, {
            "image_id": index, "file_name": p.name, "ori_shape": img.shape[:2],
            "condition": self._condition(p.stem, img),
        }

    def obb(self, index: int) -> np.ndarray:
        """Oriented boxes (n, 4, 2) for this image -- kept so an OBB head can be
        added later without regenerating any data."""
        s, e = self.cache["starts"][index], self.cache["starts"][index + 1]
        return self.cache["polys"][s:e].reshape(-1, 4, 2)
