#!/usr/bin/env python3
"""
Data augmentation for images + COCO-style polygon segmentations.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

from PIL import Image


Point = Tuple[float, float]


@dataclass(frozen=True)
class Transform:
    name: str
    func: Callable[[Point, int, int], Point]
    out_size: Callable[[int, int], Tuple[int, int]]
    sequence: Tuple[str, ...]


class CocoAugmenter:
    def __init__(
        self,
        images_dir: Path,
        json_dir: Path,
        out_dir: Path,
        log_csv: Path | None = None,
    ):
        self.images_dir = images_dir
        self.json_dir = json_dir
        self.out_dir = out_dir
        self.log_csv = log_csv or (out_dir / "augmentation_log.csv")

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with self.log_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "input_image",
                    "input_json",
                    "transform",
                    "output_image",
                    "output_json",
                    "status",
                    "message",
                ]
            )
            for image_path in self._iter_images(self.images_dir):
                json_path = self.json_dir / (image_path.stem + ".json")
                if not json_path.exists():
                    writer.writerow(
                        [
                            str(image_path),
                            str(json_path),
                            "",
                            "",
                            "",
                            "skipped",
                            "missing json",
                        ]
                    )
                    continue
                self._process_pair(image_path, json_path, writer)

    def _process_pair(self, image_path: Path, json_path: Path, writer: csv.writer) -> None:
        image = Image.open(image_path)
        w, h = image.size
        with json_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        images = coco.get("images", [])
        if len(images) != 1:
            writer.writerow(
                [
                    str(image_path),
                    str(json_path),
                    "",
                    "",
                    "",
                    "skipped",
                    f"expected 1 image entry, found {len(images)}",
                ]
            )
            print(f"[warn] {json_path} has {len(images)} image entries; skipping")
            return

        image_entry = self._find_image_entry(coco, image_path.name)
        if image_entry is None:
            writer.writerow(
                [
                    str(image_path),
                    str(json_path),
                    "",
                    "",
                    "",
                    "skipped",
                    "image entry file_name mismatch",
                ]
            )
            print(f"[warn] no matching image entry in {json_path} for {image_path.name}; skipping")
            return

        for t in self._transforms():
            try:
                out_image = self._apply_image_transform(image, t.sequence)
                out_w, out_h = out_image.size

                out_name = f"{image_path.stem}_{t.name}{image_path.suffix}"
                out_image_path = self.out_dir / out_name
                out_image.save(out_image_path)

                out_coco = json.loads(json.dumps(coco))
                out_image_entry = self._find_image_entry(out_coco, image_path.name)
                if out_image_entry is None:
                    writer.writerow(
                        [
                            str(image_path),
                            str(json_path),
                            t.name,
                            "",
                            "",
                            "skipped",
                            "image entry file_name mismatch in output",
                        ]
                    )
                    continue
                out_image_entry["file_name"] = out_name
                out_image_entry["width"] = out_w
                out_image_entry["height"] = out_h

                out_annotations = [
                    a
                    for a in out_coco.get("annotations", [])
                    if a.get("image_id") == out_image_entry.get("id")
                ]
                for ann in out_annotations:
                    if "segmentation" in ann:
                        ann["segmentation"] = self._transform_segmentation(ann["segmentation"], t, w, h)
                        bbox, area = self._bbox_area_from_segmentation(ann["segmentation"])
                        if bbox is not None:
                            ann["bbox"] = bbox
                        if area is not None:
                            ann["area"] = area

                out_json_name = f"{image_path.stem}_{t.name}.json"
                out_json_path = self.out_dir / out_json_name
                with out_json_path.open("w", encoding="utf-8") as f:
                    json.dump(out_coco, f, ensure_ascii=False)

                writer.writerow(
                    [
                        str(image_path),
                        str(json_path),
                        t.name,
                        str(out_image_path),
                        str(out_json_path),
                        "ok",
                        "",
                    ]
                )
            except Exception as e:  # pragma: no cover - logging path
                writer.writerow(
                    [
                        str(image_path),
                        str(json_path),
                        t.name,
                        "",
                        "",
                        "error",
                        str(e),
                    ]
                )

    @staticmethod
    def _iter_images(images_dir: Path) -> Iterable[Path]:
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        for path in sorted(images_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in exts:
                yield path

    @staticmethod
    def _find_image_entry(coco: dict, filename: str) -> dict | None:
        images = coco.get("images", [])
        for img in images:
            if img.get("file_name") == filename:
                return img
        return None

    @staticmethod
    def _apply_image_transform(image: Image.Image, sequence: Tuple[str, ...]) -> Image.Image:
        out = image
        for step in sequence:
            if step == "rot90":
                out = out.transpose(Image.ROTATE_270)  # 90 deg clockwise
            elif step == "rot180":
                out = out.transpose(Image.ROTATE_180)
            elif step == "rot270":
                out = out.transpose(Image.ROTATE_90)  # 270 clockwise = 90 ccw
            elif step == "flipH":
                out = out.transpose(Image.FLIP_LEFT_RIGHT)
            elif step == "flipV":
                out = out.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                raise ValueError(f"unknown transform step: {step}")
        return out

    @staticmethod
    def _transform_segmentation(segmentation: List[List[float]], t: Transform, w: int, h: int) -> List[List[float]]:
        out: List[List[float]] = []
        for poly in segmentation:
            pts = list(zip(poly[0::2], poly[1::2]))
            new_pts = [t.func((x, y), w, h) for (x, y) in pts]
            flat: List[float] = []
            for x, y in new_pts:
                flat.extend([x, y])
            out.append(flat)
        return out

    @staticmethod
    def _bbox_area_from_segmentation(segmentation: List[List[float]]) -> Tuple[List[float] | None, float | None]:
        all_pts: List[Point] = []
        total_area = 0.0
        for poly in segmentation:
            pts = list(zip(poly[0::2], poly[1::2]))
            if len(pts) >= 3:
                total_area += abs(CocoAugmenter._polygon_area(pts))
            all_pts.extend(pts)
        if not all_pts:
            return None, None
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bbox = [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)]
        return bbox, float(total_area)

    @staticmethod
    def _polygon_area(pts: List[Point]) -> float:
        area = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    @staticmethod
    def _transforms() -> List[Transform]:
        def rot90(p: Point, w: int, h: int) -> Point:
            x, y = p
            return (h - 1 - y, x)

        def rot180(p: Point, w: int, h: int) -> Point:
            x, y = p
            return (w - 1 - x, h - 1 - y)

        def rot270(p: Point, w: int, h: int) -> Point:
            x, y = p
            return (y, w - 1 - x)

        def flipH(p: Point, w: int, h: int) -> Point:
            x, y = p
            return (w - 1 - x, y)

        def flipV(p: Point, w: int, h: int) -> Point:
            x, y = p
            return (x, h - 1 - y)

        def compose(f1, f2, size1: Callable[[int, int], Tuple[int, int]]):
            def _f(p: Point, w: int, h: int) -> Point:
                p1 = f1(p, w, h)
                w1, h1 = size1(w, h)
                return f2(p1, w1, h1)
            return _f

        def size_same(w: int, h: int) -> Tuple[int, int]:
            return w, h

        def size_swap(w: int, h: int) -> Tuple[int, int]:
            return h, w

        t_rot90 = Transform("rot_90", rot90, size_swap, ("rot90",))
        t_rot180 = Transform("rot_180", rot180, size_same, ("rot180",))
        t_rot270 = Transform("rot_270", rot270, size_swap, ("rot270",))
        t_flipH = Transform("flip_H", flipH, size_same, ("flipH",))
        t_flipV = Transform("flip_V", flipV, size_same, ("flipV",))
        t_rot90_flipH = Transform(
            "rot_90_flip_H",
            compose(rot90, flipH, size_swap),
            size_swap,
            ("rot90", "flipH"),
        )
        t_rot270_flipH = Transform(
            "rot_270_flip_H",
            compose(rot270, flipH, size_swap),
            size_swap,
            ("rot270", "flipH"),
        )

        return [
            t_rot90,
            t_rot180,
            t_rot270,
            t_flipH,
            t_flipV,
            t_rot90_flipH,
            t_rot270_flipH,
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate augmented images and COCO segmentations")
    parser.add_argument("--images_dir", required=True, type=Path)
    parser.add_argument("--json_biigle_dir", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument(
        "--log_csv",
        type=Path,
        default=None,
        help="CSV log path (default: out_dir/augmentation_log.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    augmenter = CocoAugmenter(
        images_dir=args.images_dir,
        json_dir=args.json_biigle_dir,
        out_dir=args.out_dir,
        log_csv=args.log_csv,
    )
    augmenter.run()


if __name__ == "__main__":
    main()
