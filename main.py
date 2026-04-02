import argparse
import csv
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class SplitPaths:
    images_dir: Path
    labels_dir: Path


def load_config(path: Path) -> dict:
    """Load a YAML configuration file."""
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to load YAML config files. Install it with `pip install pyyaml`."
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping (key/value pairs).")
    return data


def resolve_project_root(config: dict, config_path: Path) -> Path:
    project_dir = Path(config.get("project_dir", "."))
    if not project_dir.is_absolute():
        project_dir = (config_path.parent / project_dir).resolve()
    else:
        project_dir = project_dir.resolve()
    return project_dir


def default_predict_config_path() -> Path:
    """Return the default prediction config path shipped with this repo."""
    return Path(__file__).resolve().parent / "configs" / "predict_config.yaml"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def list_json_files(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() == ".json"])


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        self.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


@contextmanager
def tee_stdout_stderr(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        orig_out = sys.stdout
        orig_err = sys.stderr
        sys.stdout = _TeeStream(orig_out, log_f)
        sys.stderr = _TeeStream(orig_err, log_f)
        try:
            yield
        finally:
            sys.stdout = orig_out
            sys.stderr = orig_err


def parse_categories(json_paths: List[Path]) -> Tuple[Dict[int, str], Dict[str, int]]:
    id_to_name: Dict[int, str] = {}
    for p in json_paths:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for cat in data.get("categories", []):
            cid = int(cat["id"])
            name = str(cat["name"])
            if cid in id_to_name and id_to_name[cid] != name:
                raise ValueError(
                    f"Category id {cid} has conflicting names: "
                    f"{id_to_name[cid]} vs {name} in {p}"
                )
            id_to_name[cid] = name
    if not id_to_name:
        raise ValueError("No categories found in training annotations.")
    id_sorted = sorted(id_to_name.keys())
    name_to_idx = {id_to_name[cid]: idx for idx, cid in enumerate(id_sorted)}
    return id_to_name, name_to_idx


def write_data_yaml(
    dataset_root: Path,
    names: List[str],
) -> Path:
    data_yaml = dataset_root / "data.yaml"
    with data_yaml.open("w", encoding="utf-8") as f:
        f.write(f"path: {dataset_root.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("names:\n")
        for name in names:
            f.write(f"  - {name}\n")
    return data_yaml


def convert_split(
    split_name: str,
    split_paths: SplitPaths,
    dataset_root: Path,
    name_to_idx: Dict[str, int],
    include_categories: List[str] | None,
) -> None:
    img_out_dir = dataset_root / "images" / split_name
    lbl_out_dir = dataset_root / "labels" / split_name
    ensure_dir(img_out_dir)
    ensure_dir(lbl_out_dir)

    for json_path in list_json_files(split_paths.labels_dir):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        images = data.get("images", [])
        if not images:
            raise ValueError(f"No images entry in {json_path}")
        img_info = images[0]
        file_name = img_info.get("file_name") or (json_path.stem + ".jpg")
        img_w = float(img_info["width"])
        img_h = float(img_info["height"])

        src_img = split_paths.images_dir / file_name
        if not src_img.exists():
            raise FileNotFoundError(f"Image not found: {src_img}")

        dst_img = img_out_dir / file_name
        safe_symlink_or_copy(src_img, dst_img)

        label_lines: List[str] = []
        for ann in data.get("annotations", []):
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            cat_id = int(ann["category_id"])
            cat_name = None
            for cat in data.get("categories", []):
                if int(cat["id"]) == cat_id:
                    cat_name = str(cat["name"])
                    break
            if cat_name is None:
                continue
            if include_categories and cat_name not in include_categories:
                continue
            if cat_name not in name_to_idx:
                continue
            cls = name_to_idx[cat_name]
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            if isinstance(seg, dict):
                continue
            for polygon in seg:
                if len(polygon) < 6:
                    continue
                coords = []
                for i in range(0, len(polygon), 2):
                    x = float(polygon[i]) / img_w
                    y = float(polygon[i + 1]) / img_h
                    coords.append(f"{x:.6f}")
                    coords.append(f"{y:.6f}")
                label_lines.append(f"{cls} " + " ".join(coords))

        label_path = lbl_out_dir / (Path(file_name).stem + ".txt")
        with label_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))


def prepare_dataset(config: dict) -> Path:
    dataset_root = Path(config["dataset_root"]).resolve()
    ensure_dir(dataset_root)

    splits = {
        "train": SplitPaths(
            images_dir=Path(config["train_images_dir"]).resolve(),
            labels_dir=Path(config["train_labels_dir"]).resolve(),
        ),
        "val": SplitPaths(
            images_dir=Path(config["val_images_dir"]).resolve(),
            labels_dir=Path(config["val_labels_dir"]).resolve(),
        ),
        "test": SplitPaths(
            images_dir=Path(config["test_images_dir"]).resolve(),
            labels_dir=Path(config["test_labels_dir"]).resolve(),
        ),
    }

    train_jsons = list_json_files(splits["train"].labels_dir)
    id_to_name, name_to_idx = parse_categories(train_jsons)

    include_categories = config.get("include_categories")
    if include_categories:
        include_categories = [str(x) for x in include_categories]

    for split_name, split_paths in splits.items():
        convert_split(split_name, split_paths, dataset_root, name_to_idx, include_categories)

    names = [name for _, name in sorted(id_to_name.items(), key=lambda x: x[0])]
    data_yaml = write_data_yaml(dataset_root, names)
    return data_yaml


def plot_metrics(results_csv: Path, output_dir: Path) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    if not results_csv.exists():
        return

    df = pd.read_csv(results_csv)
    epoch_col = "epoch"
    map_col = None
    map50_col = None
    for c in df.columns:
        if "metrics/mAP50-95(B)" in c or "metrics/mAP50-95(M)" in c:
            map_col = c
        if "metrics/mAP50(B)" in c or "metrics/mAP50(M)" in c:
            map50_col = c

    if map_col:
        plt.figure()
        plt.plot(df[epoch_col], df[map_col])
        plt.xlabel("epoch")
        plt.ylabel("mAP50-95")
        plt.title("mAP vs epochs")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "map_vs_epochs.png", dpi=150)
        plt.close()

    if map50_col:
        plt.figure()
        plt.plot(df[epoch_col], df[map50_col])
        plt.xlabel("epoch")
        plt.ylabel("mAP50")
        plt.title("mAP50 vs epochs")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "map50_vs_epochs.png", dpi=150)
        plt.close()


def extract_metrics(metrics) -> dict:
    def to_list(x):
        if x is None:
            return None
        try:
            return [float(v) for v in x]
        except TypeError:
            return [float(x)]

    out = {}
    for attr in ["p", "r", "ap", "ap50"]:
        val = getattr(metrics, attr, None)
        if val is not None:
            out[attr] = to_list(val)

    map50 = getattr(metrics, "map50", None)
    map50_95 = getattr(metrics, "map", None)
    out["map50"] = float(map50) if map50 is not None else None
    out["map50_95"] = float(map50_95) if map50_95 is not None else None
    return out


def write_metrics_report(
    output_dir: Path,
    class_names: List[str],
    train_metrics: dict | None,
    val_metrics: dict | None,
    test_metrics: dict | None,
) -> None:
    report_path = output_dir / "metrics.txt"

    def fmt(v):
        return "NA" if v is None else f"{v:.6f}"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("Per-class precision/recall/AP (validation split):\n")
        if val_metrics and val_metrics.get("p") and val_metrics.get("r") and val_metrics.get("ap"):
            for i, name in enumerate(class_names):
                p = val_metrics["p"][i] if i < len(val_metrics["p"]) else None
                r = val_metrics["r"][i] if i < len(val_metrics["r"]) else None
                ap = val_metrics["ap"][i] if i < len(val_metrics["ap"]) else None
                f.write(f"- {name}: P={fmt(p)} R={fmt(r)} AP={fmt(ap)}\n")
        else:
            f.write("- NA\n")

        f.write("\nAggregate metrics:\n")
        f.write(f"- mAP50 train: {fmt(train_metrics.get('map50') if train_metrics else None)}\n")
        f.write(f"- mAP50 val: {fmt(val_metrics.get('map50') if val_metrics else None)}\n")
        f.write(f"- mAP50 test: {fmt(test_metrics.get('map50') if test_metrics else None)}\n")

        if val_metrics and val_metrics.get("p") and val_metrics.get("r"):
            p_mean = sum(val_metrics["p"]) / len(val_metrics["p"])
            r_mean = sum(val_metrics["r"]) / len(val_metrics["r"])
            f1 = 2 * p_mean * r_mean / (p_mean + r_mean) if (p_mean + r_mean) > 0 else 0.0
            f.write(f"- F1 (val, mean P/R): {fmt(f1)}\n")
        else:
            f.write("- F1 (val, mean P/R): NA\n")


def parse_color(value: str) -> Tuple[int, int, int]:
    value = value.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) == 6:
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
        return r, g, b
    parts = [p.strip() for p in value.split(",")]
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), int(parts[2])
    try:
        from PIL import ImageColor

        return ImageColor.getrgb(value)
    except (ValueError, ImportError) as exc:
        raise ValueError(
            "Color must be a name (e.g. red), hex (#RRGGBB), or RGB (R,G,B)."
        ) from exc


def draw_contours_on_image(
    image_path: Path,
    json_path: Path,
    out_path: Path,
    color: Tuple[int, int, int],
    line_width: int,
    show_class_name: bool = False,
    font_size: int = 30,
) -> None:
    """Draw COCO segmentation polygons, optionally adding class name labels."""
    from PIL import Image, ImageDraw
    from PIL import ImageFont

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    font = None
    if show_class_name:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            pil_dir = Path(ImageFont.__file__).resolve().parent
            candidates = [
                pil_dir / "fonts" / "DejaVuSans.ttf",
                pil_dir / "fonts" / "DejaVuSans-Bold.ttf",
                Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
                Path("/Library/Fonts/Arial.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    font = ImageFont.truetype(candidate.as_posix(), font_size)
                    break

    categories = {}
    if show_class_name:
        for cat in data.get("categories", []):
            try:
                categories[int(cat["id"])] = str(cat["name"])
            except (KeyError, TypeError, ValueError):
                continue

    for ann in data.get("annotations", []):
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        seg = ann.get("segmentation", [])
        if not seg or isinstance(seg, dict):
            continue
        class_name = None
        if show_class_name:
            try:
                class_name = categories.get(int(ann.get("category_id")))
            except (TypeError, ValueError):
                class_name = None
        for polygon in seg:
            if len(polygon) < 6:
                continue
            pts = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            if len(pts) >= 2:
                draw.line(pts + [pts[0]], fill=color, width=line_width)
            if show_class_name and class_name and pts:
                min_point = min(pts, key=lambda p: p[1])
                text_x = max(int(min_point[0]), 0)
                text_y = max(int(min_point[1] - font_size), 0)
                draw.text((text_x, text_y), class_name, fill=color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def draw_robo_contours_on_image(
    image_path: Path,
    json_path: Path,
    out_path: Path,
    color: Tuple[int, int, int],
    line_width: int,
    show_confidence: bool = False,
    show_class_name: bool = False,
    font_size: int = 30,
) -> None:
    """Draw prediction polygons from Robo-style JSONs with `predictions.points`."""
    from PIL import Image, ImageDraw
    from PIL import ImageFont

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    font = None
    if show_confidence or show_class_name:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            pil_dir = Path(ImageFont.__file__).resolve().parent
            candidates = [
                pil_dir / "fonts" / "DejaVuSans.ttf",
                pil_dir / "fonts" / "DejaVuSans-Bold.ttf",
                Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
                Path("/Library/Fonts/Arial.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    try:
                        font = ImageFont.truetype(str(candidate), font_size)
                        break
                    except OSError:
                        continue
            if font is None:
                font = ImageFont.load_default()

    preds_block = data.get("predictions", {})
    predictions = preds_block.get("predictions", [])
    if not isinstance(predictions, list):
        predictions = []

    for pred in predictions:
        points = pred.get("points", [])
        if not isinstance(points, list) or len(points) < 3:
            continue
        pts: List[Tuple[float, float]] = []
        for pt in points:
            if isinstance(pt, dict):
                try:
                    pts.append((float(pt["x"]), float(pt["y"])))
                except (KeyError, TypeError, ValueError):
                    continue
            elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                try:
                    pts.append((float(pt[0]), float(pt[1])))
                except (TypeError, ValueError):
                    continue
        if len(pts) < 2:
            continue
        draw.line(pts + [pts[0]], fill=color, width=line_width)
        if (show_confidence or show_class_name) and pts:
            label_parts: List[str] = []
            if show_class_name:
                class_name = pred.get("class")
                if class_name is not None:
                    label_parts.append(str(class_name))
            if show_confidence:
                confidence = pred.get("confidence")
                if confidence is not None:
                    label_parts.append(f"{float(confidence):.4f}")
            if label_parts:
                if show_confidence and show_class_name and len(label_parts) == 2:
                    label = ", ".join(label_parts)
                else:
                    label = " ".join(label_parts)
                min_point = min(pts, key=lambda p: p[1])
                text_x = max(int(min_point[0]), 0)
                text_y = max(int(min_point[1] - font_size), 0)
                draw.text((text_x, text_y), label, fill=color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def draw_pred_contours_on_image(
    image_path: Path,
    json_path: Path,
    out_path: Path,
    color: Tuple[int, int, int],
    line_width: int,
    show_confidence: bool = False,
    show_class_name: bool = False,
    font_size: int = 30,
) -> None:
    """Draw prediction polygons, optionally adding class names/confidence labels."""
    from PIL import Image, ImageDraw
    from PIL import ImageFont

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    font = None
    if show_confidence or show_class_name:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            pil_dir = Path(ImageFont.__file__).resolve().parent
            candidates = [
                pil_dir / "fonts" / "DejaVuSans.ttf",
                pil_dir / "fonts" / "DejaVuSans-Bold.ttf",
                Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
                Path("/Library/Fonts/Arial.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    try:
                        font = ImageFont.truetype(str(candidate), font_size)
                        break
                    except OSError:
                        continue
            if font is None:
                font = ImageFont.load_default()

    for det in data.get("detections", []):
        polygon = det.get("polygon", [])
        if not polygon or len(polygon) < 3:
            continue
        pts = [(float(x), float(y)) for x, y in polygon]
        if len(pts) >= 2:
            draw.line(pts + [pts[0]], fill=color, width=line_width)
        if (show_confidence or show_class_name) and pts:
            label_parts: List[str] = []
            if show_class_name:
                class_name = det.get("class_name")
                if class_name is not None:
                    label_parts.append(str(class_name))
            if show_confidence:
                confidence = det.get("confidence")
                if confidence is not None:
                    label_parts.append(f"{float(confidence):.4f}")
            if label_parts:
                if show_confidence and show_class_name and len(label_parts) == 2:
                    label = ", ".join(label_parts)
                else:
                    label = " ".join(label_parts)
                min_point = min(pts, key=lambda p: p[1])
                text_x = max(int(min_point[0]), 0)
                text_y = max(int(min_point[1] - font_size), 0)
                draw.text((text_x, text_y), label, fill=color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def plot_segmentation(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    color: Tuple[int, int, int],
    line_width: int,
    output_ext: str | None,
    show_class_name: bool,
    font_size: int,
) -> None:
    """Overlay BIIGLE/COCO polygons on images, optionally adding class name labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for json_path in list_json_files(labels_dir):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        images = data.get("images", [])
        if not images:
            continue
        file_name = images[0].get("file_name") or (json_path.stem + ".jpg")
        image_path = images_dir / file_name
        if not image_path.exists():
            continue
        stem = Path(file_name).stem + "_segm"
        ext = output_ext if output_ext else ".jpg"
        out_path = out_dir / f"{stem}{ext}"
        draw_contours_on_image(
            image_path,
            json_path,
            out_path,
            color,
            line_width,
            show_class_name=show_class_name,
            font_size=font_size,
        )


def plot_predictions_robo(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    color: Tuple[int, int, int],
    line_width: int,
    output_ext: str | None,
    show_confidence: bool,
    show_class_name: bool,
    font_size: int,
) -> None:
    """Overlay prediction polygons from Robo-style JSONs on images (match by JSON stem)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_paths = list_json_files(labels_dir)
    if not json_paths:
        print(f"No JSON files found in {labels_dir}.")
        return
    processed = 0
    skipped = 0
    missing_images = 0
    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        stem = json_path.stem
        candidates = []
        for suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".ppm"):
            candidate = images_dir / f"{stem}{suffix}"
            if candidate.exists():
                candidates.append(candidate)
        if not candidates:
            wildcard = list(images_dir.glob(f"{stem}.*"))
            candidates = [p for p in wildcard if p.is_file()]
        if not candidates:
            print(f"Warning: image not found for {json_path} (stem: {stem}) in {images_dir}")
            missing_images += 1
            skipped += 1
            continue
        if len(candidates) > 1:
            preferred = {".jpg", ".jpeg", ".png"}
            preferred_candidates = [p for p in candidates if p.suffix.lower() in preferred]
            if preferred_candidates:
                candidates = preferred_candidates
        image_path = sorted(candidates)[0]
        stem = image_path.stem + "_pred"
        ext = output_ext if output_ext else ".jpg"
        out_path = out_dir / f"{stem}{ext}"
        draw_robo_contours_on_image(
            image_path,
            json_path,
            out_path,
            color,
            line_width,
            show_confidence=show_confidence,
            show_class_name=show_class_name,
            font_size=font_size,
        )
        processed += 1
    print(
        "plot-predictions-robo summary: "
        f"processed={processed}, skipped={skipped}, missing_images={missing_images}"
    )


def plot_predictions(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    color: Tuple[int, int, int],
    line_width: int,
    output_ext: str | None,
    show_confidence: bool,
    show_class_name: bool,
    font_size: int,
) -> None:
    """Overlay YOLO prediction polygons on images using prediction JSONs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for json_path in list_json_files(labels_dir):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        file_name = data.get("image") or (json_path.stem + ".jpg")
        image_path = images_dir / file_name
        if not image_path.exists():
            continue
        stem = Path(file_name).stem + "_pred"
        ext = output_ext if output_ext else ".jpg"
        out_path = out_dir / f"{stem}{ext}"
        draw_pred_contours_on_image(
            image_path,
            json_path,
            out_path,
            color,
            line_width,
            show_confidence=show_confidence,
            show_class_name=show_class_name,
            font_size=font_size,
        )


def train(
    config_path: Path,
    plot_test_contours: bool = False,
    contours_out: Path | None = None,
    contour_color: Tuple[int, int, int] = (255, 0, 0),
    contour_width: int = 5,
    contours_ext: str | None = None,
    save_console: bool = False,
    plot_test_contours_class_name: bool = True,
    plot_test_contours_fontsize: int = 30,
) -> None:
    """Prepare dataset and train a YOLOv11m-seg model using the training config.

    When plot_test_contours is enabled, class labels and font size are controlled
    by plot_test_contours_class_name and plot_test_contours_fontsize.
    """
    config_path = config_path.resolve()
    config = load_config(config_path)
    if not save_console:
        save_console = bool(config.get("save_console", False))
    data_yaml = prepare_dataset(config)

    from ultralytics import YOLO

    model_path = config.get("model", "yolo11m-seg.pt")
    model = YOLO(model_path)

    run_name = config.get("run_name") or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    project_root = resolve_project_root(config, config_path)
    runs_dir = project_root / "runs"
    ensure_dir(runs_dir)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=int(config.get("epochs", 100)),
        imgsz=int(config.get("imgsz", 1024)),
        batch=int(config.get("batch", 4)),
        workers=int(config.get("workers", 2)),
        device=str(config.get("device", "cpu")),
        cache=bool(config.get("cache", False)),
        project=str(runs_dir),
        name=str(run_name),
        pretrained=bool(config.get("pretrained", True)),
    )

    save_dir = runs_dir / run_name
    ensure_dir(save_dir)
    log_path = save_dir / "console.log"
    if save_console:
        with tee_stdout_stderr(log_path):
            results = model.train(**train_kwargs)
    else:
        results = model.train(**train_kwargs)
    save_dir = Path(model.trainer.save_dir)

    plot_metrics(save_dir / "results.csv", save_dir)

    val_metrics = model.val(
        split="val",
        plots=True,
        save_dir=str(save_dir / "val"),
        exist_ok=True,
    )
    train_metrics = model.val(
        split="train",
        plots=False,
        save_dir=str(save_dir / "train"),
        exist_ok=True,
    )
    test_metrics = model.val(
        split="test",
        plots=False,
        save_dir=str(save_dir / "test"),
        exist_ok=True,
    )

    class_names = model.names if hasattr(model, "names") else []
    val_seg_metrics = getattr(val_metrics, "seg", None)
    train_seg_metrics = getattr(train_metrics, "seg", None)
    test_seg_metrics = getattr(test_metrics, "seg", None)

    write_metrics_report(
        save_dir,
        class_names,
        extract_metrics(train_seg_metrics) if train_seg_metrics else None,
        extract_metrics(val_seg_metrics) if val_seg_metrics else None,
        extract_metrics(test_seg_metrics) if test_seg_metrics else None,
    )

    if plot_test_contours:
        test_images_dir = Path(config["test_images_dir"]).resolve()
        test_labels_dir = Path(config["test_labels_dir"]).resolve()
        out_dir = contours_out or (save_dir / "test_contours")
        plot_segmentation(
            test_images_dir,
            test_labels_dir,
            out_dir,
            contour_color,
            contour_width,
            contours_ext,
            show_class_name=plot_test_contours_class_name,
            font_size=plot_test_contours_fontsize,
        )

    print(f"Training complete. Outputs in: {save_dir}")


def predict(
    config_path: Path,
    weights: Path | None,
    source_dir: Path | None,
    out_dir: Path | None,
    save_console: bool = False,
) -> None:
    """Run inference on new images using the prediction config (no dataset prep)."""
    config = load_config(config_path)
    if not save_console:
        save_console = bool(config.get("save_console", False))
    imgsz = int(config.get("imgsz", 1024))
    confidence = float(config.get("conf", 0.25))
    device = str(config.get("device", "cpu"))
    generate_contours = bool(config.get("generate_contours", False))
    debug_shapes = bool(config.get("debug_shapes", False))
    contour_color = parse_color(str(config.get("contour_color", "#ff0000")))
    contour_width = int(config.get("contour_width", 5))
    contours_ext = config.get("contours_ext")
    contours_confidence = bool(config.get("contours_confidence", False))
    contours_class_name = bool(config.get("contours_class_name", False))
    contours_fontsize = int(config.get("contours_fontsize", 30))

    if weights is None:
        weights_value = config.get("weights")
        weights = Path(weights_value) if weights_value else None
    if source_dir is None:
        source_value = config.get("source_dir")
        source_dir = Path(source_value) if source_value else None
    config_out_value = config.get("output_dir")
    if out_dir is None:
        out_dir = Path(config_out_value) if config_out_value else None

    if weights is None or source_dir is None or out_dir is None:
        raise ValueError(
            "Predict requires weights, source_dir, and output_dir. "
            "Provide them in configs/predict_config.yaml or via CLI flags."
        )

    from ultralytics import YOLO
    from ultralytics.utils.ops import masks2segments

    model = YOLO(str(weights))
    ensure_dir(out_dir)

    def _run_predict():
        summary_rows: List[Tuple[str, float, int, int]] = []
        for img_path in sorted(source_dir.iterdir()):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                continue
            start_time = time.perf_counter()
            results = model.predict(
                source=str(img_path),
                conf=confidence,
                imgsz=imgsz,
                device=device,
            )
            if not results:
                continue
            r = results[0]
            out = {"image": img_path.name, "detections": []}
            if r.masks is None:
                out_path = out_dir / f"{img_path.stem}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2)
                elapsed = time.perf_counter() - start_time
                summary_rows.append((img_path.name, round(elapsed, 4), 0, 0))
                continue

            mask_h, mask_w = r.masks.data.shape[-2], r.masks.data.shape[-1]
            orig_h, orig_w = (
                r.masks.orig_shape if getattr(r.masks, "orig_shape", None) else (None, None)
            )
            if debug_shapes:
                scale_x = (orig_w / mask_w) if orig_w and mask_w else 1.0
                scale_y = (orig_h / mask_h) if orig_h and mask_h else 1.0
                print(
                    "[predict] image=%s orig_shape=%s mask_shape=(%s,%s) scale=(%.6f,%.6f)"
                    % (
                        img_path.name,
                        str(r.masks.orig_shape),
                        mask_h,
                        mask_w,
                        scale_x,
                        scale_y,
                    )
                )
            segments = r.masks.xy if hasattr(r.masks, "xy") else masks2segments(r.masks.data)
            for i, seg in enumerate(segments):
                if seg.shape[0] < 3:
                    continue
                cls = int(r.boxes.cls[i].item()) if r.boxes is not None else -1
                detection_confidence = (
                    float(r.boxes.conf[i].item()) if r.boxes is not None else None
                )
                if detection_confidence is not None:
                    detection_confidence = round(detection_confidence, 4)
                polygon = seg.reshape(-1, 2)
                polygon = [[int(round(x)), int(round(y))] for x, y in polygon.tolist()]
                out["detections"].append(
                    {
                        "class_id": cls,
                        "class_name": r.names.get(cls, str(cls)),
                        "confidence": detection_confidence,
                        "polygon": polygon,
                    }
                )

            out_path = out_dir / f"{img_path.stem}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            elapsed = time.perf_counter() - start_time
            num_objects = len(out["detections"])
            num_classes = len({d["class_id"] for d in out["detections"]})
            summary_rows.append((img_path.name, round(elapsed, 4), num_objects, num_classes))

        if summary_rows:
            summary_path = out_dir / "predictions_summary.csv"
            with summary_path.open("w", encoding="utf-8", newline="") as csv_f:
                writer = csv.writer(csv_f)
                writer.writerow(
                    [
                        "imagefile",
                        "processing_time_sec",
                        "number_segm_objects",
                        "num_classes",
                    ]
                )
                writer.writerows(summary_rows)

        if generate_contours:
            contours_out = config.get("contours_out_dir")
            contours_dir = Path(contours_out) if contours_out else (out_dir / "contours")
            plot_predictions(
                images_dir=source_dir,
                labels_dir=out_dir,
                out_dir=contours_dir,
                color=contour_color,
                line_width=contour_width,
                output_ext=str(contours_ext) if contours_ext else ".jpg",
                show_confidence=contours_confidence,
                show_class_name=contours_class_name,
                font_size=contours_fontsize,
            )

        print(f"Predictions saved to: {out_dir}")

    if save_console:
        log_path = out_dir / "console.log"
        with tee_stdout_stderr(log_path):
            _run_predict()
    else:
        _run_predict()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv11m-seg training and inference")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Prepare dataset and train")
    p_train.add_argument("--config", required=True, type=Path, help="Path to config YAML")
    p_train.add_argument(
        "--plot-test-contours",
        action="store_true",
        help="Generate contour overlays for the test split after training",
    )
    p_train.add_argument(
        "--plot-test-contours-class-name",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write class names above each test contour (default: enabled)",
    )
    p_train.add_argument(
        "--plot-test-contours-fontsize",
        type=int,
        default=30,
        help="Font size for test contour labels (default: 30)",
    )
    p_train.add_argument(
        "--contours-out",
        type=Path,
        default=None,
        help="Output directory for test contour overlays (default: <run_dir>/test_contours)",
    )
    p_train.add_argument(
        "--contour-color",
        type=str,
        default="#ff0000",
        help="Contour color as a name (e.g. red), hex (#RRGGBB), or RGB (R,G,B)",
    )
    p_train.add_argument(
        "--contour-width",
        type=int,
        default=8,
        help="Contour line width in pixels",
    )
    p_train.add_argument(
        "--contours-ext",
        type=str,
        default=None,
        help="Output image extension (e.g. .jpg or .png). Defaults to .jpg.",
    )
    p_train.add_argument(
        "--save-console",
        action="store_true",
        help="Save console output to <project_dir>/runs/<run_name>/console.log",
    )

    p_predict = sub.add_parser("predict", help="Run inference and export polygons")
    p_predict.add_argument(
        "--config",
        type=Path,
        default=default_predict_config_path(),
        help="Path to prediction config YAML (default: configs/predict_config.yaml)",
    )
    p_predict.add_argument("--weights", type=Path, help="Path to trained weights")
    p_predict.add_argument("--source", type=Path, help="Directory of input images")
    p_predict.add_argument("--out", type=Path, help="Output directory for JSON results")
    p_predict.add_argument(
        "--save-console",
        action="store_true",
        help="Save console output to <output_dir>/console.log (from config)",
    )

    p_plot = sub.add_parser(
        "plot-biigle-segmentation",
        help="Overlay COCO polygons on images (BIIGLE format)",
    )
    p_plot.add_argument("--images", required=True, type=Path, help="Directory of input images")
    p_plot.add_argument(
        "--json-dir",
        required=True,
        type=Path,
        help="Directory of COCO JSONs",
    )
    p_plot.add_argument("--out", required=True, type=Path, help="Output directory for overlays")
    p_plot.add_argument(
        "--contour-color",
        type=str,
        default="#ff0000",
        help="Contour color as a name (e.g. red), hex (#RRGGBB), or RGB (R,G,B)",
    )
    p_plot.add_argument(
        "--contour-width",
        type=int,
        default=5,
        help="Contour line width in pixels",
    )
    p_plot.add_argument(
        "--contours-ext",
        type=str,
        default=None,
        help="Output image extension (e.g. .jpg or .png). Defaults to .jpg.",
    )
    p_plot.add_argument(
        "--class_name",
        action="store_true",
        help="Write class names above each segmentation contour",
    )
    p_plot.add_argument(
        "--fontsize",
        type=int,
        default=30,
        help="Font size for class labels (default: 30)",
    )

    p_plot_robo = sub.add_parser(
        "plot-predictions-robo",
        help="Overlay Robo-style prediction polygons on images",
    )
    p_plot_robo.add_argument("--images", required=True, type=Path, help="Directory of input images")
    p_plot_robo.add_argument(
        "--json-dir",
        required=True,
        type=Path,
        help="Directory of Robo segmentation JSONs",
    )
    p_plot_robo.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for overlays",
    )
    p_plot_robo.add_argument(
        "--contour-color",
        type=str,
        default="#ff0000",
        help="Contour color as a name (e.g. red), hex (#RRGGBB), or RGB (R,G,B)",
    )
    p_plot_robo.add_argument(
        "--contour-width",
        type=int,
        default=5,
        help="Contour line width in pixels",
    )
    p_plot_robo.add_argument(
        "--contours-ext",
        type=str,
        default=None,
        help="Output image extension (e.g. .jpg or .png). Defaults to .jpg.",
    )
    p_plot_robo.add_argument(
        "--confidence",
        action="store_true",
        help="Write confidence values above each prediction contour",
    )
    p_plot_robo.add_argument(
        "--class_name",
        action="store_true",
        help="Write class names above each prediction contour",
    )
    p_plot_robo.add_argument(
        "--fontsize",
        type=int,
        default=30,
        help="Font size for confidence/class labels (default: 30)",
    )

    p_plot_pred = sub.add_parser(
        "plot-predictions",
        help="Overlay YOLO prediction polygons on images",
    )
    p_plot_pred.add_argument("--images", required=True, type=Path, help="Directory of input images")
    p_plot_pred.add_argument(
        "--json-dir",
        required=True,
        type=Path,
        help="Directory of prediction JSONs",
    )
    p_plot_pred.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for overlays",
    )
    p_plot_pred.add_argument(
        "--contour-color",
        type=str,
        default="#ff0000",
        help="Contour color as a name (e.g. red), hex (#RRGGBB), or RGB (R,G,B)",
    )
    p_plot_pred.add_argument(
        "--contour-width",
        type=int,
        default=5,
        help="Contour line width in pixels",
    )
    p_plot_pred.add_argument(
        "--contours-ext",
        type=str,
        default=None,
        help="Output image extension (e.g. .jpg or .png). Defaults to .jpg.",
    )
    p_plot_pred.add_argument(
        "--confidence",
        action="store_true",
        help="Write confidence values above each prediction contour",
    )
    p_plot_pred.add_argument(
        "--class_name",
        action="store_true",
        help="Write class names above each prediction contour",
    )
    p_plot_pred.add_argument(
        "--fontsize",
        type=int,
        default=30,
        help="Font size for confidence/class labels (default: 30)",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "train":
        train(
            args.config,
            plot_test_contours=args.plot_test_contours,
            contours_out=args.contours_out,
            contour_color=parse_color(args.contour_color),
            contour_width=args.contour_width,
            contours_ext=args.contours_ext,
            save_console=args.save_console,
            plot_test_contours_class_name=args.plot_test_contours_class_name,
            plot_test_contours_fontsize=args.plot_test_contours_fontsize,
        )
    elif args.command == "predict":
        predict(args.config, args.weights, args.source, args.out, save_console=args.save_console)
    elif args.command == "plot-biigle-segmentation":
        plot_segmentation(
            args.images,
            args.json_dir,
            args.out,
            parse_color(args.contour_color),
            args.contour_width,
            args.contours_ext,
            show_class_name=args.class_name,
            font_size=args.fontsize,
        )
    elif args.command == "plot-predictions-robo":
        plot_predictions_robo(
            args.images,
            args.json_dir,
            args.out,
            parse_color(args.contour_color),
            args.contour_width,
            args.contours_ext,
            show_confidence=args.confidence,
            show_class_name=args.class_name,
            font_size=args.fontsize,
        )
    elif args.command == "plot-predictions":
        plot_predictions(
            args.images,
            args.json_dir,
            args.out,
            parse_color(args.contour_color),
            args.contour_width,
            args.contours_ext,
            show_confidence=args.confidence,
            show_class_name=args.class_name,
            font_size=args.fontsize,
        )


if __name__ == "__main__":
    main()
