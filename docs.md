# Documentation

## Overview
This project provides a command-line interface to:
1. Convert per-image COCO JSON annotations into YOLO segmentation labels.
2. Train a YOLOv11m-seg model using Ultralytics.
3. Run inference and export per-image polygon detections using a separate prediction config.

The workflow is optimized for segmentation and classification of insects with clean backgrounds. The dataset is expected to be in per-image COCO format, one JSON per image.

## Architecture
The implementation is a single Python entry point:
- [main.py](/Users/aavelino/PycharmProjects/yolo_segm/main.py)

High-level steps:
1. **Dataset preparation**
   - Reads JSONs from `*_labels_dir`.
   - Links or copies images from `*_images_dir`.
   - Writes YOLO segmentation labels into `dataset_root/labels/<split>`.
   - Writes `data.yaml` for Ultralytics.
2. **Training**
   - Starts from `yolo11m-seg.pt` (default).
   - Uses Ultralytics `YOLO(...).train()`.
3. **Evaluation and plots**
   - Extracts `results.csv` to generate `map_vs_epochs.png` and `map50_vs_epochs.png`.
   - Runs `.val()` on train/val/test splits.
   - Writes `metrics.txt` with per-class metrics and aggregate results.
4. **Inference**
   - Runs `.predict()` on images in a directory.
   - Exports polygon contours per instance as JSON.
5. **Contour overlays**
   - Draws COCO polygon contours on top of the original image.
   - Draws YOLO prediction polygons from prediction JSON outputs.
   - Can be run as a standalone CLI command or after training/prediction.

## Inputs

### Training Configuration JSON
The training workflow expects a JSON file with the following fields:

Required:
- `train_images_dir`
- `train_labels_dir`
- `val_images_dir`
- `val_labels_dir`
- `test_images_dir`
- `test_labels_dir`
- `dataset_root`

Optional:
- `model` (default `yolo11m-seg.pt`)
- `epochs` (default `100`)
- `imgsz` (default `1024`)
- `batch` (default `4`)
- `workers` (default `2`)
- `device` (`cpu`, `mps`, `0`)
- `cache` (default `false`)
- `pretrained` (default `true`)
- `project_dir` (default `.`; outputs go under `<project_dir>/runs`)
- `run_name` (default `insect_seg_train`)
- `include_categories` (default `null`)
CLI-only options:
- `--plot-test-contours` to generate overlay images for the test split after training.
- `--contours-out` to override the output directory for test overlays.

### Prediction Configuration JSON
Inference runs use a separate JSON file that only includes prediction-time settings:

Required:
- None (all fields have defaults)

Optional:
- `imgsz` (default `1024`)
- `conf` (default `0.25`)
- `device` (`cpu`, `mps`, `0`)
- `source_dir` (directory of input images)
- `output_dir` (directory for JSON predictions)
- `weights` (path to trained weights)
- `generate_contours` (default `false`)
- `contours_out_dir` (defaults to `<output_dir>/contours`)
- `contour_color` (default `#ff0000`)
- `contour_width` (default `5`)
- `contours_ext` (default `.jpg`)
- `contours_confidence` (default `false`)
- `contours_class_name` (default `false`)
- `contours_fontsize` (default `30`)
- `generate_contours` uses the prediction JSON format and the `plot-predictions` implementation.

### Data Layout
Each split (train/val/test) must be:
- Images in a directory (e.g. `train_images_dir`)
- JSON files in another directory (e.g. `train_labels_dir`)

Each JSON must describe exactly one image (single entry in `images`), and the JSON file name must match the image file name (e.g. `capt0025.json` for `capt0025.jpg`).

## Outputs

### Training Outputs
Written under:
```
<project_dir>/runs/<run_name>/
```

Key artifacts:
- `weights/best.pt` and `weights/last.pt`
- `results.csv`
- `map_vs_epochs.png`
- `map50_vs_epochs.png`
- `PR_curve.png` (Ultralytics plots)
- `metrics.txt`
- `test_contours/` (if enabled)

### metrics.txt
Contains:
- Precision per class (validation split)
- Recall per class (validation split)
- Average precision per class (validation split)
- mAP@50 for train/val/test
- F1 (computed from mean P/R on validation)

### Inference Outputs
Each image produces:
```
<out_dir>/<image_stem>.json
```
with polygon coordinates and class predictions.
Prediction JSON structure:
- `image`: source image file name.
- `detections`: list of objects containing `class_id`, `class_name`, `confidence`, and `polygon` (list of `[x, y]` points).

## CLI Usage
Train:
```bash
python main.py train --config configs/train_config.yaml
```

Train and generate test overlays with labels (default enabled):
```bash
python main.py train --config configs/train_config.yaml --plot-test-contours \
  --plot-test-contours-class-name --plot-test-contours-fontsize 30
```

Disable test overlay labels:
```bash
python main.py train --config configs/train_config.yaml --plot-test-contours \
  --no-plot-test-contours-class-name
```

Predict:
```bash
python main.py predict --config configs/predict_config.yaml --weights /path/to/best.pt --source /path/to/images --out /path/to/output_json
```
If `--config` is omitted, `configs/predict_config.yaml` is used by default.

Plot contours (BIIGLE/COCO JSONs):
```bash
python main.py plot-biigle-segmentation --images /path/to/images --json-dir /path/to/jsons --out /path/to/overlays
```
When `--class_name` is used, labels are resolved from COCO `categories` by `category_id`.

Plot prediction contours (Robo-style JSONs):
```bash
python main.py plot-predictions-robo --images /path/to/images --json-dir /path/to/robo_jsons --out /path/to/overlays
```
Robo JSONs should include `predictions.predictions[*].points` with `x`/`y` pairs.
Images are matched by JSON filename stem (e.g. `image_001.json` -> `image_001.jpg`).
Example files are available under `examples/robo_segmentation/`.

Plot prediction contours:
```bash
python main.py plot-predictions --images /path/to/images --json-dir /path/to/pred_jsons --out /path/to/overlays
```

Add confidence/class labels:
```bash
python main.py plot-predictions --images /path/to/images --json-dir /path/to/pred_jsons --out /path/to/overlays \
  --confidence --class_name --fontsize 30
```

With custom color/width/extension:
```bash
python main.py plot-biigle-segmentation --images /path/to/images --json-dir /path/to/jsons --out /path/to/overlays \
  --contour-color "#ff0000" --contour-width 5 --contours-ext .jpg
```

Add class labels to segmentation contours:
```bash
python main.py plot-biigle-segmentation --images /path/to/images --json-dir /path/to/jsons --out /path/to/overlays \
  --class_name --fontsize 30
```
```bash
python main.py plot-predictions-robo --images /path/to/images --json-dir /path/to/robo_jsons --out /path/to/overlays \
  --confidence --class_name --fontsize 30
```

## Performance Guidance
For macOS Catalina (CPU-only):
- Set `device` to `"cpu"` or `"mps"` (if supported).
- Reduce `imgsz` to 640 or 512 for faster training.
- Reduce `batch` if memory is limited.

For GPU/Colab:
- Set `device` to `"0"`.
- Increase `batch` and `imgsz` as GPU memory allows.

## Notes and Assumptions
- Only polygon `segmentation` is supported.
- `iscrowd=1` annotations are skipped.
- Category mapping is built from training JSONs and applied to all splits.
