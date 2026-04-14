# YOLOv11m-Seg Training and Inference (COCO JSON per Image)

This project prepares a YOLOv11 segmentation dataset from per-image COCO JSON files, trains a `yolo11m-seg` model with Ultralytics, and exports segmentation polygons for inference images. Training and prediction use separate config files to keep dataset paths out of inference-only runs.

## What It Does
- Converts per-image COCO JSON annotations into YOLO-seg polygon labels.
- Builds a YOLO dataset structure under `dataset_root`.
- Trains Ultralytics `yolo11m-seg` with your dataset.
- Produces:
  - `best.pt` and `last.pt` model weights.
  - `map_vs_epochs.png`, `map50_vs_epochs.png`.
  - Precision–Recall curves from Ultralytics validation.
  - `metrics.txt` with per-class precision/recall/AP and aggregate mAP50 for train/val/test plus F1.
- Runs inference and exports per-image polygon detections in JSON.
- Overlays prediction contours from YOLO JSON outputs.

## Requirements
- Python 3.9+ (3.10+ recommended)
- Ultralytics (YOLOv11)
- `numpy`, `pandas`, `matplotlib`, `PyYAML`

Install them with (recommended):

```bash
pip install -r requirements.txt
```

or:
```bash
pip install ultralytics pandas matplotlib pillow pyyaml
```

## Input Data Format
Each image has a matching COCO JSON file with the same base name (e.g. `capt0025.jpg` + `capt0025.json`), containing:
- `images`: single entry with `file_name`, `width`, `height`
- `categories`: list of class names
- `annotations`: per-instance `bbox` and `segmentation` polygons

Only polygon segmentations are used. `iscrowd=1` annotations are ignored.

## Project Files
- [main.py](/Users/aavelino/PycharmProjects/yolo_segm/main.py): CLI entry point for training and inference.
- [configs/train_config.yaml](/Users/aavelino/PycharmProjects/yolo_segm/configs/train_config.yaml): training configuration template.
- [configs/predict_config.yaml](/Users/aavelino/PycharmProjects/yolo_segm/configs/predict_config.yaml): prediction configuration template.

## Training Configuration File
Edit `configs/train_config.yaml` with your paths:
```yaml
train_images_dir: /absolute/path/to/train/images
train_labels_dir: /absolute/path/to/train/labels_json
val_images_dir: /absolute/path/to/val/images
val_labels_dir: /absolute/path/to/val/labels_json
test_images_dir: /absolute/path/to/test/images
test_labels_dir: /absolute/path/to/test/labels_json
dataset_root: ./yolo_dataset
model: yolo11m-seg.pt
epochs: 100
imgsz: 1024
batch: 4
workers: 2
device: cpu
cache: false
pretrained: true
project_dir: .
run_name: insect_seg_train
include_categories: null
save_console: true
```

Key options:
- `model`: Ultralytics checkpoint (use `yolo11m-seg.pt` for v11).
- `imgsz`: training image size (reduce for CPU-only).
- `device`: `"cpu"`, `"mps"` (macOS), or `"0"` for GPU.
- `include_categories`: optional list of class names to include.
- `project_dir`: project root where training outputs are written under `runs/`. Relative paths are resolved against the training config file location.

## Prediction Configuration File
Use `configs/predict_config.yaml` for inference-only runs:
```yaml
source_dir: /absolute/path/to/images
output_dir: /absolute/path/to/output_json
weights: /absolute/path/to/best.pt
imgsz: 1024
conf: 0.25
device: cpu
generate_contours: true
contours_out_dir: /absolute/path/to/output_contours
contour_color: red
contour_width: 8
contours_ext: .jpg
contours_confidence: true
contours_class_name: true
contours_fontsize: 70
debug_shapes: true
save_console: true
```

Key options:
- `source_dir`: directory of input images.
- `output_dir`: directory for JSON predictions.
- `weights`: path to trained weights for inference.
- `imgsz`: inference image size.
- `conf`: confidence threshold for detections.
- `device`: `"cpu"`, `"mps"` (macOS), or `"0"` for GPU.
- `generate_contours`: when `true`, creates contour overlay images after prediction.
- Contours from predictions are rendered using the `plot-predictions` implementation.
- `contours_out_dir`: optional output directory for contour overlays (defaults to `<output_dir>/contours`).
- `contour_color`, `contour_width`, `contours_ext`: contour rendering settings.
- `contours_confidence`: when `true`, writes confidence values above prediction contours.
- `contours_class_name`: when `true`, writes class names above prediction contours.
- `contours_fontsize`: font size for contour labels (default `30`).

## Run From CLI
Train:
```bash
python main.py train --config configs/train_config.yaml
```

Train and auto-generate contour overlays for the test split:
```bash
python main.py train --config configs/train_config.yaml --plot-test-contours
```
Test contour labels are enabled by default. You can control labeling and font size:
```bash
python main.py train --config configs/train_config.yaml --plot-test-contours \
  --plot-test-contours-class-name --plot-test-contours-fontsize 40
```
Disable labels if needed:
```bash
python main.py train --config configs/train_config.yaml --plot-test-contours \
  --no-plot-test-contours-class-name
```

Predict:
```bash
python main.py predict --config configs/predict_config.yaml --weights /path/to/best.pt --source /path/to/images --out /path/to/output_json
```
If `--config` is omitted, `configs/predict_config.yaml` is used by default.

Plot contour overlays (from BIIGLE/COCO JSONs):
```bash
python main.py plot-biigle-segmentation --images /path/to/images --json-dir /path/to/jsons --out /path/to/overlays
```
Class labels (when enabled) are resolved from the COCO `categories` list by `category_id`.

Plot prediction overlays (from Robo-style JSONs):
```bash
python main.py plot-predictions-robo --images /path/to/images --json-dir /path/to/robo_jsons --out /path/to/overlays
```
Robo JSONs are expected to include `predictions.predictions[*].points`, where each point has `x`/`y`.
Images are matched by JSON filename stem (e.g. `image_001.json` -> `image_001.jpg`).
Example files are available under `examples/robo_segmentation/`.

Contour options:
```bash
python main.py plot-biigle-segmentation --images /path/to/images --json-dir /path/to/jsons --out /path/to/overlays \
  --contour-color red --contour-width 8 --contours-ext .jpg
```

Add class labels on segmentation contours:
```bash
python main.py plot-biigle-segmentation --images /path/to/images --json-dir /path/to/jsons --out /path/to/overlays \
  --class_name --fontsize 70
```
```bash
python main.py plot-predictions-robo --images /path/to/images --json-dir /path/to/robo_jsons --out /path/to/overlays \
  --confidence --class_name --fontsize 70
```

Plot contour overlays (from prediction JSONs):
```bash
python main.py plot-predictions --images /path/to/images --json-dir /path/to/pred_jsons --out /path/to/overlays
```

Add confidence/class labels on contours:
```bash
python main.py plot-predictions --images /path/to/images --json-dir /path/to/pred_jsons --out /path/to/overlays \
  --confidence --class_name --fontsize 70 --contour-color red --contour-width 8
```

## Data Augmentation (Images + COCO JSON)
Use the `data-augmentation` CLI to generate 7 augmented variants per image and update COCO polygon segmentations to match:
```bash
./data-augmentation --images_dir /path/to/images \
  --json_biigle_dir /path/to/json_biigle \
  --out_dir /path/to/output/dir
```

Generated variants per input image:
1. `*_rot_90` (90° clockwise)
2. `*_rot_180`
3. `*_rot_270`
4. `*_flip_H`
5. `*_flip_V`
6. `*_rot_90_flip_H`
7. `*_rot_270_flip_H`

Notes:
- Each output image has a matching JSON with transformed `segmentation`.
- `bbox` and `area` are recomputed from the transformed polygons.
- JSONs with `images` length != 1 are skipped with a warning.
- A CSV log is written to `<out_dir>/augmentation_log.csv` by default.

Optional log path:
```bash
./data-augmentation --images_dir /path/to/images \
  --json_biigle_dir /path/to/json_biigle \
  --out_dir /path/to/output/dir \
  --log_csv /path/to/log.csv
```

## Outputs
Training outputs are written under:
```
<project_dir>/runs/<run_name>/
```
Typical files:
- `weights/best.pt`, `weights/last.pt`
- `results.csv` (metrics per epoch)
- `map_vs_epochs.png`, `map50_vs_epochs.png`
- `PR_curve.png` (from validation plots; stored under `<project_dir>/runs/<run_name>/val/`)
- `metrics.txt`
- `test_contours/` (if `--plot-test-contours` is used)
Validation and evaluation artifacts are stored under:
- `<project_dir>/runs/<run_name>/val/`
- `<project_dir>/runs/<run_name>/train/`
- `<project_dir>/runs/<run_name>/test/`

Inference outputs:
- One JSON per image, each containing a list of detections with polygon points.
- Prediction JSON structure:
  - `image`: source image file name.
  - `detections`: list of objects containing `class_id`, `class_name`, `confidence`, and `polygon` (list of `[x, y]` points).

## Colab / GPU Notes
Set:
- `device`: `"0"` (or `"0,1"` for multi-GPU)
- Increase `batch` and `imgsz` as GPU allows

## Troubleshooting
- If Ultralytics can’t find `yolo11m-seg.pt`, install/upgrade `ultralytics` or download the weights.
- Ensure JSONs contain valid polygons and correct image sizes.
