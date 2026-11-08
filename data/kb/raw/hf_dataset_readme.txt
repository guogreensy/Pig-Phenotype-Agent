---
license: cc-by-4.0
task_categories:
image-classification
tags:
animal
pig
posture
livestock
multi-camera
viewpoint
agriculture
language:
en
size_categories:
10K<n<100K
---
# Viewpoint-Aware Pig Posture Recognition Dataset
This dataset supports multi-camera, viewpoint-aware pig posture recognition in livestock barn environments. It contains real-world pig images, bounding box annotations, posture class labels, and per-instance camera viewpoint angles (azimuth and elevation) derived from PnP-based camera calibration.
Code: Anil-Bhujel/viewpoint-aware-pig-posture-recognition on GitHub
---
## Dataset Summary
Images were captured from 2 real-world pig pens using 4 cameras per pen — 2 overhead fisheye turret cameras and 2 RGB Orbbec depth cameras — installed at different positions and angles. This multi-viewpoint setting captures natural variation in how postures appear under different camera perspectives.
Each annotation includes:
A bounding box around an individual pig
A posture class label (5 classes)
Camera viewpoint angles (azimuth and elevation) for that pig's position
The source image filename and resolution
## Posture Classes
| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | Laterallyingleft | Pig lying on its side, left laterally |
| 1 | Laterallyingright | Pig lying on its side, right laterally |
| 2 | Sitting | Pig in sitting posture |
| 3 | Standing | Pig standing upright |
| 4 | Sternal_lying | Pig lying on sternum (chest-down, sphinx-like) |
---
## Dataset Structure
## Dataset Statistics
| Split | Instances | Images |
|-------|-----------|--------|
| Train | 22,933 | 3,090 |
| Seen-VP test | 2,603 | 300 |
| Unseen-VP test | 11,708 | 1,350 |
---
## CSV Columns
Each CSV file has the following columns:
| Column | Type | Description |
|--------|------|-------------|
| row_id | str | Unique instance identifier |
| image_id | str | Filename of the source image |
| width | int | Image width in pixels |
| height | int | Image height in pixels |
| bbox | str | Bounding box in [x, y, w, h] format (XYWH, pixel coords) |
| class_id | int | Posture class label (0–4, see table above) |
| worldx | float | Pig floor position X (normalised, floorwidth = 1.0) |
| world_y | float | Pig floor position Y (normalised) |
| azimuth_deg | float | Camera-to-pig azimuth angle in degrees (−180 to +180) |
| elevation_deg | float | Camera elevation angle in degrees (zenith convention: 0 = overhead, 90 = horizontal) |
| elevationdowndeg | float | Positive-down elevation (0–90°, always overhead positive) |
| azimuth_sin | float | sin(azimuth) — used directly as model feature |
| azimuth_cos | float | cos(azimuth) — used directly as model feature |
| elevation_sin | float | sin(elevation) — used directly as model feature |
| elevation_cos | float | cos(elevation) — used directly as model feature |
| angle_valid | int | 1 = valid angle computed; 0 = outside pen or undistort failed |
| campossource | str | "pnp" (PnP pipeline) or "config" (manual config pipeline) |
| arrow_u | float | Azimuth direction unit vector X component (visualisation helper) |
| arrow_v | float | Azimuth direction unit vector Y component (visualisation helper) |
The unseen-VP test CSV also includes auxiliary columns inclclass and orientclass for sub-category analysis.
---
## Camera Setup
| Camera ID | Type | Model | Resolution |
|-----------|------|-------|-----------|
| pen1turcam1, pen1turcam2 | Fisheye | Turret overhead | 1280 × 720 |
| pen2turcam1, pen2turcam2 | Fisheye | Turret overhead | 1280 × 720 |
| pen1orbcam1, pen1orbcam2 | Pinhole | Orbbec Femto RGB | 1920 × 1080 |
| pen2orbcam1, pen2orbcam2 | Pinhole | Orbbec Femto RGB | 1920 × 1080 |
Camera intrinsics (checkerboard .npz for fisheye cameras and manufacturer .ini for Orbbec pinhole cameras) are available in the companion code repository.
---
## Viewpoint Angle Convention
Angles follow the zenith convention:
Azimuth (azimuth_deg): angle of the horizontal projection of the camera-to-pig ray, measured from the +X axis, counter-clockwise positive (−180° to +180°).
Elevation (elevation_deg, zenith): 0° means the camera is directly overhead the pig; 90° means the camera is at the same height as the pig (horizontal line of sight).
The (sin, cos) encoding avoids angle-wrapping discontinuities and is used directly as the four-dimensional angle feature vector [azimuthsin, azimuthcos, elevationsin, elevationcos] in model training.
---
## How to Load
## Using HuggingFace datasets
## Manual download
## Load CSV + images with pandas
## PyTorch DataLoader
The full training pipeline, PigCropDataset, and all model code live in the companion GitHub repository. Clone it first, then point it at this dataset:
---
## Using with the Code Repository
The GitHub repository contains the full pipeline for:
Camera calibration (PnP from 4 annotated floor corners)
Per-instance viewpoint angle computation
DINOv2-based posture classifier training and evaluation
Quick start with this dataset:
For full instructions including camera re-calibration, all training flags, and evaluation, see the code README.
---
## Benchmark Results
Results with the provided bestdinoangledomainmodel.pt checkpoint (DINOv2-base backbone + MLP head + angle conditioning + domain adversarial training):
## Seen Viewpoints Test Set
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Lateral lying left | 0.820 | 0.921 | 0.868 |
| Lateral lying right | 0.843 | 0.877 | 0.860 |
| Sitting | 0.844 | 0.771 | 0.806 |
| Standing | 0.986 | 0.985 | 0.985 |
| Sternal lying | 0.904 | 0.855 | 0.879 |
| Macro avg | 0.879 | 0.882 | 0.880 |
| Accuracy | | | 92.51% |
## Unseen Viewpoints Test Set
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Lateral lying left | 0.847 | 0.870 | 0.858 |
| Lateral lying right | 0.867 | 0.859 | 0.863 |
| Sitting | 0.746 | 0.788 | 0.766 |
| Standing | 0.959 | 0.984 | 0.972 |
| Sternal lying | 0.875 | 0.821 | 0.847 |
| Macro avg | 0.859 | 0.864 | 0.861 |
| Accuracy | | | 91.07% |
---
## Seen vs. Unseen Viewpoint Splits
The dataset is split to evaluate viewpoint generalisation:
Seen-VP test set: Images from the same camera positions that appear in training. Tests in-distribution performance.
Unseen-VP test set: Images from camera positions held out from training. Tests how well the model generalises to new camera angles.
The large unseen-VP test set (11,708 instances) provides a realistic evaluation of cross-camera generalisation.
---
## Data Collection
Images were collected from pig barn environments using ceiling- and wall-mounted cameras. Pigs were annotated with bounding boxes and posture labels. Camera calibration parameters were estimated using the PnP algorithm applied to 4 manually annotated pen-floor corner points per camera, without requiring physical measurement of the pen dimensions.
---
## License
This dataset is released under the Creative Commons Attribution 4.0 (CC BY 4.0) license.
---
## Citation
If you use this dataset, please cite: