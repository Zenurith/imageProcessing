# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **waste/garbage segmentation and detection** project using the TACO dataset. It explores multiple deep learning approaches: RT-DETR (detection), Mask R-CNN (instance segmentation), YOLOv8 (segmentation), Faster R-CNN, and Watershed. The dataset has 9 classes: Bottle, Cigarette, Foam, Glass, Metal, Other, Paper, Plastic, Unlabeled.

## Environment Setup

Python 3.12 + CUDA 12.4. Install in this order:

```bash
py -3.12 -m venv venv
.\venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m ipykernel install --user --name=waste-seg
jupyter notebook
```

## Running Notebooks

Activate `venv` and select the `waste-seg` kernel in Jupyter. Notebooks must be run in order within each folder.

## Architecture

```
1.Data Analysis/        # EDA on COCO-format annotations
2.Preprocessing/        # Annotation splitting, dataset splitting, COCO→YOLO conversion
Model training/         # One notebook per model architecture
archive/dataset_v2/     # COCO-format dataset (train/val/test splits with JSON annotations)
runs/
  mask_rcnn/            # best_mask_rcnn.pth, results JSON, prediction PNGs
  rtdetr/train/         # Ultralytics RT-DETR run; weights/best.pt is the primary trained model
taco_rtdetr.yaml        # Dataset config for Ultralytics (paths, 9 class names)
```

## Trained Models

| Model | Path | Framework |
|---|---|---|
| RT-DETR (best) | `runs/rtdetr/train/weights/best.pt` | Ultralytics |
| Mask R-CNN (best) | `runs/mask_rcnn/best_mask_rcnn.pth` | PyTorch / torchvision |
| YOLOv8-seg | `Model training/yolo26n.pt` (base weights) | Ultralytics |

RT-DETR was trained with `imgsz=640`, `batch=4`, `epochs=100`, AdamW optimizer on GPU 0.

## Dataset

- Format: COCO JSON for Mask R-CNN / standard splits; YOLO format for Ultralytics models
- Config: `taco_rtdetr.yaml` points to `archive/dataset_v2/`
- Preprocessing notebooks convert COCO → YOLO format and handle annotation/image splits
