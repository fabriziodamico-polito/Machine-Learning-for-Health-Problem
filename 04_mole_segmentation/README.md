# Skin Lesion Segmentation — Mole Border Detection

## Objective

Develop an **image processing pipeline** to automatically segment skin moles from dermoscopic images and extract their borders. This is a critical step in computer-aided diagnosis of melanoma, where asymmetry and border irregularity are key diagnostic features.

## Dataset

| Property | Value |
|----------|-------|
| **Type** | Dermoscopic images (RGB, JPEG) |
| **Categories** | `low_risk`, `medium_risk`, `melanoma` |
| **Total Images** | 56 |

## Techniques

| Method | Description |
|--------|-------------|
| **Grayscale Conversion** | RGB to single-channel intensity |
| **K-Means Clustering** | Color quantization into 3 clusters to separate skin, mole, and background |
| **DBSCAN** | Density-based spatial clustering to identify the mole region among dark pixel groups |
| **Median Filtering** | Spatial smoothing to remove noise from the segmentation mask |
| **Sobel Edge Detection** | Gradient-based border extraction using horizontal and vertical Sobel kernels |

## Pipeline

```
Load Image → Grayscale → K-Means (3 clusters)
→ Select Darkest Cluster → DBSCAN (spatial grouping)
→ Select Cluster Closest to Image Center → Crop
→ Median Filter (smoothing) → Sobel Filters → Border Overlay
```

## Key Results

- The pipeline successfully segments moles across all risk categories
- **DBSCAN** effectively separates the true mole from artifacts (hair, ruler marks) by leveraging spatial density
- The **Sobel-based border** accurately delineates the mole contour for further morphological analysis
- Cluster selection based on proximity to image center provides a robust heuristic

## How to Run

```bash
cd 04_mole_segmentation
python mole_segmentation.py
```

## Files

| File | Description |
|------|-------------|
| `mole_segmentation.py` | Main segmentation pipeline |
| `mole_segmentation.ipynb` | Jupyter notebook with visual outputs |
| `data/images/` | Dermoscopic images (56 samples) |
