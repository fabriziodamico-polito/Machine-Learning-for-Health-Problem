# Skin-Lesion Segmentation - Mole Border Detection

## Objective

Build an unsupervised image-processing pipeline that proposes a skin-lesion region and extracts its border from a dermoscopic image. It demonstrates how clustering and spatial filters can transform raw pixels into a candidate contour for later analysis.

## Data

The repository contains 56 JPEG images organized as `low_risk`, `medium_risk` and `melanoma`. The original source and redistribution terms are not documented in the available project material; see [DATASETS.md](../DATASETS.md). These folder labels are not used to train or clinically validate the segmentation algorithm.

## Pipeline

```text
RGB image -> grayscale -> K-Means with three intensity clusters
-> select darkest cluster -> DBSCAN spatial grouping
-> choose cluster nearest the image center -> crop
-> median filter -> Sobel gradients -> border overlay
```

| Component | Role |
| --- | --- |
| K-Means | Separates pixels into coarse intensity groups |
| DBSCAN | Groups candidate dark pixels by spatial density |
| Center heuristic | Selects one candidate lesion region |
| Median filter | Reduces isolated mask noise |
| Sobel filters | Extract the proposed border |

## Interpretation and limitations

The output is a qualitative candidate segmentation. The repository contains no expert reference masks, so accuracy, sensitivity and border error cannot be measured. Hair, rulers, off-center lesions, uneven illumination and multiple dark regions can break the center-and-intensity heuristics. The pipeline is educational and must not be used for melanoma screening or diagnosis.

## Run

```bash
cd 04_mole_segmentation
python mole_segmentation.py
```

| File | Description |
| --- | --- |
| `mole_segmentation.py` | Segmentation and border-extraction pipeline |
| `data/images/` | The 56 dermoscopic input images |
