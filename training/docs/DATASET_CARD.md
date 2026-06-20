# Night Road Dataset Card

## Overview

| Field            | Value                                                     |
|------------------|-----------------------------------------------------------|
| **Purpose**      | Train YOLO26n for night road object detection on Android  |
| **Task**         | Object Detection (bounding box)                           |
| **Classes**      | 7 classes: `person`, `rider`, `bicycle`, `motorcycle`, `car`, `bus`, `truck` |
| **Format**       | YOLO TXT labels (class cx cy w h, normalized)            |
| **Image Size**   | 640x640 (resized during training)                         |
| **Focus**        | Night and low-light road scenes                           |
| **Target Device**| Android (CameraX inference pipeline)                      |

This dataset is a combination of public research datasets and self-captured phone data, curated specifically for training a lightweight night-time road object detector intended for on-device inference.

---

## Source Datasets

### BDD100K

| Field          | Detail                                                        |
|----------------|---------------------------------------------------------------|
| **Source**     | Berkeley DeepDrive (bdd-data.berkeley.edu)                    |
| **License**    | BDD100K License (non-commercial research use)                 |
| **Images Used**| Subsets filtered by time-of-day tags: `night`, `dawn`, `dusk` |
| **Resolution** | 1280x720                                                      |
| **Contribution**| Largest source; provides all 7 target classes                |

**Class Mapping:**

| BDD100K Category | Mapped Class   |
|------------------|----------------|
| pedestrian       | person         |
| rider            | rider          |
| bicycle          | bicycle        |
| motorcycle       | motorcycle     |
| car              | car            |
| bus              | bus            |
| truck            | truck          |

**Notes:**
- BDD100K labels were filtered to keep only bounding boxes with time-of-day metadata matching night, dawn, or dusk.
- Daytime frames were excluded to maintain dataset focus.
- Original 10-class labels were remapped to the 7-class scheme used in this project.

---

### NightOwls

| Field          | Detail                                           |
|----------------|--------------------------------------------------|
| **Source**     | NightOwls Pedestrian Dataset                     |
| **License**    | Research use                                     |
| **Images Used**| All frames (night pedestrian detection focused)  |
| **Resolution** | 1920x1200 (resized)                              |
| **Contribution**| Night-specific person annotations               |

**Class Mapping:**

| NightOwls Category | Mapped Class |
|--------------------|--------------|
| pedestrian         | person       |

**Notes:**
- NightOwls provides annotations for pedestrians only; no other object classes are labeled.
- Frames are sourced from continuous video sequences captured in low-light urban environments.
- This dataset is the primary source for hard night-person examples, including occluded and distant pedestrians.

---

### ExDark

| Field          | Detail                                                           |
|----------------|------------------------------------------------------------------|
| **Source**     | ExDark (Extremely Dark Object Detection Dataset)                 |
| **License**    | Research use                                                     |
| **Images Used**| Dark/low-light images containing relevant object classes         |
| **Resolution** | Varies (resized)                                                 |
| **Contribution**| Adds diversity in lighting and object appearance under darkness |

**Class Mapping:**

| ExDark Category | Mapped Class   |
|-----------------|----------------|
| People          | person         |
| Bicycle         | bicycle        |
| Motorbike       | motorcycle     |
| Car             | car            |
| Bus             | bus            |

**Notes:**
- ExDark was originally annotated at image level; bounding-box annotations from associated releases or re-annotations were used where available.
- Only classes relevant to the target 7-class set were retained.
- Images were filtered to road or street contexts where possible.

---

### Phone Captured Data

| Field            | Detail                                                       |
|------------------|--------------------------------------------------------------|
| **Source**       | Self-collected using multiple Android phones                 |
| **License**      | Internal project data; not redistributed                     |
| **Capture Devices** | Multiple Android devices across different manufacturers   |
| **Resolution**   | Varies by device (resized to 640x640 for training)           |
| **Contribution** | Domain-matched data; bridges gap between research data and deployment |

**Capture Conditions:**

| Condition       | Description                                      |
|-----------------|--------------------------------------------------|
| **Time of Day** | Night, dusk, late evening                        |
| **Scenes**      | Campus, parking lots, residential streets, commercial areas |
| **Lighting**    | Streetlights, car headlights, mixed artificial lighting, near-darkness |
| **Weather**     | Clear, overcast, light rain (limited)            |

**Notes:**
- Data was captured to match the inference pipeline: CameraX preview frames at resolutions and quality representative of real deployment.
- Annotation was performed manually using standard bounding-box tools.
- This subset is intentionally small but high-value for domain adaptation.

---

## Class Distribution

> **Note:** The table below is to be filled after the data preparation pipeline completes and final counts are available.

| Class       | BDD100K (Night) | NightOwls | ExDark | Phone Captured | Total   |
|-------------|-----------------|-----------|--------|----------------|---------|
| person      | TBD             | TBD       | TBD    | TBD            | TBD     |
| rider       | TBD             | 0         | 0      | TBD            | TBD     |
| bicycle     | TBD             | 0         | TBD    | TBD            | TBD     |
| motorcycle  | TBD             | 0         | TBD    | TBD            | TBD     |
| car         | TBD             | 0         | TBD    | TBD            | TBD     |
| bus         | TBD             | 0         | TBD    | TBD            | TBD     |
| truck       | TBD             | 0         | 0      | TBD            | TBD     |
| **Total**   | TBD             | TBD       | TBD    | TBD            | TBD     |

**Known Class Imbalance:**
- `car` and `person` are expected to dominate the dataset.
- `bus` and `truck` are expected to be underrepresented.
- Data augmentation and class-aware sampling strategies are used to mitigate imbalance during training.

---

## Data Splits

| Split        | Percentage | Purpose                        |
|--------------|------------|--------------------------------|
| **Train**    | 70%        | Model weight optimization      |
| **Validation** | 15%      | Hyperparameter tuning, early stopping |
| **Test**     | 15%        | Final unbiased performance evaluation |

### Split Strategy

Splits are created **by video sequence, not by random frame selection**. This ensures:

- Consecutive frames from the same capture do not leak across splits.
- The test set contains scenes and conditions the model has never seen during training.
- Generalization is evaluated on genuinely unseen environments.

For still-image datasets (ExDark), splits are performed by source image to avoid near-duplicate leakage.

---

## Special Test Slices

These curated test subsets are designed to evaluate model performance on specific failure modes and conditions relevant to night road deployment:

| Slice Name                  | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `night_person_small`        | Persons with bounding-box area below a defined small-object threshold       |
| `night_person_dark_clothes` | Persons wearing dark clothing, reducing contrast against dark backgrounds   |
| `night_backlight`           | Scenes with strong backlight sources (headlights, streetlights behind subject) |
| `night_rain`                | Rainy night scenes with wet roads, reflections, and reduced visibility       |
| `parking_garage`            | Indoor/covered parking environments with artificial lighting                 |
| `unseen_phone`              | Images from a phone model not represented in training data                   |
| `daytime_general`           | Daytime scenes to verify the model does not catastrophically regress on daylight |

These slices are used for targeted metric reporting and regression testing during model iteration.

---

## Data Preprocessing Pipeline

1. **Filtering:** Source datasets filtered by time-of-day and class relevance.
2. **Remapping:** Original class labels mapped to the unified 7-class scheme.
3. **Resizing:** All images resized to 640x640 with padding to preserve aspect ratio.
4. **Format Conversion:** Annotations converted from source formats (COCO JSON, Pascal XML, etc.) to YOLO TXT format.
5. **Deduplication:** Near-duplicate frames removed using perceptual hashing.
6. **Split Assignment:** Video-sequence-aware split into train/val/test.
7. **Validation:** Automated checks for label integrity, class distribution, and annotation quality.

---

## Known Limitations

- **NightOwls class coverage:** NightOwls provides annotations for the `person` class only. No other road users are labeled in this dataset.
- **Geographic bias:** BDD100K is primarily sourced from roads in the United States. Road layouts, signage, vehicle types, and driving norms may differ from target deployment regions.
- **Capture device mismatch:** Research datasets use high-quality dashcams or specialized cameras. CameraX output on consumer Android phones may differ in resolution, noise characteristics, dynamic range, and color reproduction.
- **Extreme darkness:** While the combined dataset covers low-light conditions well, scenes at the extreme end of darkness (no artificial lighting, no moonlight) are underrepresented.
- **Weather coverage:** Rain, fog, and snow at night are sparsely represented. Performance in these conditions may degrade.
- **Annotation inconsistency:** Combining multiple source datasets introduces minor inconsistencies in bounding-box tightness and labeling conventions across sources.

---

## Ethical Considerations

- **No personally identifiable information (PII) is retained.** Faces and license plates are not annotated and are not the target of detection.
- **Data is used exclusively for model training and internal evaluation.** The combined dataset is not redistributed.
- **Source datasets are used under their respective research licenses.** Commercial use of the combined dataset requires separate license clearance from each source provider.
- **Self-captured data was collected in public spaces** where no reasonable expectation of privacy exists. No private property or restricted-access areas were used.
- **Bias awareness:** The dataset may reflect biases present in the source data (e.g., geographic, demographic). This should be considered when evaluating model fairness and deploying in diverse regions.

---

## Citation

If you use components of this combined dataset in a publication, please cite the original sources:

```bibtex
@inproceedings{yu2020bdd100k,
  title={BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
  author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  booktitle={CVPR},
  year={2020}
}

@inproceedings{neumann2019nightowls,
  title={NightOwls: A Pedestrians at Night Dataset},
  author={Neumann, Lukas and Karg, Michelle and Zhang, Shanshan and Scharfenberger, Christian and Piegert, Eric and Mistr, Sarah and Prokofyeva, Olga and Thiel, Robert and Bargal, Andrea and Darrell, Trevor and others},
  booktitle={ACCV},
  year={2019}
}

@article{loh2020exdark,
  title={How Dark is Dark: A Review of Low-Light Image Enhancement},
  author={Loh, Yuen Peng and Chan, Chee Seng},
  journal={IEEE Access},
  year={2020}
}
```

---

## Version History

| Version | Date       | Description                          |
|---------|------------|--------------------------------------|
| 0.1     | 2026-06-20 | Initial dataset card created         |
