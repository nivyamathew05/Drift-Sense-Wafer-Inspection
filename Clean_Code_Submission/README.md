# Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

**Applied Materials Problem Statement 02 - Phase 1 Submission**  
Organized as part of **SEMICON India / Hackathon 2026**

---

## 📌 Executive Summary

During semiconductor wafer inspection, motion stages experience mechanical drift, thermal expansion, and vibration errors. To revisit an exact microscopic inspection site across different dies, inspection tools must align a high-magnification (**100x**, $1000 \times 1000$ pixels) reference image inside a wider field-of-view low-magnification (**10x**, $1000 \times 1000$ pixels) search image.

**Drift-Sense** delivers a robust, reproducible Python solution featuring:
1. **Literature-Grounded SEM Synthetic Data Generator**: Synthesizes realistic **DRAM** and **FinFET** wafer layout pairs with physical SEM noise (Poisson shot noise, astigmatic spot blur, scan line jitter, dielectric charging streaks, gamma/contrast shifts, scale variations 9:1–11:1, and small rotations 1–2°).
2. **Scale-Aware Sub-Pixel Localizer**: Combines multi-scale normalized cross-correlation (NCC), parabolic sub-pixel peak interpolation, and spatial center-distance tie-breaking to accurately recover target $(x, y)$ coordinates even under extreme periodic pattern ambiguity.
3. **RGB Optical Extension (Bonus)**: Generalization module for 3-channel optical microscope navigation.

---

## 📁 Repository Structure

```text
submission/
├── solution_presentation.pptx       # Mandatory Solution Presentation (Hackathon 2026 Template)
├── solution_presentation.pdf        # Portal Upload PDF Format
├── README.md                        # Project documentation and execution instructions
├── requirements.txt                 # Environment dependencies
├── generate_dataset.py              # Synthetic dataset generator script
├── localize.py                      # Localization evaluation script
├── configs/
│   └── default_config.json          # System configuration file
├── src/
│   ├── __init__.py                  # Package init
│   ├── sem_generator.py             # DRAM & FinFET SEM layout and physics engine
│   ├── localizer.py                 # Multi-scale sub-pixel cross-correlation localizer
│   └── optical_extension.py         # Bonus credit: RGB optical image processor
├── results/
│   ├── dataset_manifest.csv         # Benchmark evaluation manifest (35 pairs)
│   ├── summary_metrics.json         # Aggregate pass rates and runtime statistics
│   ├── evaluation_plots.png         # Accuracy curve, error distribution, and runtime plots
│   └── failure_analysis.png         # Visual failure case analysis with annotations
└── references/
    └── SEM_Literature_Citations.md  # Grounding literature citations (IEEE/SPIE papers)
```

---

## 📐 Coordinate System & Conventions

* **Origin $(0,0)$**: Defined at the **Top-Left** corner of the search image.
* **X-axis**: Increases **Rightward** (from 0 to 999).
* **Y-axis**: Increases **Downward** (from 0 to 999).
* **Output Format**: Predicted target center $(x, y)$ in float pixel coordinates.
* **Tie-Breaking Rule**: When multiple matching candidates fall within 90% of the maximum correlation score (common in highly periodic DRAM/FinFET arrays), the algorithm selects the candidate whose center is **closest to the search image center $(500, 500)$**.

---

## 🚀 Environment Setup & Installation

### Prerequisites
* Python 3.9+ (Tested on Python 3.11 / 3.12)
* Windows / Linux / macOS

### Installation Steps
```bash
# 1. Clone or extract submission repository
cd submission

# 2. Install required dependencies
pip install -r requirements.txt
```

---

## ⚡ Execution Commands

### 1. Generate Synthetic SEM Dataset (35 Benchmark Pairs)
```bash
python generate_dataset.py --num_pairs 35 --output_dir results/dataset --manifest results/dataset_manifest.csv
```

### 2. Run Benchmark Localization & Metric Evaluation
```bash
python localize.py --manifest results/dataset_manifest.csv
```

### 3. Evaluate Single Reference & Search Image Pair
```bash
python localize.py --ref results/dataset/pair_001_ref.png --search results/dataset/pair_001_search.png
```

---

## 📊 Benchmark Results & Performance Summary

Evaluated on 35 randomized synthetic test cases (DRAM & FinFET, noise 0.5–1.5, scale 9:1–11:1, rotation -2° to +2°):

| Metric | Measured Value | Requirement / Target |
| :--- | :---: | :---: |
| **Pass Rate @ 5-pixel threshold** | **100.0%** | $\ge 90\%$ |
| **Pass Rate @ 4-pixel threshold** | **100.0%** | $\ge 85\%$ |
| **Pass Rate @ 2-pixel threshold** | **97.1%** | High Accuracy |
| **Pass Rate @ 1-pixel threshold** | **91.4%** | Ultra Precision |
| **Sub-Pixel Accuracy (<0.5px)** | **85.7%** | Advanced Performance |
| **Mean Euclidean Error** | **0.42 px** | Sub-pixel |
| **Median Euclidean Error** | **0.28 px** | Sub-pixel |
| **Mean Runtime Per Pair** | **38.4 ms** | Real-time (<50ms) |

---

## 🔬 Literature Justification

* **DRAM Geometry**: *Kim et al.* (IEEE TED 2020) - Periodic active cell matrix, wordline/bitline tracks.
* **FinFET Geometry**: *Auth et al.* (IEEE IEDM 2012) - High aspect-ratio 3D silicon fins crossed by poly gates.
* **SEM Poisson Noise & Astigmatism**: *Reimer, L.* (Springer SEM Physics) - Electron counting Poisson noise & anisotropic lens blur.
* **Charging Streaks**: *Cazaux, J.* (J. Appl. Phys. 1999) - Electrostatic decay streaks along fast scan lines.

---

## 💡 Failure Case Analysis & Explainability

In rare cases with extreme scale mismatch (>11:1) combined with severe charging streak noise across a 100% periodic DRAM array, spatial correlation peak values across adjacent array cells can differ by $< 0.01$. Although our **Center-Proximity Tie-Breaking Rule** correctly resolves $> 97\%$ of spatial ambiguities, boundary cells near the outer search image margin can exhibit slight multi-cell offset shifts. This is documented and visualized in `results/failure_analysis.png`.
