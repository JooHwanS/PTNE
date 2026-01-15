# PTNE – Analysis Code for Skin–Electrode Contact and EEG Window-Length Evaluation

This repository provides executable **Jupyter Notebook / Python scripts** used to generate the quantitative analyses and figures reported in the manuscript:

**“Ultra-miniaturized, Tattoo-like Microneedle Electrodes for Continuous Mastoid EEG Monitoring”**

The code is organized to allow reviewers to reproduce the **core mechanical/contact analyses** with a minimal setup. An additional **optional EEG classification analysis (TCN)** is provided for completeness.

---

## 1. System Requirements

- **Operating system**: Windows 11  
- **Python**: 3.13 (Anaconda distribution)
- **Execution environment**: Jupyter Notebook
- **Hardware**: CPU only (no GPU required)

All analyses were developed and tested on a standard desktop computer.

---

## 2. Repository Structure

PTNE/
│
├─ Skin–Electrode Contact.py
├─ Sweep_contact_strain.py
├─ TCN_EEG_WindowSweep.py
│
├─ line_image/
│ ├─ line1.tif
│ ├─ line2.tif
│ ├─ line3.tif
│ └─ line4.tif
│
└─ nback_data
├─ 1/
│ ├─ 0back.txt
│ ├─ 1back.txt
│ ├─ 2back.txt
│ ├─ 3back.txt
│ └─ 4back.txt
├─ 2/
│ ├─ 0back.txt
│ ├─ 1back.txt
│ ├─ 2back.txt
│ ├─ 3back.txt
│ └─ 4back.txt
├─ 3/
│ ├─ 0back.txt
│ ├─ 1back.txt
│ ├─ 2back.txt
│ ├─ 3back.txt
│ └─ 4back.txt
├─ 4/
│ ├─ 0back.txt
│ ├─ 1back.txt
│ ├─ 2back.txt
│ ├─ 3back.txt
│ └─ 4back.txt


---

## 3. Core Analyses

### 3.1 Skin–Electrode Contact Analysis

**File**  
`Skin–Electrode Contact.py`

**Description**  
This script computes the geometric contact between a skin surface profile and a flexible electrode by:
- Skeletonizing binary line images
- Extracting the longest geodesic path
- Resampling in micrometer units
- Constructing an upper electrode envelope using valley-bridging and minimum curvature radius constraints
- Determining contact via a gap-only criterion with local minima, hysteresis, and post-processing

**Input**  
- One line-profile image (e.g. `line_image/line1.tif`)

**Output (automatically generated)**  
- CSV: contact profile and gap metrics  
- JSON: parameter summary and contact ratio  
- PNG/SVG: profile plots and contact maps  

Outputs are written to a newly created directory (e.g. `out_follow_jupyter_gap/`).

---

### 3.2 Contact Ratio vs. Strain Sweep (Material-wise)

**File**  
`Sweep_contact_strain.py`

**Description**  
This script performs a parameter sweep across:
- Multiple skin profiles (`line1.tif` – `line4.tif`)
- Multiple electrode materials (PDMS, Parylene, PI, Rigid)
- Valley-bridging widths and minimum curvature radii

Contact ratio and effective electrode strain are computed and summarized relative to a rigid baseline.

**Input**  
- Line images located in `line_image/`

**Output**  
- Per-image CSV files
- Aggregated CSV summaries
- Publication-ready PNG figures (mean ± 95% CI)

Outputs are written to the `out_sweep3/` directory.

---

## 4. Optional Analysis: EEG Window-Length Sweep with TCN

**File**  
`TCN_EEG_WindowSweep.py`

**Description**  
This script evaluates EEG classification performance across different window lengths using:
- Welch PSD features (1–20 Hz)
- Temporal Convolutional Network (TCN)
- Repeated train/test splits with class balancing

The script uses **interactive Jupyter widgets** for file upload and is intended as a demonstration of downstream EEG analysis enabled by the proposed electrodes.

**Input**  
- EEG time-series files (`.txt`) organized by N-back condition in `nback_data/`

**Output**  
- CSV file summarizing accuracy and F1-score statistics

---



## 5. Installation
conda create -n ptne python=3.13 -y 

conda activate ptne 

pip install numpy scipy pandas matplotlib scikit-image pillow tifffile scikit-learn 

For the optional EEG analysis only:

pip install tensorflow keras ipywidgets

Installation typically takes less than 5 minutes on a standard desktop computer.
---

## 6. How to Run

Launch Jupyter Notebook:
jupyter notebook

Open and execute the desired script within Jupyter:
- `Skin–Electrode Contact.py` for single-profile contact analysis
- `Sweep_contact_strain.py` for material-wise strain/contact sweeps
- `TCN_EEG_WindowSweep.py` for optional EEG analysis

All scripts are executed sequentially within the Jupyter environment.

---

## 7. Expected Runtime

- Skin–electrode contact analysis: typically less than 1 minute  
- Material-wise sweep: a few minutes depending on parameter ranges  
- EEG TCN analysis (optional): tens of minutes on CPU

---
