# Cryptopond Bot Detection

![Python](https://img.shields.io/badge/python-3.12-blue) ![Pandas](https://img.shields.io/badge/pandas-1.6.2-orange) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-green) ![LightGBM](https://img.shields.io/badge/LightGBM-4.0.0-purple)

## Overview

This repository contains a **bot detection pipeline** for wallet activity on blockchain networks. The main goal is to identify wallets likely operated by bots using **anomaly detection**, **heuristic rules**, and **supervised learning refinement**.

The workflow includes:

1. Data loading and preprocessing
2. Feature engineering for wallet activity
3. Exploratory data analysis (EDA) and visualization
4. Heuristic labeling rules to detect bot-like behavior
5. Multiple anomaly detection models (IsolationForest, LocalOutlierFactor, OneClassSVM)
6. Ensemble and aggressive labeling strategies
7. Optional LightGBM supervised refinement using pseudo-labels
8. Exporting a clean dataset with `address` and `bot` labels

---

## Repository Structure

```text
cryptopond-bot-detection/
│
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   └── dune_results.csv        # Original dataset (not pushed, in .gitignore)
├── notebooks/
│   └── bot_detection_dune.ipynb
├── src/
│   ├── botdetec.py             # Initial EDA + heuristic detection
│   ├── botv2.py                # Aggressive detection & anomaly ensemble
│   └── botfinish.py            # Export address + bot labels (~10% bot)
└── results/
    └── bot_detected_dataset.csv  # Output dataset
```

---

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/your-username/cryptopond-bot-detection.git
cd cryptopond-bot-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

1. Place the dataset in `data/dune_results.csv`.
2. Run the initial detection script:

```bash
python src/botdetec.py
```

3. Run the aggressive anomaly detection and ensemble:

```bash
python src/botv2.py
```

4. Export the final bot-labeled dataset:

```bash
python src/botfinish.py
```

The final dataset `bot_detected_dataset.csv` will contain:

| address   | bot |
| --------- | --- |
| 0x1234... | 1   |
| 0xabcd... | 0   |

---

## Features & Methodology

* **Activity features**: active days, trade frequency, volume, unique tokens
* **Heuristic rules**: initial labeling based on thresholds
* **Anomaly detection**: IsolationForest, LOF, OneClassSVM
* **Ensemble**: combining multiple detectors for robust predictions
* **Supervised refinement**: optional LightGBM pseudo-labeling
* **Visualization**: PCA projections, EDA plots, label distribution

---

## Notes

* The `data/` folder is ignored in `.gitignore` to avoid pushing raw datasets.
* The aggressive detection can label up to ~80% of wallets as bots, depending on chosen thresholds.
* Scripts are designed to be modular and reproducible for future competitions.

---

## Citation / Reference

If you use this repository for a competition or research, please cite:

```
Achan, "Cryptopond Bot Detection Pipeline", GitHub repository, 2025.
```

---

