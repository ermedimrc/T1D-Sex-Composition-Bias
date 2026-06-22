# T1D-Sex-Composition-Bias

Source code accompanying the Bachelor's Thesis:

**Demographic Bias in Blood Glucose Prediction: Evaluating Sex Effects in Type 1 Diabetes Forecasting**

**Author:** Mehdi Iabouten  
**Institution:** University of Granada (UGR)  
**Degree:** Bachelor's Degree in Computer Engineering  
**Year:** 2026

## Overview

This repository contains the source code used to investigate the impact of training-set sex composition on the performance and fairness of LSTM-based blood glucose prediction models for individuals with Type 1 Diabetes (T1D).

The study evaluates whether varying the proportion of male and female patients in the training set affects predictive performance on male and female test populations. Experiments were conducted on three Continuous Glucose Monitoring (CGM) datasets spanning controlled clinical trials and real-world clinical practice:

- REPLACE-BG
- DiaTrend
- T1DiabetesGranada

A total of eleven sex-proportion configurations were evaluated, ranging from 100% male / 0% female to 0% male / 100% female.

## Main Findings

- Training-set sex composition has no clinically meaningful impact on prediction performance.
- Performance remains remarkably stable across all eleven configurations.
- A persistent sex-related performance gap exists, but its direction depends on the dataset.
- The same gap is present in a sex-agnostic persistence baseline, suggesting that it originates from intrinsic differences in glycaemic dynamics rather than demographic imbalance in the training data.

## Installation

### Using Conda (recommended)

```bash
conda env create -f environment.yml
conda activate tf15
```

### Using pip

```bash
pip install -r requirements.txt
```

## Experimental Pipeline

The workflow implemented in this repository follows the methodology described in the thesis:

1. Dataset preprocessing and harmonisation.
2. Sliding-window generation.
3. Patient-level fold creation.
4. Sex-proportion sampling.
5. LSTM training and evaluation.
6. Statistical analysis.
7. Figure and table generation.

## Data Availability

Patient data are not distributed through this repository.

In accordance with the data-sharing restrictions of the original datasets, the following materials are excluded:

- Raw CGM data
- Processed datasets
- Generated windows
- Cross-validation folds
- Any patient-derived artefacts

Researchers with authorised access to the original datasets may reproduce the complete experimental pipeline using the source code provided in this repository.

## Citation

If you use this repository, please cite:

> Iabouten, M. (2026). *Demographic Bias in Blood Glucose Prediction: Evaluating Sex Effects in Type 1 Diabetes Forecasting*. Bachelor's Thesis, University of Granada.

## License

This project is released under the MIT License.
