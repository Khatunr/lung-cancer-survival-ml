**Assignment 2 — Data Acquisition and Preprocessing**

**Course:** Data Acquisition & Data Preparation

**Author:** Rumana Khatun

**Date:** 22SEP2026

## Project Overview

This repository contains the data acquisition and preprocessing pipeline for a
capstone project that predicts individual lung cancer patient survival outcomes
using machine learning. Patients are classified into two survival categories:

- **Short-Term Survivor:** survival < 60 months
- **Long-Term Survivor:** survival ≥ 60 months

The cleaned dataset produced here will be used in the subsequent modeling stage.

---

## Dataset

- **Source:** [Kaggle — Lung Cancer Prediction](https://www.kaggle.com/datasets/rashadmammadov/lung-cancer-prediction)
- **Author:** Rashad Mammadov
- **Size:** 23,658 rows × 38 columns
- **Cleaned file:** `data/lung_cancer_cleaned.csv`

The dataset contains demographic, clinical, treatment, comorbidity, and
laboratory variables for lung cancer patients.

---

## Preprocessing Summary

1. **Data quality assessment** — Checked missing values, duplicates, and data types.
   Result: 0 missing values, 0 duplicates, no data type issues.

2. **Summary statistics** — Applied describe().T to all numerical variables.
   Result: All values fall within plausible ranges.

3. **Missing values** — Analysis only; no treatment performed.
   Result: No imputation required.

4. **Outlier detection** — Applied the IQR method (1.5 × IQR rule) to all numerical variables.
   Result: 0 outliers detected.

5. **Target creation** — Derived the binary survival outcome from Survival_Months.
   Result: Survival_Months < 60 → Short-Term (11,710 patients);
   Survival_Months ≥ 60 → Long-Term (11,948 patients).

6. **Cleaning** — Removed Patient_ID and Survival_Months from the predictor set.
   Result: Avoids ID noise and target leakage.

7. **Train/Test split** — 80/20 stratified split with random_state=42.
   Result: Split performed before encoding and scaling.

8. **Encoding** — Applied one-hot encoding (drop_first=True) fitted on the training set only.
   Result: Test set columns aligned to training schema.

9. **Scaling** — Applied StandardScaler fitted on the training set only.
   Result: Same transformation applied to both training and test sets.

10. **Save** — Wrote the final cleaned dataset to disk.
    Result: Single file saved as data/lung_cancer_cleaned.csv.

---

## How to Reproduce

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/lung-cancer-survival-ml.git
cd lung-cancer-survival-ml
2. Install dependencies
bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
3. Launch Jupyter
bash
jupyter notebook
4. Open the notebook
Navigate to notebooks/Assignment_2_Data_Preprocessing.ipynb and run all cells
top to bottom.

The cleaned dataset will be written to data/lung_cancer_cleaned.csv.

Methodology Notes
Avoiding Target Leakage
Survival_Months was used to construct the target Survival_Group. It was
removed from the predictor set to prevent the model from being trained on
information that would not be available at prediction time.

Split Before Preprocessing
The train/test split was performed before encoding and scaling.
Transformers were fitted on the training set only and applied to the test set.
This prevents information leakage from the test set into the training pipeline.

No Imputation Was Needed
All 38 original variables contained zero missing values. No imputation was
performed because fabricating missing-data treatment would be methodologically
incorrect.

No Outliers Were Removed
The IQR method identified zero outliers across all numerical variables.
No records were removed.

License
Code and notebook: MIT License

Dataset: Subject to the original Kaggle license. See the Kaggle page for terms.

Acknowledgments
Dataset provided by Rashad Mammadov via Kaggle.
