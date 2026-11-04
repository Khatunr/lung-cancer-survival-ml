
# Lung Cancer Survival Prediction (Machine Learning)

## Project Overview
This repository contains the baseline machine learning models for predicting lung cancer survival groups (Long-Term vs. Short-Term) based on the Kaggle Lung Cancer Prediction dataset (23,658 records). 

The goal of this project is to apply classification algorithms and evaluate their performance in a healthcare context. 

## Dataset
* **Source:** Kaggle - Lung Cancer Prediction Dataset (Mammadov, 2025).
* **Target Variable:** `Survival_Group` (Binary classification: Long-Term ≥ 60 months, Short-Term < 60 months).
* **Class Balance:** ~50.5% Long-Term, ~49.5% Short-Term (Well balanced).

## Baseline Modeling Workflow
1. **Data Cleaning:** Removed ID columns, dropped `Survival_Months` to prevent target leakage, and removed redundant raw text columns that were already one-hot encoded.
2. **Preprocessing:** Utilized `ColumnTransformer` and `Pipeline` to ensure no data leakage. Numeric features were imputed (median) and standardized; categorical features were imputed (most frequent) and one-hot encoded.
3. **Train/Test Split:** 80/20 stratified split to maintain class proportions.
4. **Models Evaluated:**
   * Logistic Regression
   * Decision Tree (max_depth=8, min_samples_leaf=20)
   * K-Nearest Neighbors (n_neighbors=15)
   * *Additional:* Random Forest (n_estimators=200, max_depth=10)
5. **Evaluation Metrics:** Accuracy, Macro Precision/Recall/F1, ROC-AUC, and 3-fold stratified cross-validation.

## Key Findings & Limitations
All models (including Random Forest) achieved an accuracy and Macro-F1 of approximately **0.50** (chance level). 

A diagnostic check revealed that the proportion of Long-Term survivors was ~50% across all Cancer Stages (Stage I to Stage IV). In real clinical data, Stage IV patients should have much lower survival rates. This absence of biological signal strongly suggests the dataset is **synthetic**, meaning no predictive pattern exists for the models to learn.

## How to Run
1. Clone this repository.
2. Ensure dependencies are installed: `pip install pandas numpy scikit-learn matplotlib`.
3. Place the `cleaned_dataset_assignment3.csv` file in the `data/` directory.
4. Run the Jupyter Notebook `Assignment_4_Baseline_Models.ipynb`.

**Assignment 2 — Data Acquisition and Preprocessing**

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
   
# Assignment 3: Exploratory Data Analysis (EDA)

## Project Overview
This project explores a cleaned lung cancer dataset to understand patient characteristics, examine how variables relate to the `Survival_Group` target, create additional features, and test basic feature-selection and dimensionality-reduction methods.

The target has two classes: `Short-Term` and `Long-Term` survival. The analysis is exploratory and does not establish causal relationships.

## Files
- `Assignment_3_EDA.ipynb` — notebook containing dataset overview, descriptive statistics, visualizations, correlation analysis, business insights, feature engineering, Variance Threshold, PCA, and reflection.
- `EDA_Report.pdf` — concise report summarizing the main findings.
- `cleaned_dataset_assignment3.csv` — updated dataset after feature engineering (include this file in the repository).
- `data/lung_cancer_cleaned.csv` — source cleaned dataset, if required by the notebook's relative path.

## Dataset
The notebook's recorded starting dataset contains 23,658 rows and 48 columns. The survival outcome is nearly balanced:
- Long-Term: 11,948 patients (50.50%)
- Short-Term: 11,710 patients (49.50%)

Selected descriptive statistics:
- Mean age: approximately 54.44 years (standard deviation 14.40).
- Mean tumor size: approximately 55.38 mm (standard deviation 26.00).
- Mean systolic blood pressure: approximately 134.46.
- Mean diastolic blood pressure: approximately 84.48.

## Feature Engineering
The notebook creates and visualizes five features:

1. **`Age_Group`** — groups age into 30–49, 50–64, and 65–79 categories. This makes it easier to compare survival distributions across age bands.
2. **`Tumor_Size_Group`** — groups tumor size into Small (up to 40 mm), Medium (40–70 mm), and Large (over 70 mm). This supports categorical comparisons of tumor size and survival.
3. **`Comorbidity_Count`** — sums seven yes/no comorbidity indicators. This gives a simple count of recorded comorbidities per patient.
4. **`Treatment_Count`** — sums three treatment indicator columns. It is intended as a compact summary of treatment indicators; its interpretation as a count of treatments depends on whether the underlying columns can be simultaneously active.
5. **`Blood_Pressure_Difference`** — calculates systolic blood pressure minus diastolic blood pressure, creating a single feature representing the difference between the two readings.

The notebook includes plots comparing each engineered feature with `Survival_Group`.

## EDA and Correlation Findings
The notebook contains more than 10 visualizations covering survival-group distribution, age, tumor size, performance status, cancer stage, treatment type, gender, smoking history, correlation patterns, and the five engineered features.

The outcome classes are balanced. Mean age and mean tumor size are very similar across survival groups. The correlation analysis found no predictor pairs exceeding an absolute Pearson correlation of 0.70 in the saved notebook output. Correlations between individual numeric predictors and the encoded survival target were small, so no single numeric predictor showed a strong linear relationship with the target.

### Business Insights
1. The survival target is balanced, with 50.50% Long-Term and 49.50% Short-Term patients.
2. Mean ages for Long-Term and Short-Term groups are approximately 54.5 and 54.4 years, respectively.
3. Mean tumor sizes are approximately 55.6 mm for Long-Term and 55.1 mm for Short-Term patients.
4. The stage visualization shows broadly similar outcome distributions across stages in this dataset; stage alone does not strongly separate the classes visually.
5. Individual numeric predictor correlations with the target are weak, suggesting that combined patterns may be more useful than relying on a single feature.

These are observations from this dataset, not clinical conclusions or evidence of causation.

## Variance Threshold and PCA
- **Variance Threshold:** Using a threshold of 0.01 retained all 50 predictors in the notebook output.
- **PCA:** The recorded analysis required 39 components to retain 90% of variance and 42 components to retain 95% of variance. Moving from 50 predictors to 39 components is a 22% reduction in feature count.
- **Logistic Regression comparison (values currently recorded in the saved notebook):**

| Approach | Number of features | Accuracy |
|---|---:|---:|
| Original features | 50 | 49.79% |
| PCA features (90% variance) | 39 | 49.64% |

The recorded accuracies are close to 50%, so this baseline model did not demonstrate strong predictive performance. PCA reduced the number of inputs with a small change in accuracy, but the transformed components are less directly interpretable.

**Before submission:** If the PCA comparison was rerun using `PCA(n_components=0.90)` fitted on the training set, update the feature count and accuracy in this README and in `EDA_Report.pdf` to match the new notebook output. The notebook currently available in this workspace still has the older comparison output.

## How to Run the Notebook
1. Clone or download this repository.
2. Ensure the notebook's input CSV is in the expected relative location. The original notebook reads `../data/lung_cancer_cleaned.csv`; either preserve that folder structure or update the path to match the repository.
3. Install the libraries used by the notebook if they are not already available, including `pandas`, `numpy`, `matplotlib`, `seaborn`, and `scikit-learn`.
4. Open `Assignment_3_EDA.ipynb` in Jupyter Notebook, JupyterLab, or a compatible notebook environment.
5. Run the notebook cells from top to bottom. Confirm the final PCA comparison table and use those latest values in this README and PDF report.

## Limitations
This is an exploratory analysis and a basic baseline-model comparison. Correlation and visual patterns do not imply causation. More robust model evaluation, additional metrics, and further investigation would be needed before using the model for prediction.

## Assignment 4: Baseline Modeling
## Objective
Train and compare baseline supervised binary classifiers to predict `Survival_Group` (`Short-Term` or `Long-Term`) using the cleaned dataset from Assignment 3.

## Dataset
- File: `cleaned_dataset_assignment3.csv`
- Rows and columns in the uploaded file: 23,658 rows × 57 columns.
- Target distribution:
  - Long-Term: 11,948 (50.50%)
  - Short-Term: 11,710 (49.50%)
- Missing values in the uploaded CSV: 0
- Duplicate rows in the uploaded CSV: 0

## Methods
- Train/test split: 80% training / 20% test, stratified by target, `random_state=42`.
- Numeric preprocessing: median imputation and standard scaling.
- Categorical preprocessing: most-frequent imputation and one-hot encoding.
- Preprocessing is inside scikit-learn pipelines to avoid fitting transformations on the test set.
- Raw `Stage`, `Treatment`, `Gender`, and `Smoking_History` columns are excluded because corresponding encoded columns already exist.
- Baseline models: Logistic Regression; Decision Tree (`max_depth=8`, `min_samples_leaf=20`); K-Nearest Neighbors (`n_neighbors=15`, distance weighting).

## Held-out test results
| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Long-Term F1 | Short-Term F1 | Long-Term ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.4989 | 0.4981 | 0.4982 | 0.4955 | 0.5370 | 0.4541 | 0.5027 |
| Decision Tree | 0.4939 | 0.4909 | 0.4921 | 0.4760 | 0.5727 | 0.3794 | 0.4984 |
| K-Nearest Neighbors | 0.4934 | 0.4934 | 0.4934 | 0.4934 | 0.5001 | 0.4866 | 0.4979 |

## Interpretation
The strongest macro-F1 among these baseline runs was achieved by **Logistic Regression** (0.4955). Nevertheless, test accuracy and ROC-AUC are close to 0.50, suggesting that the current feature set provides little useful separation between the two labels in this experiment. These models should not be treated as clinically useful or validated predictors.

## Run the notebook
1. Place `Assignment_4_Baseline_Models.ipynb` in the `Assignment_4` folder.
2. Keep `cleaned_dataset_assignment3.csv` in the same folder as the notebook, or update `DATA_PATH` in the notebook.
3. Install dependencies if needed: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `jupyter`.
4. Run all cells from top to bottom.

## Limitations and future work
- Check how `Survival_Group` was created and whether it is meaningful and reliable.
- Verify that each predictor would be available at the intended prediction time.
- Compare with a majority-class baseline and tune hyperparameters using cross-validation.
- Examine class-specific errors and validate on independent data if available.
- These are educational baseline models, not a clinical decision-support system.

## Reproducibility
Random seed: 42. Test size: 20%. The results in this README were generated from the uploaded CSV with the settings listed above. Re-run the notebook if the dataset or settings change.

## Assignment 5: Advanced Modeling, Optimization, and Evaluation

### Optimization (Model Tuning)
- **Algorithms:** Implemented Random Forest and Gradient Boosting ensembles.
- **Hyperparameter Tuning:** Used `GridSearchCV` with 5-fold Stratified Cross-Validation to optimize the Random Forest. We tuned `n_estimators`, `max_depth`, and `min_samples_split`.

### Evaluation and Explainability
- **Cross-Validation:** 5-fold CV confirmed stable performance, though still near the ~50% chance level.
- **Explainability (SHAP):** Generated SHAP Summary Plots to interpret feature importance. The results showed that no features had a strong impact on predictions.
- **Data Diagnostic:** A check on Cancer Stage vs. Survival revealed that Stage I and Stage IV both had ~50% survival rates. This proves the dataset is synthetic and lacks biological signal.
- **Model Selection:** No model was selected for deployment due to chance-level performance.

### How to Run Assignment 5
1. Navigate to `notebooks/Assignment_5_Advanced_Modeling.ipynb`.
2. Ensure `shap` is installed (`pip install shap`).
3. Run all cells to reproduce the tuning, cross-validation, and SHAP plots.
4. View `model_comparison.csv` and `shap_summary.png` in the `results/` folder.

