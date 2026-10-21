# Lung Cancer Survival Group Prediction — Assignment 4

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

