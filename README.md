 ----

# Mechanical Testing of Alloys – MPEA Dataset Analysis

This project explores the correlation between **mechanical properties** (like hardness, yield strength, and tensile strength) and **microstructural features** of various multi-principal element alloys (MPEAs). The workflow includes data cleaning, exploratory data analysis, feature engineering, and the use of machine learning models for property prediction.

> 🔗 [**Full Report**](https://docs.google.com/document/d/1btxtT-SmBUF6v-7wH-NZ2UH2jyR-tcQuggAdtP4Mabs/edit?tab=t.0#heading=h.8ujurwn0h7gp)

----

## Objective

To perform tensile or hardness tests on different alloys and correlate their mechanical properties with microstructural features.  
This study utilizes the **MPEA (Multi-Principal Element Alloys)** dataset containing experimental data on various alloy compositions, mechanical properties (HV, YS, UTS), processing methods, and microstructural classifications.

---

## Project Structure

```
├── cleaned_mpea_data.csv              # Preprocessed dataset (cleaned)
├── encoded_mpea_data.csv              # One-hot encoded dataset for modeling
├── best_rf_model.pkl                  # Best trained Random Forest model
├── best_model_weights.pt              # Saved model weights (Random Forest or Neural Network)
├── mechanical-testing-of-alloys.ipynb # Jupyter Notebook (all code)
└── README.md                          # Project overview
```

----

## Methodology

### 1. **Data Loading**
- Dataset: `/kaggle/input/mpea-dataset/MPEA_dataset.csv`
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `PyTorch`, `SHAP`

### 2. **Data Cleaning**
- Dropped columns with >70% missing values.
- Filled numeric columns with **median** and categorical with **mode**.
- Dropped rows missing **essential columns** like `FORMULA`, `PROPERTY: HV`, etc.

### 3. **Feature Engineering**
- One-hot encoding for categorical columns:
  - `FORMULA`, `PROPERTY: Microstructure`, `PROPERTY: Processing method`, `PROPERTY: BCC/FCC/other`
- Dropped non-essential columns like reference metadata.

### 4. **Data Preprocessing for Microstructure**
- Mapped microstructure types to numeric values (`FCC=0`, `BCC=1`, etc.)
- Visualized correlations and distributions.

### 5. **Model Training & Evaluation**
Two regression models were trained to predict `PROPERTY: HV`:

#### Random Forest Regressor
- Hyperparameter tuning with `GridSearchCV`
- Used best estimator from grid search

#### Neural Network (PyTorch)
- Three-layer feedforward architecture
- Trained for 100 epochs using MSE loss and Adam optimizer

----

## Model Evaluation

| **Metric**              | **Random Forest**       | **Neural Network**       | **Better Model**     |
|------------------------|-------------------------|---------------------------|-----------------------|
| **R² Score**            | 0.0247                  | -0.2591                  |  Random Forest       |
| **Mean Absolute Error** | 67.15                   | 84.12                    |  Random Forest       |
| **Root Mean Square Error** | 109.75              | 124.70                   |  Random Forest       |

> The **Random Forest** model consistently outperformed the Neural Network and was saved for deployment.

---

## Files Included

| File Name                  | Description                                               |
|---------------------------|-----------------------------------------------------------|
| `mechanical-testing-of-alloys.ipynb` | Full code notebook for preprocessing, analysis, and modeling |
| `cleaned_mpea_data.csv`   | Cleaned version of the raw dataset                        |
| `encoded_mpea_data.csv`   | One-hot encoded dataset ready for modeling                |
| `best_rf_model.pkl`       | Best performing Random Forest model (via GridSearchCV)    |
| `best_model_weights.pt`   | Model metadata (model type, architecture/hyperparams)     |

---

## Visualizations & SHAP

- Distribution plots of target variables (`HV`, `YS`, `UTS`)
- Correlation heatmaps between numeric variables
- Boxplots of Microstructure vs. Hardness
- SHAP summary plots for feature importance (Random Forest)

---

## Libraries Used

- `pandas`, `numpy` for data manipulation  
- `matplotlib`, `seaborn` for visualization  
- `scikit-learn` for ML modeling  
- `PyTorch` for building the neural network  
- `SHAP` for model explainability  

---

## Conclusion

This project demonstrates a complete data science pipeline applied to material science. By integrating preprocessing, visualization, ML modeling, and interpretation, we identify key microstructural factors influencing mechanical performance. The Random Forest model proved most effective for predicting hardness (HV).

---

