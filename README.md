# Predicting Recruitment Rate (RR) in Clinical Trials

## Overview
This project focuses on predicting the Study Recruitment Rate (RR) for clinical trials using advanced machine learning models and feature engineering techniques. Recruitment Rate is a critical metric in the drug development process, and accurate predictions can help optimize recruitment efforts and ensure timely completion of trials.

---

## Methodology

### Data Preprocessing
- Imputed missing values and standardized numerical data.
- Removed non-alphanumeric characters from textual columns.
- Applied one-hot encoding for categorical features.
- Converted date columns into durations to capture temporal trends.
- Dropped columns with excessive missing values and irrelevant features.

### Feature Construction
- Generated embeddings from textual data using **BioBERT**.
- Combined textual embeddings with numerical features to create a robust feature set.

### Model Training and Evaluation
- Used **Gradient Boosting Machine (GBM) Regressor** for training.
- Performed hyperparameter tuning using **Bayesian Optimization**.
- Evaluated the model using:
  - **Root Mean Square Error (RMSE)**
  - **Mean Absolute Error (MAE)**
  - **R² Score**

### Visualization
- Plotted distributions of the target variable and residuals.
- Used **SHAP values** to interpret model predictions and rank feature importance.

---

## Framework/Tools Used
- **Transformers**: Generated textual embeddings with BioBERT.
- **PyTorch**: Accelerated computations using GPU.
- **Scikit-learn**: Used for training and evaluating the Gradient Boosting Regressor.
- **NumPy**: Facilitated numerical computations.
- **Pandas**: Handled data preprocessing and management.
- **Bayesian Optimization**: Tuned hyperparameters using the `bayes_opt` library.
- **Matplotlib & Seaborn**: Created data visualizations.
- **Google Colab**: Used a GPU-enabled environment for training.
- **SHAP**: Explained model predictions and analyzed feature importance.

---

## Results
- **Root Mean Square Error (RMSE)**: 0.30
- **Mean Absolute Error (MAE)**: 0.0791
- **R² Score**: 0.45

### Key Insights
- **Duration of Trial**, **Enrollment**, and **Primary Completion Duration of Trial** were the most influential features.
- **SHAP visualizations** highlighted feature importance and enhanced model explainability.

---

## Challenges & Limitations

### External Factors
- Lack of location and sponsor-specific data could limit prediction accuracy.

### BioBERT Limitations
- Moderate performance in Named Entity Recognition (NER) tasks compared to larger LLMs.

### Skewed Recruitment Rates
- Overfitting and skewness in the Recruitment Rates remain challenging to address.

### Temporal Dynamics
- Techniques like **Temporal Fusion Transformers (TFT)** could better model time-dependent patterns.

---

## Next Steps
1. Use advanced large language models (e.g., GPT-4, Llama-3) for richer textual embeddings.
2. Implement reinforcement learning for dynamic feature selection.
3. Explore hyperparameter tuning using techniques like Hyperband.
4. Develop a continuous learning framework to adapt to new data.
---

## References
- Inhyuk Lee et al., "BioBERT: A Pre-trained Biomedical Language Representation Model for Biomedical Text Mining," *Bioinformatics*, Volume 36, Issue 4, February 2020.
- Corpus Publishers, "Optimizing Patient Recruitment for Clinical Trials: A Hybrid Classification Model and Game-Theoretic Approach for Strategic Interaction," *IEEE Access*, vol. 12, 2024.
- Additional references available in the project documentation.
---

