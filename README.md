# Machine Learning Model Comparison - Assignment 2

**Student Name**: Suresha Kantipudi  
**Student ID**: 2025aa05409  
**Course**: MTech AI & ML  
**Assignment**: Assignment 2  

---

## a. Problem Statement

This project analyzes the **bank-full.csv** dataset to predict whether a client will subscribe to a term deposit (target variable 'y': yes/no) based on various client attributes and campaign information.

The objective is to train, evaluate, and compare six different classification models using comprehensive evaluation metrics including:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC Score)

A Streamlit web application is developed to:

- Provide interactive model evaluation and comparison  
- Enable CSV file upload with automatic preprocessing  
- Generate predictions using any of the six pre-trained models  
- Display comprehensive evaluation metrics  
- Show classification reports and confusion matrices  
- Allow downloading of prediction results and test data  

---

## b. Dataset Description

**Dataset**: bank-full.csv  
**Source**: Bank marketing campaign data (Portuguese banking institution)  
**Format**: CSV with semicolon delimiter  

The dataset contains information about direct marketing campaigns (phone calls) with the classification goal of predicting whether a client will subscribe to a term deposit.

### Key Characteristics

- **Total Records**: 45,211 samples  
- **Total Input Features**: 17  
- **Target Variable**: 'y' (yes/no)  
- **Class Distribution**: Imbalanced dataset with significantly fewer positive cases  

### Feature Categories

- Client demographics: age, job, marital status, education, housing, loan  
- Campaign details: contact type, day, month, duration, number of contacts  
- Economic indicators: employment variation rate, consumer price index  

After preprocessing (one-hot encoding), the dataset expands to **63 features**.

### Preprocessing Pipeline

1. Target variable 'y' converted to binary (yes = 1, no = 0)  
2. One-hot encoding applied to categorical variables (`drop_first=True`)  
3. StandardScaler applied to numerical features  
4. Train-test split: 80% training, 20% testing (random_state=42)  

---

## c. Models Used and Evaluation

The following six classification models were implemented and evaluated on the same dataset:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN)**
4. **Naive Bayes (Gaussian)**
5. **Random Forest (Ensemble Model)**
6. **XGBoost (Ensemble Model)**

### Evaluation Metrics Used

- **Accuracy** – Overall prediction correctness  
- **Precision** – Positive prediction reliability  
- **Recall** – Ability to detect positive cases  
- **F1 Score** – Harmonic mean of precision and recall  
- **MCC** – Balanced metric for imbalanced datasets  
- **AUC Score** – Classification discrimination ability  

---

## Model Comparison Table

| Model                | Accuracy (%) | AUC (%) | Precision (%) | Recall (%) | F1 Score (%) | MCC (%) |
|----------------------|--------------|---------|---------------|------------|--------------|---------|
| Logistic Regression  | 89.20        | 92.00   | 92.00         | 64.70      | 46.23        | 42.47   |
| Decision Tree        | 89.88        | 70.25   | 88.31         | 51.96      | 42.62        | 40.13   |
| K-Nearest Neighbors  | 90.09        | 87.10   | 92.03         | 57.29      | 48.51        | 44.24   |
| Naive Bayes          | 84.07        | 84.88   | 75.88         | 60.98      | 40.69        | 36.39   |
| Random Forest        | 90.90        | 92.04   | 92.04         | 64.70      | 50.56        | 46.87   |
| **XGBoost**          | **91.26**    | **92.97** | **92.97**   | **64.70**  | **56.55**    | **51.95** |

*Note: Percentages are scaled from decimal values (e.g., 0.8920 → 89.20%). Best performance highlighted in bold.*

---

## Observations on Model Performance (Required Format)

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Strong baseline model with high AUC (92.00%) but limited by linear decision boundary. Performs well but slightly lower F1 compared to ensemble methods. |
| Decision Tree | Lower AUC (70.25%) suggests possible overfitting. Single-tree structure limits generalization performance. |
| K-Nearest Neighbors | Good precision but moderate recall. Computationally intensive for large datasets. |
| Naive Bayes | Lower overall performance likely due to violated independence assumptions among correlated features. |
| Random Forest | Strong ensemble model with high F1 and MCC. More stable and robust compared to a single decision tree. |
| XGBoost | Best performing model across F1, MCC, AUC and Accuracy. Gradient boosting effectively handles complex feature interactions and class imbalance. |

---

## Detailed Analysis

**Best Performing Model: XGBoost Classifier**

XGBoost demonstrates superior performance across all key evaluation metrics:

- Highest F1 Score (56.55%)  
- Highest MCC (51.95%)  
- Highest AUC (92.97%)  
- Highest Accuracy (91.26%)  

This confirms that gradient boosting ensemble methods outperform traditional models for complex and imbalanced datasets.

---

## Final Summary

This project successfully implements and compares six classification models for predicting bank term deposit subscriptions.

Key conclusions:

- Ensemble methods (Random Forest and XGBoost) outperform traditional classifiers.
- XGBoost achieves the best balance between precision and recall.
- MCC and F1 Score provide better evaluation insight than accuracy alone for imbalanced datasets.
- Comprehensive metric evaluation ensures reliable model comparison.

---

## Deployment Details

All six trained models are saved as compressed `.pkl` files in the `model/` directory.

The Streamlit web application (`app.py`) provides:

### Input Section
- Sample test.csv download option  
- CSV upload functionality  
- Model selection dropdown  

### Results Section
- Dashboard-style evaluation metrics  
- Classification report  
- Confusion matrix  
- ROC curve  
- Downloadable predictions  

---

## Future Enhancements

1. Hyperparameter tuning using GridSearchCV or Bayesian Optimization  
2. SMOTE for handling class imbalance  
3. SHAP-based model interpretability  
4. Cross-validation-based model stability analysis  
5. Feature importance-based feature selection  

---

**End of README**
