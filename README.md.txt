# 🏦 Loan Approval Prediction — ML Classification Study

## Overview
An end-to-end machine learning project that predicts whether a loan application will be approved or rejected based on applicant demographics, financial history, and loan details.

## Tech Stack
Python | scikit-learn | XGBoost | pandas | SMOTE | matplotlib | seaborn

## Models Compared
| Model | Type |
|---|---|
| Logistic Regression | Linear |
| Naive Bayes | Probabilistic |
| KNN | Instance-based (GridSearchCV tuned) |
| Decision Tree | Pre + Post Pruning (ccp_alpha) |
| SVC | RBF Kernel |
| Random Forest | Ensemble |
| Gradient Boosting | Ensemble |
| AdaBoost | Ensemble |
| XGBoost | Ensemble (Best Performer) |

## Key Highlights
- Leak-free sklearn Pipeline: imputation + encoding + scaling in one step
- SMOTE applied on training data only to handle class imbalance
- 5-Fold Stratified Cross-Validation for robust evaluation
- Evaluated on Accuracy, F1 Score, and ROC-AUC
- Feature importance averaged across Random Forest and XGBoost
- Decision Tree pruning: pre-pruning (max_depth) and post-pruning (ccp_alpha)
- User-friendly prediction function with human-readable inputs

## How to Run
1. Clone this repository
   git clone https://github.com/prasanth1816/LoanApprovalPrediction-ML
2. Install dependencies
   pip install -r requirements.txt
3. Open the notebook
   jupyter notebook Loan_Approval_Prediction.ipynb
4. Run all cells top to bottom