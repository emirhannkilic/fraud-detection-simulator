# Fraud Detection + Business Impact Simulator

End-to-end machine learning pipeline on the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset.

## Dataset
- 284,807 transactions, 492 fraud (0.17%)
- 28 PCA-transformed features (V1-V28), Amount, Time, Class

## Project Structure
```
fraud-detection-simulator/
├── notebooks/
├── src/
├── models/
├── data/
├── outputs/
├── assets/
├── app.py
└── requirements.txt
```

## Sprint Progress
- [x] Sprint 1 - EDA
- [ ] Sprint 2 - Modeling
- [ ] Sprint 3 - Explainability
- [ ] Sprint 4 - Business Impact
- [ ] Sprint 5 - Streamlit Dashboard

## Key Findings - EDA
- Severe class imbalance: 99.83% normal, 0.17% fraud
- Fraud transactions have higher mean amount (122 vs 88) but lower max (2125 vs 25691)
- V features are uncorrelated with each other (PCA design)
- Two daily peaks visible in Time distribution