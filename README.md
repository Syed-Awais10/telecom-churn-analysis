# Telecom Customer Churn Analysis

**Tools:** Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit-learn | imbalanced-learn  
**Dataset:** IBM Telco Customer Churn (7,032 customers, 20 features)

## Project Summary
End-to-end churn analysis and predictive modelling to identify at-risk telecom 
customers and recommend data-driven retention strategies.

## Key Findings
- Month-to-month customers churn at **42.7%** vs **2.8%** for two-year contracts — a 15x gap
- Median churner leaves by **month 10**, not month 18 as mean suggested — 
  reframing the critical intervention window to the first 6 months
- Fiber optic customers churn at **41.9%** despite paying premium prices — 
  indicating a value perception problem
- Churned customers pay **$13.13/month more** on average than retained customers

## Model Results
| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | 80% | 79% |
| Churn Recall | 68% (after threshold tuning) | 51% |
| ROC-AUC | 0.836 | 0.814 |
| CV ROC-AUC | 0.863 ± 0.012 | — |

## Techniques Used
- SMOTE to handle 73/27 class imbalance
- Decision threshold tuning (0.5 → 0.40) to improve churn recall
- 5-fold cross-validation to confirm model stability
- Correlation heatmap with multicollinearity analysis

## Business Impact
Scaled to a 1M-subscriber telco, the model flags ~152,000 at-risk customers 
annually. Retaining 10% of flagged customers at $70/month average = **$12.7M+ 
in saved annual revenue**.

## Files
| File | Description |
|------|-------------|
| `churn_analysis.ipynb` | Full analysis notebook |
