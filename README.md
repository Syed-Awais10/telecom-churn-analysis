# Telecom Customer Churn Analysis
 
Exploratory analysis and churn prediction on the public **IBM Telco Customer Churn** dataset, using Python (pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn).
 
**Dataset:** 7,043 customers and 21 columns ([Kaggle: blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)). 11 rows have a blank `TotalCharges` and were dropped, leaving **7,032 customers**. Overall churn rate: **26.6%**.
 
> All business figures that refer to "1 million subscribers" are an **illustrative scenario with stated assumptions**, not something this dataset measures.
 
## Key findings
 
| Finding | Evidence |
|---|---|
| Contract type shows the widest churn gap | Month-to-month 42.7% vs one-year 11.3% vs two-year 2.8% (about 15x between month-to-month and two-year) |
| Churn is concentrated early in the customer lifetime | Median churned customer has a tenure of 10 months (mean 18). 41.9% of churners leave within 6 months and 55.5% within 12. Churn rate by tenure: 53.3% (0-6 months), 35.9% (7-12), 28.7% (13-24), 20.4% (25-48), 9.5% (49-72) |
| Fiber optic customers churn the most | 41.9% vs 19.0% for DSL and 7.4% for no internet. Fiber customers also pay more on average ($91.50/month vs $58.09 for DSL) |
| Churners pay more per month | $74.44 vs $61.31 for retained customers (a difference of $13.13, driven partly by the fiber customers above) |
| Strongest correlations with churn | Tenure -0.35, fiber optic +0.31, electronic check payment +0.30, two-year contract -0.30 |
 
Correlations show association, not cause. Nothing here demonstrates that changing contract type, price or payment method would change churn.
 
## Model results
 
Stratified 80/20 split (5,625 train / 1,407 test). Scaling and cross-validation run inside a scikit-learn pipeline, so no information from the test set or from held-out folds leaks into training.
 
| Held-out test set, threshold 0.5 | Logistic Regression | Random Forest (default settings) |
|---|---|---|
| Accuracy | 80.4% | 79% |
| Churn precision | 64.8% | 62% |
| Churn recall | 57.5% | 50% |
| ROC-AUC | **0.836** | 0.815 |
 
- 5-fold cross-validated ROC-AUC for Logistic Regression: **0.846 ± 0.005**.
- Always predicting "no churn" would score 73.4% accuracy, so accuracy alone says little here. ROC-AUC and churn recall are the more informative numbers.
- Logistic Regression scored slightly higher than an untuned Random Forest. The gap is small and the Random Forest was not tuned.
### Decision threshold
 
| Threshold | Precision | Recall |
|---|---|---|
| 0.50 | 64.8% | 57.5% |
| 0.40 | 57.8% | 68.4% (+11 points) |
 
Lowering the threshold to 0.40 catches more churners at the cost of more false alarms. 0.40 was picked from a small grid (0.50, 0.45, 0.40, 0.35) and then checked on out-of-fold predictions from the training data. F1 is fairly flat across 0.30-0.45 (about 0.61-0.64), so 0.40 is a recall-focused operating point, not a sharp optimum. The right threshold in practice depends on the cost of missing a churner versus contacting someone who was not going to leave, which is not estimated here.
 
### SMOTE comparison
 
| Training approach | CV ROC-AUC | Test ROC-AUC | Churn precision / recall at 0.5 |
|---|---|---|---|
| No resampling (final model) | 0.846 ± 0.005 | 0.836 | 0.648 / 0.575 |
| SMOTE inside each CV fold | 0.846 ± 0.005 | 0.834 | 0.497 / 0.781 |
 
SMOTE did not improve ROC-AUC. It only moved the operating point (more recall, less precision, accuracy 80% to 73%), which adjusting the threshold does without creating synthetic customers. The simpler model without resampling was kept.
 
An earlier version of this notebook applied SMOTE before cross-validation, which let synthetic samples leak across folds and inflated the CV score to 0.863. Moving SMOTE inside the pipeline gave the honest figure above.
 
### Feature importance
 
Tenure and MonthlyCharges rank in the top three for both models. Tenure and TotalCharges are highly correlated (0.83), so credit is split unstably between them. Importance charts show magnitude, not direction or cause.
 
## Business impact scenario (illustrative)
 
Assumptions are chosen, not estimated from the data: 1,000,000 subscribers, 10% of the churners the model catches are retained, $70/month revenue per retained customer, retained for 12 months.
 
- Churners per year at 26.6%: about 265,800
- Caught by the model at the 0.5 threshold (recall 57.5%): about 152,800, which requires flagging about 235,900 customers (about 83,000 false alarms)
- Retained at 10%: about 15,300 customers, or about **$12.8M** of annual revenue
- Break-even: the scenario stops being positive if contacting or incentivising each flagged customer costs more than about $54 per year
- Not included: offer costs, margin (this is revenue), and the real retention rate, which is the largest driver. The result scales linearly with it (5% gives about $6.4M, 20% about $25.7M)
## Limitations
 
- One public dataset from a single operator. Results may not transfer to other markets or time periods.
- Tenure is a snapshot, so there is no time dimension and no survival analysis.
- All findings are associations. No causal claims are made about contract type, price or payment method.
- The test set was used to compare the two models and reported once for the chosen threshold, so the 0.836 AUC carries a small optimism. It is consistent with the 0.846 cross-validated figure.
- The Random Forest was not hyperparameter-tuned.
## How to run
 
1. Open `Churn_analysis.ipynb` in Google Colab (or Jupyter).
2. Install dependencies: `pip install kagglehub imbalanced-learn scikit-learn pandas matplotlib seaborn`.
3. Authenticate with Kaggle using `kagglehub.login()` or Colab Secrets. **Never paste an API token into a notebook cell.**
4. Use **Runtime, Restart session and run all** so every printed number comes from a single run.
## Repository contents
 
- `Churn_analysis.ipynb`: full analysis (cleaning, EDA, modelling, threshold analysis, SMOTE comparison, scenario)
- `README.md`: this file
 
