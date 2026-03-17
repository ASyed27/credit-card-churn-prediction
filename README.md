# Credit Card Customer Churn Prediction
### DSCI 521 — Data Analysis and Interpretation
**Drexel University | Winter 2026**

**Team Members:**
- Ahmed Syed (as6387@drexel.edu) — Data preparation, business problem framing, project scoping
- Sohan Miryalkar (sm5243@drexel.edu) — Exploratory data analysis, statistical analysis, data visualization
- Aadesh Kalbhor (ak4567@drexel.edu) — Predictive modeling, model evaluation, results interpretation

---

## Project Overview

Credit card companies lose significant revenue when customers close their accounts. Unlike sudden cancellations, churn is a gradual process — declining transactions, falling balances, and reduced engagement signal attrition weeks before it happens. This project uses exploratory data analysis, statistical testing, and supervised machine learning to identify behavioral warning signals and predict customer churn before it occurs.

**Key findings:**
- Behavioral features (transaction count, amount, revolving balance) are the strongest churn predictors
- Demographic features (age, tenure, dependents) show no statistically significant difference between churned and existing customers
- A bimodal transaction distribution revealed two distinct behavioral segments within existing customers — Low-Volume customers exhibit pre-churn behavioral patterns
- Random Forest achieves 95% accuracy and 83% F1-score on the attrited class, outperforming Logistic Regression on all key metrics
- 5-fold cross-validation confirms model stability (RF: 95.5% accuracy ±0.6%, 84.8% F1 ±2.1%)

---

## Repository Contents
```
project/
├── DSCI521_CreditCard_Churn_Final.ipynb
├── BankChurners.csv
├── DSCI521_Churn_Presentation.pptx
└── README.md
```
---

## Dataset

**Source:** [Kaggle — Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)

**Origin:** analyttica.com

**License:** CC0 (Public Domain)

**To acquire the dataset:**
1. Go to https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers
2. Click the Download button (requires a free Kaggle account)
3. Unzip the downloaded file
4. Rename or locate the file `BankChurners.csv`
5. Place it in the same directory as the notebook, or upload it to your Google Colab session

**Dataset summary:**
- 10,127 customer records
- 23 original columns (21 after removing 2 leakage columns)
- 20 features used for modeling (Available Credit dropped due to r=0.99 with Credit Limit)
- Target variable: `Attrition_Flag` (Existing Customer / Attrited Customer)
- Class distribution: 83.9% Existing / 16.1% Attrited

---

## How to Run the Notebook

### Option 1 — Google Colab (Recommended)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook** and upload `DSCI521_CreditCard_Churn_Final.ipynb`
3. Upload `BankChurners.csv` to the Colab session:
   - Click the **folder icon** in the left sidebar
   - Click the **upload icon** and select `BankChurners.csv`
4. Run all cells top to bottom using **Runtime → Run all**

### Option 2 — Local Jupyter Notebook

1. Install required libraries (see Dependencies below)
2. Place `BankChurners.csv` in the same directory as the notebook
3. Launch Jupyter: `jupyter notebook`
4. Open `DSCI521_CreditCard_Churn_Final.ipynb`
5. Run all cells top to bottom using **Cell → Run All**

---

## Dependencies

All required libraries are available in the standard Python data science stack. Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

If using a fresh environment:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

**Library versions used:**
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

No additional downloads or API keys are required beyond the dataset.

---

## Notebook Structure

The notebook is organized into 8 sections that should be run in order:

| Section | Description |
|---------|-------------|
| **1. Imports & Setup** | Load all libraries and configure plot settings |
| **2. Data Loading & Cleaning** | Load BankChurners.csv, drop leakage columns, create binary churn label, define feature groups |
| **3. Target Distribution** | Visualize and quantify the 83.9/16.1% class imbalance |
| **4. All Numeric Features** | Histogram grid comparing all 14 numeric features across churn groups |
| **5. Bimodal Analysis & Segment Profiling** | Identify bimodal transaction distribution, split existing customers into High-Volume (60+) and Low-Volume (<60) segments, compare 11 features across segments |
| **6. Categorical & Statistical Analysis** | Churn rates by category, correlation heatmap, t-tests on all numeric features, boxplots, mutual information scoring |
| **7. Predictive Modeling** | Logistic Regression and Random Forest training, evaluation, confusion matrices, feature importance, model comparison, 5-fold cross-validation |
| **8. Summary** | Key findings table |

**Important:** Run cells in order from top to bottom. Later sections depend on variables defined in earlier sections. Do not skip cells.

---

## Key Results

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | 85% | 95% |
| Attrited Precision | 52% | 93% |
| Attrited Recall | 81% | 77% |
| Attrited F1-Score | 63% | 83% |

**5-Fold Cross-Validation (Random Forest):**
- Accuracy: 95.5% (±0.6%)
- F1-Score: 84.8% (±2.1%)

**Recommended model:** Random Forest — superior precision (93% vs 52%) makes it preferable for targeted retention campaigns where minimizing false alarms is critical.

---

## Challenges Encountered

**1. Class Imbalance**
The raw dataset is 83.9% existing customers and 16.1% attrited. A naive model could achieve 84% accuracy by predicting "existing" for every customer. We addressed this using `class_weight='balanced'` in both models, which automatically adjusts the learning process to treat both classes equally regardless of their frequency.

**2. Feature Redundancy**
`Credit_Limit` and `Avg_Open_To_Buy` (Available Credit) were found to be nearly perfectly correlated (r=0.99). Keeping both features would inflate feature importance scores and introduce redundancy. We dropped `Avg_Open_To_Buy` before modeling, reducing the feature set from 21 to 20.

**3. Bimodal Distribution**
The transaction count distribution among existing customers showed an unexpected bimodal shape — two distinct peaks rather than a single normal distribution. This required additional investigation to understand and ultimately led to the segment profiling analysis, which became one of the most actionable findings of the project.

**4. No Temporal Data**
The dataset is a static snapshot of customer behavior at a single point in time. Churn is inherently a temporal process — a customer declining from 80 to 30 transactions over 6 months is a very different risk profile than one who has consistently made 30 transactions. Without time-series data, we cannot capture behavioral trajectories, only current state.

---

## Known Limitations

1. **No Temporal Dimension** — Static snapshot only. Time-series transaction history would improve predictive power significantly.

2. **Dataset Size** — 10,127 rows is sufficient for this scope but smaller than real banking deployments. Model generalizability across different institutions or card products is unknown.

3. **No Hyperparameter Tuning** — Both models use default parameters. GridSearchCV tuning could improve Random Forest recall on the attrited class.

4. **No Feature Engineering** — All features are raw point-in-time values. Derived features such as contact-to-transaction ratios or rolling spending trend metrics could capture behavioral dynamics more precisely.

5. **Black Box Model** — Random Forest does not explain individual predictions. Adding SHAP (SHapley Additive exPlanations) values would allow retention teams to understand why a specific customer was flagged, which is critical for real-world deployment and stakeholder trust.

---

## How to Continue This Project

This project is structured for continued development. To extend the analysis:

- **Hyperparameter tuning:** Add a GridSearchCV cell after the Random Forest training cell using `n_estimators`, `max_depth`, `min_samples_split`, and `class_weight` as search parameters
- **Additional models:** XGBoost and Gradient Boosting are natural next steps — import from `xgboost` and `sklearn.ensemble`
- **SHAP explanations:** Install with `pip install shap` and use `shap.TreeExplainer(rf)` for Random Forest
- **Feature engineering:** Create derived columns in the data preparation section before the train/test split
- **Segment-specific models:** Filter `df_model` by `Transactor_Segment` and train separate models for High-Volume and Low-Volume customers

---

## Contact

For questions about this project or the code:
- Ahmed Syed — as6387@drexel.edu
- Sohan Miryalkar — sm5243@drexel.edu
- Aadesh Kalbhor — ak4567@drexel.edu
