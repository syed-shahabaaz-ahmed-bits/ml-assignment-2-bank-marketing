# Bank Marketing Term Deposit Classification

## a. Problem statement
Predict whether a bank client will subscribe to a term deposit (`y`: yes/no) based on
telemarketing campaign attributes from the UCI Bank Marketing dataset. This is a binary
classification problem with class imbalance. The `duration` feature is excluded because it
leaks post-call information.

## b. Dataset description
- **Source:** UCI Bank Marketing (`bank-full.csv`)
- **Instances:** ~45,211
- **Features used:** all original features except `duration` (≥12)
- **Target:** `y` (yes/no)
- **Preprocessing:** StandardScaler for numeric columns; OneHotEncoder for categoricals;
  stratified 80/20 split (`random_state=42`)
- **Imbalance handling:** `class_weight='balanced'` for Logistic Regression, Decision Tree,
  and Random Forest

## c. Github Repository Link
https://github.com/syed-shahabaaz-ahmed-bits/ml-assignment-2-bank-marketing

## d. Models used
Comparison on the held-out test set (same split as `data/test_data.csv`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7548 | 0.7722 | 0.2662 | 0.6238 | 0.3732 | 0.2853 |
| Decision Tree | 0.8429 | 0.6081 | 0.3187 | 0.3015 | 0.3099 | 0.2214 |
| K-Nearest Neighbors | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Random Forest | 0.8789 | 0.7903 | 0.4790 | 0.3998 | 0.4359 | 0.3705 |

### Observations

With only ~11% of clients subscribing (`yes`), accuracy alone is misleading: a model that
always predicts `no` would score ~89% accuracy while finding zero subscribers. Under this
imbalance, recall (how many actual subscribers are caught), AUC (ranking quality), F1
(harmonic mean of precision and recall), and MCC (correlation between predictions and truth)
are more informative than accuracy.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieves the highest recall (62.4%), so it flags the largest share of true subscribers, helped by `class_weight='balanced'`. Its AUC of 0.77 is second-best, meaning probability scores rank likely subscribers reasonably well. Precision is low (26.6%), so many flagged clients will not subscribe - acceptable if the campaign cost of outreach is low. F1 (0.37) and MCC (0.29) sit mid-pack: strong at finding positives but noisy on the minority class. |
| Decision Tree | Reports 84.3% accuracy but the weakest AUC (0.61) and lowest recall (30.2%), indicating it misses most subscribers despite looking good on accuracy. F1 (0.31) and MCC (0.22) are the worst in the comparison, suggesting overfitting to majority-class patterns in training. Single trees are a poor fit for this high-dimensional, imbalanced problem without stronger regularization or ensembling. |
| K-Nearest Neighbors | Shows the highest accuracy (88.7%) and precision (55.3%) but the lowest recall (19.8%) - it rarely predicts `yes`, so it misses ~80% of subscribers. AUC (0.70) is moderate, yet F1 (0.29) and MCC (0.28) confirm the precision-recall trade-off hurts overall usefulness. Distance metrics in sparse one-hot space and lack of explicit class-weighting make KNN overly conservative on the minority class. |
| Naive Bayes | Delivers a balanced middle ground: recall 46.4%, precision 37.1%, F1 0.41, and MCC 0.33 - second-best MCC after Random Forest. AUC of 0.75 supports decent ranking of subscription probability despite Gaussian NB assumptions on mixed feature types. A solid baseline when interpretability and training speed matter, though it trails the ensemble on the strongest composite metrics. |
| Random Forest | Leads on AUC (0.79), F1 (0.44), and MCC (0.37) with recall ~40% and precision ~48% - the best overall balance for this imbalanced task. Ensemble averaging reduces single-tree overfitting seen in the Decision Tree while `class_weight='balanced'` improves minority-class detection. Recommended when the bank wants both reliable probability ranking and a workable precision-recall trade-off before contact. |
| Overall Winner for your dataset? | **Random Forest** - highest AUC, F1, and MCC on the held-out test set. Logistic Regression is preferable if maximizing recall (catching subscribers) matters most and false positives are cheap; Naive Bayes is a strong lightweight alternative. Accuracy-led rankings (KNN, Decision Tree) are misleading here because they largely reflect the ~89% majority `no` class. |

## How to run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python model/train.py
PYTHONPATH=. streamlit run app.py
```

## Streamlit app
The Streamlit UI computes **Accuracy, AUC, Precision, Recall, F1, and MCC live** on the
uploaded CSV (predictions via saved `.joblib` pipelines). It does **not** read
`model/last_test_metrics.csv`. Upload `data/test_data.csv` in the app to evaluate all five
models; the displayed metrics should match the table in section **d. Models used** above
(same held-out test split).

## Streamlit Cloud
Deploy with entrypoint `app.py`. After deployment, upload `data/test_data.csv` to verify
live metrics match the README table.
