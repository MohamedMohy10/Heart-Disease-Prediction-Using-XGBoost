# Heart Disease Classifier (XGBoost)
---

## Project Description

A simple machine learning pipeline that predicts the presence of heart disease using clinical markers. 
This project leverages a large-scale dataset of 60,000 patient records to achieve high-precision classification.

---

## Dataset

- **Dataset:** Predicting Heart Disease 
- **Source:** [Kaggle](https://www.kaggle.com/competitions/playground-series-s6e2/data)  
- **Size:**  
  - ~ +60,000 rows (generated data, see dataset description above)
---

## 🛠️ Tech Stack
* **Language:** Python 3.12.10
* **Libraries:** Scikit-Learn, XGBoost, Pandas, Joblib
* **Visualization:** Matplotlib, Seaborn
---

## Results

**Final model performance on test set:**

| Metric         | Result |
|----------------|--------|
| Precision      | **89%** |
| Recall         | **89%** |
| F1-score       | **89%** |
| Accuracy       | **88.87%** |

Confusion Matrix:

|      | precision  | recall | f1-score | 
|-------|------------|--------|--------|  
| **0** |    0.89    |  0.91  |    0.90  |   
| **1** |    0.88    |  0.87  |    0.87  |
|       |            |        |          |
| macro avg   |    0.89  |    0.89  |    0.89  | 
| weighted avg    |   0.89   |   0.89  |    0.89 |


**Highlights:**

- At first, I compared different algorithms : LogisticRegression, SVM, and RandomForest which achieved the best results
- XGBoost got a slightly improved performance than RandomForests (Expected)

---
