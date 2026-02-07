# Retail-Intelligence-ML-Segmentation
A dual-stage Machine Learning framework for customer income prediction and strategic segmentation using US Census data.
# Retail Intelligence: Customer Income Prediction & Strategic Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

## 📌 Project Overview
This project delivers a data-driven decision support system for a retail client to optimize marketing ROI. Using US Census data (approx. 200,000 records), the system addresses two core business objectives:
1. **High-Value Lead Identification**: Predicting whether a customer earns >$50,000 to target premium product campaigns.
2. **Customer Persona Profiling**: Segmenting the customer base into 5 actionable clusters for personalized marketing strategies.

## 🚀 Key Results
* **Classification Performance**: Achieved a **0.94 ROC-AUC** using ensemble tree-based models. 
* **Targeting Efficiency**: Improved high-income lead identification by **8-13x** compared to random targeting.
* **Strategic Impact**: Identified "White Collar Professionals" as the primary target segment, showing a high-income density of 18.2% vs the 6.2% baseline.
* **Feature Engineering**: Reduced 42 raw variables into 14 high-interpretability features (e.g., `is_fulltime_fullyear`, `has_capital_income`), improving PR-AUC by 12%.

## 🛠 Tech Stack
* **Languages**: Python
* **Libraries**: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Techniques**: SMOTE (for class imbalance), K-Means Clustering, Gradient Boosting, Logistic Regression.

## 📈 Pipeline Architecture
1. **EDA**: Handled a 15:1 class imbalance and performed correlation analysis on 42 demographic features.
2. **Modeling**: Implemented a two-stage approach:
   - *Stage 1*: Logistic Regression baseline for interpretability.
   - *Stage 2*: Tree-based models with optimized thresholds for business-specific Precision-Recall tradeoffs.
3. **Segmentation**: Used the Elbow Method and Silhouette Analysis to define 5 customer personas, mapping them to specific marketing channels (Digital vs. Direct Mail).

## 📂 Repository Structure
* `notebooks/`: Step-by-step experimentation from raw data to final models.
* `reports/`: Comprehensive business report summarizing technical findings and strategic recommendations.
* `requirements.txt`: Necessary libraries to reproduce the environment.

## 🔧 Installation & Usage
1. Clone the repo: `git clone https://github.com/YourUsername/Retail-Intelligence.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebooks in order: `01_EDA` -> `02_Baseline` -> `03_Advanced_Modeling` -> `04_Segmentation`.

## ✍️ Author
**Yangfan Cui** - PhD Candidate specializing in Survival Analysis & Machine Learning.
