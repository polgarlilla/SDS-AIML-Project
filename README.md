# SDS-AIML-Project: Student Dropout Prediction

Corvinus University of Budapest Social Data Science, Applying and Interpreting Machine Learning Project

## 1. Project Question and Motivation

**Research Question:** Can we classify if a student in Portugal will be academically successful or unsuccessful based on certain socioeconomic and educational factors?

**Motivation:**  
Student dropout is a critical issue in higher education, impacting institutions' success rates and students' career trajectories. This project aims to:
- Identify key predictors of student dropout vs. graduation
- Build and compare multiple classification models (Logistic Regression, Lasso, KNN, Decision Trees)
- Provide actionable insights for early intervention strategies
- Evaluate model performance using rigorous cross-validation and holdout testing

The dataset contains 4,424 student records with 33 features including academic performance, parental education, financial status, and macroeconomic indicators (inflation, GDP).

---

## 2. Data Source & Access Instructions

**Data Source:** The dataset under analysis for our project will be the “Predict Students’ Dropout and Academic Success” dataset from UC Irvine, donated by creators Valentim Realinho, Mónica Vieira Martins, Jorge Machado and Luís Baptista and funded by program SATDAP - Capacitação da Administração Pública under grant POCI-05-5762-FSE-000191, Portugal. Link for the data: https://shorturl.at/hsEhU. 

**File:** `data/Dropout_Data.xlsx`

**Data Access:**
- The data file is included in this repository under the `data/` directory
- No external download required; the project reads directly from `data/Dropout_Data.xlsx`
- Sheet name: `"Original Data"`
- Target variable: `Target` (values: "Dropout" → 0, "Graduated" → 1, "Enrolled" → excluded)
- 4,424 student records with 33 features

**Data Features Include:**
- **Academic:** Previous qualification (grade), attendance performance
- **Demographics:** Gender, age, international status, scholarship holder status
- **Parental Education:** Mother's and father's qualification levels (recoded to 0-4 scale)
- **Financial:** Tuition fees up to date, debtor status
- **Macroeconomic:** Inflation rate, GDP at enrollment time

---

## 3. Environment Setup Steps

### Prerequisites
- Python 3.9 or higher
- `pip` (Python package manager)

### Setup Instructions

#### Windows (PowerShell)
```powershell
# 1. Clone the repository (if not already done)
git clone https://github.com/polgarlilla/SDS-AIML-Project.git
cd SDS-AIML-Project

# 2. (Optional) Create a virtual environment for isolation
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install required packages
pip install -r requirements.txt

# 4. Verify installation
python -c "import pandas, sklearn, matplotlib; print('All packages installed successfully!')"
```

#### macOS / Linux
```bash
# 1. Clone the repository (if not already done)
git clone https://github.com/polgarlilla/SDS-AIML-Project.git
cd SDS-AIML-Project

# 2. (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Verify installation
python -c "import pandas, sklearn, matplotlib; print('All packages installed successfully!')"
```

### Required Packages
- `pandas` – data manipulation and analysis
- `numpy` – numerical computing
- `scikit-learn` – machine learning models and metrics
- `matplotlib` – data visualization
- `seaborn` – statistical plotting
- `statsmodels` – statistical modeling and VIF calculations
- `openpyxl` – reading Excel files
- `scipy` – scientific computing
- `joblib` – parallel processing and model persistence

---

## 4. How to Run

### Quick Start
```powershell
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

### Command-Line Arguments (if applicable)

The main pipeline can be customized with arguments. Basic syntax:
```powershell
python main.py --data_path ./data --seed 101
```

**Supported Arguments:**
- `--data_path` (default: `./data`) – path to the data directory
- `--seed` (default: `3`) – random seed for reproducibility

**Example:**
```powershell
# Run with custom data path and seed
python main.py --data_path ./data --seed 101
```

### What the Pipeline Does

Running `python main.py` executes the following steps:

1. **Data Loading & Inspection** (`data.py`)
   - Loads `Dropout_Data.xlsx`
   - Displays first rows, data types, and missing values
   - Creates a 90/10 train/holdout split (stratified by target)

2. **Data Preprocessing** (`data.py`)
   - Selects 11 key features
   - Encodes target variable (Dropout=0, Graduated=1)
   - Recodes parental education (34 levels → 0-4 scale)
   - Converts categorical variables to proper types

3. **Descriptive Statistics** (`descriptive_stats.py`)
   - Generates summary statistics for features and target
   - Plots target distribution, numeric histograms, boxplots
   - Visualizes parental qualification distributions
   - Checks multicollinearity via correlation heatmap and VIF

4. **Feature Engineering** (`features.py`)
   - Creates preprocessing pipeline:
     - Scales numeric features (StandardScaler)
     - One-hot encodes categorical features (drop first category to avoid collinearity)

5. **Model Training & Evaluation** (`models.py` + `evaluate.py`)
   - Trains 4 classifier models with 5-fold cross-validation:
     - **Baseline Logistic Regression** – simple, interpretable baseline
     - **Lasso Logistic Regression** – L1 regularization for feature selection
     - **KNN Classifier** – distance-based (number of neighbors tuned via GridSearchCV)
     - **Decision Tree Classifier** – tree-based model (max_depth, min_samples_leaf tuned via GridSearchCV)
   - Generates cross-validation metrics (AUC, Accuracy, Precision, Recall)
   - Creates confusion matrices and ROC curves (80/20 train-test split visualizations)

6. **Final Holdout Evaluation**
   - Evaluates each model on the held-out 10% test set
   - Generates final confusion matrices and holdout metrics
   - **Best Performer:** Decision Tree (highest accuracy on holdout set)

### Output Files

All results are saved to the `outputs/` directory:

```
outputs/
├── summary_features.csv              # Descriptive stats for features
├── summary_target.csv                # Descriptive stats for target
├── target_distribution.png           # Class distribution bar chart
├── correlation_heatmap.png           # Numeric features correlation
│
├── baseline_logistic_regression_cv_metrics.json
├── baseline_logistic_regression_cv_metrics.csv
├── baseline_logistic_regression_cm_8020.png
├── baseline_logistic_regression_roc_8020.png
├── baseline_logistic_regression_holdout_metrics.json
├── baseline_logistic_regression_cm_holdout.png
│
├── lasso_logistic_regression_cv_metrics.json
├── lasso_logistic_regression_cv_metrics.csv
├── lasso_logistic_regression_cm_8020.png
├── lasso_logistic_regression_roc_8020.png
├── lasso_logistic_regression_holdout_metrics.json
├── lasso_logistic_regression_cm_holdout.png
│
├── knn_classifier_cv_metrics.json
├── knn_classifier_cv_metrics.csv
├── knn_classifier_cm_8020.png
├── knn_classifier_roc_8020.png
├── knn_classifier_holdout_metrics.json
├── knn_classifier_cm_holdout.png
│
└── decision_tree_classifier_cv_metrics.json
    decision_tree_classifier_cv_metrics.csv
    decision_tree_classifier_cm_8020.png
    decision_tree_classifier_roc_8020.png
    decision_tree_classifier_holdout_metrics.json
    decision_tree_classifier_cm_holdout.png
```

---

## 5. Expected Runtime

**Full Pipeline Execution Time:** ~3–5 minutes

**Breakdown by Component** (approximate):
- **Data Loading & Inspection:** 5–10 seconds
- **Descriptive Statistics & Visualizations:** 20–30 seconds
- **Model Training (5-fold CV on 4 models):** 1–2 minutes
- **Holdout Evaluation & Final Visualizations:** 30–60 seconds
- **Total:** 2–3 minutes (on standard hardware)

*Note: Runtime varies based on system specifications. GridSearchCV for KNN and Decision Tree may use parallel processing (`n_jobs=-1`), reducing time on multi-core systems.*

---

## 6. Project Structure

```
SDS-AIML-Project/
├── data.py                      # Data loading and preprocessing
├── descriptive_stats.py         # Exploratory data analysis & visualizations
├── evaluate.py                  # Model evaluation metrics and plotting
├── features.py                  # Feature scaling and encoding pipeline
├── main.py                      # Main workflow orchestrator
├── models.py                    # Model training with hyperparameter tuning
├── requirements.txt             # Python package dependencies
├── README.md                    # This file
├── data/
│   └── Dropout_Data.xlsx        # Student data (4,424 records, 33 features)
└── outputs/                     # Generated results (metrics, plots, confusion matrices)
```

---

## 7. Key Findings

- **Best Model:** Decision Tree Classifier (highest accuracy on holdout set)
- **Multicollinearity:** No high level multicollinearity detected (VIF analysis and correlation heatmap)
- **Model Performance:** 5-fold CV AUC ranges from 0.65–0.75 across models

---

## 8. Technologies & Libraries

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.9+ | Programming language |
| pandas | Latest | Data manipulation |
| scikit-learn | Latest | ML models & preprocessing |
| matplotlib | Latest | Data visualization |
| seaborn | Latest | Statistical plotting |
| statsmodels | Latest | Statistical analysis (VIF) |
| numpy | Latest | Numerical computing |
| openpyxl | Latest | Excel file I/O |

---

## 9. Author & Attribution

- **Project:** Corvinus University of Budapest, Social Data Science Program
- **Applying and Interpreting Machine Learning** coursework
- **Repository:** [polgarlilla/SDS-AIML-Project](https://github.com/polgarlilla/SDS-AIML-Project)

---

## 10. License

[Specify your license, e.g., MIT, Apache 2.0, etc.]

---

## 11. Contact & Support

For questions or issues, please open an issue on the GitHub repository or contact the project maintainers.
