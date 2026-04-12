# Chemical Segregation – Binary Classification Project

A machine learning project that classifies chemical samples into two categories (0 or 1) using a dataset of 3,000 samples with 3 numerical features (`f1`, `f2`, `f3`). Four classifiers are implemented **from scratch** (without using scikit-learn's built-in estimators) and benchmarked against each other: Logistic Regression, SVM, KNN, and Decision Tree.

---

## Project Structure

```
.
├── main.ipynb               # Main Jupyter Notebook (all code and analysis)
├── Logistic_X_Train.csv     # Feature data (3000 × 3: f1, f2, f3)
├── Logistic_Y_Train.csv     # Target labels (3000 × 1: binary 0/1)
└── README.md                # This file
```

---

## Package Requirements

Python **3.8 or higher** is recommended (developed on Python 3.14.2).

| Package        | Purpose                                              | Recommended Version |
|----------------|------------------------------------------------------|---------------------|
| `pandas`       | Data loading, inspection, and manipulation           | >= 1.5.0            |
| `numpy`        | Numerical operations and array handling              | >= 1.23.0           |
| `matplotlib`   | Plotting (2D/3D visualizations, ROC curves)          | >= 3.6.0            |
| `seaborn`      | Statistical data visualization and styling           | >= 0.12.0           |
| `scikit-learn` | Preprocessing, metrics, PCA, LDA, train/test split   | >= 1.1.0            |
| `ipython`      | Notebook display utilities (`display`, `HTML`)       | >= 8.0.0            |
| `jupyter`      | Running the `.ipynb` notebook                        | >= 1.0.0            |

> **Note:** `scikit-learn` is used **only** for `StandardScaler`, `train_test_split`, evaluation metrics (`accuracy_score`, `precision_score`, etc.), `PCA`, `LDA`, and `roc_curve`. All four classifiers (Logistic Regression, SVM, KNN, Decision Tree) are custom-built from scratch.

---

## Setup & Run Instructions

### Step 1 – Clone or Download the Project

Place all three files in the same directory:
- `main.ipynb`
- `Logistic_X_Train.csv`
- `Logistic_Y_Train.csv`

### Step 2 – Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### Step 3 – Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ipython jupyter
```

Or install from a requirements file (if provided):

```bash
pip install -r requirements.txt
```

<details>
<summary>requirements.txt (copy and save if needed)</summary>

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
ipython>=8.0.0
jupyter>=1.0.0
```

</details>

### Step 4 – Launch the Notebook

```bash
jupyter notebook main.ipynb
```

This will open the notebook in your browser.

### Step 5 – Run the Notebook

Inside Jupyter, select:
```
Kernel → Restart & Run All
```

This executes all cells from top to bottom, producing the full analysis, visualizations, and model evaluation results.

---

## What the Notebook Does

1. **Data Loading & Inspection** – Loads `Logistic_X_Train.csv` and `Logistic_Y_Train.csv`, checks shapes, data types, and missing values.
2. **Exploratory Data Analysis (EDA)** – Visualizes feature distributions, class balance, pairplots, and correlation heatmaps.
3. **Preprocessing** – Applies `StandardScaler` and performs an 80/20 train-test split.
4. **Dimensionality Analysis** – Runs PCA and LDA to assess linear separability and feature variance.
5. **Model Training** – Trains four custom-built classifiers:
   - `LogisticRegressionCustom` (gradient descent)
   - `SVMCustom`
   - `KNNCustom`
   - `DecisionTreeCustom`
6. **Evaluation** – Compares models using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC/AUC curves.
7. **Conclusions** – Identifies the best model (Logistic Regression, F1 = 0.9950) and assesses linear separability.

---

## Results Summary

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.9950   | 0.9933    | 0.9966 | 0.9950   |
| SVM                 | ~0.99    | ~0.99     | ~0.99  | ~0.99    |
| KNN                 | ~0.98    | ~0.98     | ~0.98  | ~0.98    |
| Decision Tree       | ~0.98    | ~0.98     | ~0.98  | ~0.98    |

**Best Model: Logistic Regression** — The dataset is linearly separable, making linear models the optimal choice.

---

## Notes

- All CSV files must reside in the **same directory** as `main.ipynb` so the relative paths (`'Logistic_X_Train.csv'`, `'Logistic_Y_Train.csv'`) resolve correctly.
- If running on **Google Colab**, upload both CSV files to the Colab session storage before executing the notebook.
- 3D plots require `mpl_toolkits.mplot3d`, which is bundled with `matplotlib` and needs no separate installation.
