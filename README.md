## 📊 Project Overview & Core Findings
This project builds a comparative benchmark between **Tree-Based Models** (Decision Trees, Random Forests) and **Geometric/Linear Models** (Logistic Regression, SVM) utilizing a 5-Fold Cross-Validation (`cv=5`) evaluation framework to eliminate data split bias across a 240-star cosmic matrix.

Our terminal test executions yielded a clear distinction in how different mathematical algorithms interact with cosmic physics data:
* **The Tree Family Dominates:** Random Forests and Decision Trees effortlessly capture the complex, multi-scale step thresholds of stellar evolution, achieving an impressive **~92-95% Mean CV Accuracy** in classification and maintaining an ultra-precise **~0.97 Mean CV $R^2$ variance** in regression.
* **Overcoming Linear Limits via Power Transforming:** Standard linear and distance models initially hit a severe performance wall due to the extreme right-skewed, exponential nature of astronomical data metrics (like Luminosity). By integrating an automated `PowerTransformer(method='yeo-johnson')` directly into the geometric preprocessing pipeline, we successfully mapped skewed feature spreads into symmetrical, Gaussian bell curves—instantly optimizing mapping efficiency.

---

## 🧬 Algorithm Requirements & Preprocessing Footprint

To satisfy the mathematical assumptions of completely different algorithm families, the input data branches into two parallel preprocessor layouts (`t1` and `t2`) before training.

### 1. Tree-Based Sub-Pipeline (`t1`)
* **Target Models:** `RandomForestClassifier`, `DecisionTreeClassifier`, `RandomForestRegressor`, `DecisionTreeRegressor`
* **Mathematical Requirement:** These models operate using column-by-column binary threshold splits (e.g., `Luminosity > 500`). They are completely scale-invariant and indifferent to feature distribution shapes or skewness.
* **Preprocessing Footprint:** * Categorical columns (`Color`) are passed through `OneHotEncoder` to generate discrete binary flags.
  * Numerical columns are passed straight through via `remainder='passthrough'`. No feature scaling or variance stabilization is applied, keeping execution lightning-fast and preventing unnecessary mathematical overhead.

### 2. Geometric & Parametric Sub-Pipeline (`t2`)
* **Target Models:** `LogisticRegression`, `SVC`, `LinearRegression`, `SVR`
* **Mathematical Requirement:** These models rely heavily on the statistical assumption of perfectly normal (Gaussian) distributions and continuous linear boundaries. Wildly skewed, exponential scales (spanning multiple orders of magnitude) cause gradient instability and heavily warp geometric distance calculations.
* **Preprocessing Footprint:**
  * Categorical columns (`Color`) are handled via `OneHotEncoder(sparse_output=False)` to generate clean numerical arrays compatible with matrix operations.
  * Numerical columns (`Temperature`, `Luminosity`, `Radius`) are processed using `PowerTransformer(method='yeo-johnson')`. This looks at the data's skewness, calculates the optimal power exponent, shifts the data into a symmetrical bell curve, and automatically handles standard feature scaling to prevent larger features from dominating.
### 🧠 Architectural Choice: Why PowerTransformer instead of StandardScaler?
While `StandardScaler` is the standard default for feature scaling, it was intentionally rejected for the geometric pipeline in favor of `PowerTransformer` due to the physics of astronomical data:

* **StandardScaler Preserves the Flaws:** Standard scaling only alters the *unit scale* (forcing a mean of 0 and standard deviation of 1), but it leaves the underlying distribution completely unchanged. Because cosmic features like `Luminosity` span 9 orders of magnitude ($0.0001$ to $500,000$), the data is heavily right-skewed. Under standard scaling, 90% of the stars remain tightly crushed near zero while a few Hypergiants remain massive geometric outliers, warping the model's spatial grid.
* **PowerTransformer Rewrites the Geometry:** The Yeo-Johnson power transform calculates a custom mathematical exponent for each feature to stabilize variance and physically reshape the distribution. It pulls the tightly packed dwarf stars apart and pulls the extreme hypergiant outliers closer to the center—actively bending the skewed data into a symmetrical, Gaussian bell curve. 
* **The Result:** The `PowerTransformer` automatically handles feature scaling while simultaneously satisfying the strict normality assumptions of linear and distance-based models. This structural data correction un-bent the mathematical relationships, instantly boosting Linear Regression variance capture ($R^2$) from `0.6858` to `0.9507`.
---

## 📈 Preprocessing Performance Metrics

By tailoring the input data shape to match what each algorithm family mathematically expects, we unlocked an immediate performance surge across all linear and distance-based architectures:

| Model Type | Algorithm | Preprocessing Metric (Standard Scaling) | Optimized Metric (Power Transformer) | Performance Shift |
| :--- | :--- | :--- | :--- | :--- |
| **Classification** | Logistic Regression | 74.58% CV Accuracy | **86.25% CV Accuracy** | **+11.67% Accuracy Boost** |
| **Classification** | Support Vector Machine (SVC) | 75.00% CV Accuracy | **83.75% CV Accuracy** | **+8.75% Accuracy Boost** |
| **Regression** | Linear Regression | 0.6858 CV $R^2$ | **0.9507 CV $R^2$** | **+26.49% Variance Captured** |
| **Regression** | Support Vector Regressor (SVR) | 0.6983 CV $R^2$ | **0.9389 CV $R^2$** | **+24.06% Variance Captured** |

---

## 📂 Repository Architecture
* **`6 class csv.csv`**: The master dataset containing physical traits for 240 observed stars.
* **`train.py`**: The central engineering pipeline. Cleans text variations, routes features through parallel `ColumnTransformer` layouts, trains all 8 models simultaneously, evaluates stability via cross-validation, and serializes optimized model pipelines into a local binary file.
* **`app.py`**: A lightweight, interactive live Command Line Interface (CLI). Loads the serialized model cluster, captures real-time user inputs, validates format inputs, and maps immediate star classifications and brightness predictions.

---
## 📜 Dataset Credit & Attribution
The foundational data utilized to train these models is sourced from the **6 Class Star Dataset** hosted on Kaggle. It tracks a well-distributed matrix of physical star variables matching core astronomical metrics:

* **Temperature:** Measured in Absolute Kelvin ($K$).
* **Luminosity:** Measured relative to the Sun ($L/L_\odot$).
* **Radius:** Measured relative to the Sun ($R/R_\odot$).
* **Absolute Magnitude ($M_v$):** The intrinsic brightness of a star if viewed from a universal baseline distance of 32.6 light-years (10 parsecs).
* **Star Type:** Target classification index ranging from 0 (Brown Dwarf) to 5 (Hypergiant).

---
## 🛠️ How to Setup and Run Local Predictions

1. Ensure you have the required libraries installed
2. Run the train.py script to generate the model bundle
3. Launch the live command-line prediction interface(app.py)
