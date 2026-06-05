# Star Properties Machine Learning Pipeline & Prediction App

An interactive, end-to-end Machine Learning pipeline implemented in Python using Scikit-Learn. This repository compares multiple classification and regression algorithms to categorize star types and predict their absolute magnitudes based on physical traits, serving as an operational command-line tool for live predictions.

---

## 📊 Project Overview & Core Findings
This project builds a comparative benchmark between **Tree-Based Models** (Decision Trees, Random Forests) and **Geometric/Linear Models** (Logistic Regression, SVM) utilizing a 5-Fold Cross-Validation (`cv=5`) evaluation framework to eliminate data split bias across a 240-star cosmic matrix.

Our terminal test executions yielded a clear distinction in how different mathematical algorithms interact with cosmic physics data:
* **The Tree Family Dominates:** Random Forests and Decision Trees effortlessly capture the complex, multi-scale step thresholds of stellar evolution, achieving a robust **~93-95% Mean CV Accuracy** in classification and maintaining **~0.97-0.98 Mean CV $R^2$ variance** in regression.
* **Linear/Geometric Limits:** Linear models hit a distinct performance wall, with standard Linear Regression bottoming out at a **0.3725 Mean CV $R^2$ score**. Because absolute magnitude relies on a logarithmic physics curve, forcing a flat, straight line through curved cosmic data causes severe underfitting.

---

## 🚀 Changelog & Updates

### 5th June 2026
* **[Optimized]** Integrated Scikit-Learn `FunctionTransformer` utilizing `np.log1p` nested inside the Linear Regression pipeline. This successfully isolated and linearized the exponential scale of the `Luminosity` feature, breaking through the previous baseline performance wall.
* **[Fixed]** Resolved a mathematical `ValueError: Input X contains NaN` crash caused by log-transforming zero-value boundaries by implementing a safe $+1$ logarithmic offset.

---

## 🧬 Algorithm Requirements & Preprocessing Footprint

To maximize efficiency and evaluate the models under real-world conditions, the codebase uses two distinct preprocessing footprints tailored to the absolute minimum requirements of each algorithm:

### 1. Tree-Based Models (`RandomForest`, `DecisionTree`)
* **Requirements:** Categorical encoding (strings to numbers) and full rows (no missing data).
* **Feature Scaling:** **Not Required.** Trees evaluate variables column-by-column using simple threshold cuts (e.g., `is Radius > 0.17?`). Mixed raw units like a 40,000K Temperature alongside a 0.15 Relative Radius do not disrupt their logic.

### 2. Distance/Linear Models (`LogisticRegression`, `LinearRegression`, `SVM/SVR`)
* **Requirements:** Categorical encoding, complete rows, and structural alignment.
* **Feature Scaling:** **ABSOLUTELY MANDATORY.** Geometric algorithms calculate spatial coordinate distances or weighted gradients. Because SVR treats data points as positions in space, a feature with massive raw numbers (like Temperature) would completely drown out a feature with small numbers (like Radius). `StandardScaler` is applied here to normalize all dimensions.

---

## 📂 Repository Architecture
* **`6 class csv.csv`**: The master dataset containing raw properties for 240 observed stars.
* **`train.py` (or `ai.py`)**: The data compilation script. It applies targeted preprocessing transformers, executes automated 5-Fold cross-validation evaluations across 8 separate models, and bundles the final trained network weights into a unified, local `star_models.pkl` bundle.
* **`app.py`**: A lightweight, live Command Line Interface (CLI). It loads the local pickle bundle, takes input from the user's keyboard, normalizes text variations, and generates instantaneous structural predictions.

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
