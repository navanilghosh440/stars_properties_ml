# Star Properties Machine Learning Pipeline & Prediction App

An interactive, end-to-end Machine Learning pipeline implemented in Python using Scikit-Learn. This repository compares multiple classification and regression algorithms to categorize star types and predict their absolute magnitudes based on physical traits, serving as an operational command-line tool for live predictions.

---

## 📊 Project Overview & Core Findings
This project builds a comparative benchmark between **Tree-Based Models** (Decision Trees, Random Forests) and **Geometric/Linear Models** (Logistic Regression, SVM) using a minimalist preprocessing architecture.

Our terminal test executions yielded a clear distinction in how different mathematical algorithms interact with cosmic physics data:
* **The Tree Family Dominates:** Random Forests and Decision Trees effortlessly capture the complex, step-based structures of stellar evolution, achieving **~95% accuracy** in classification and explaining **~99% of the variance ($R^2$)** in absolute magnitude.
* **Linear/Geometric Limits:** Linear models hit a distinct performance wall (~77% classification accuracy / ~66% regression $R^2$). Because absolute magnitude relies on a logarithmic physics curve, forcing a flat, straight line through curved cosmic data causes severe underfitting.

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
* **`train.py` (or `ai.py`)**: The data compilation script. It splits data into un-leaked sets, applies targeted preprocessing transformers, benchmarks 8 separate models, and bundles the trained network weights into a unified, local `star_models.pkl` bundle.
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
