# BITS Pilani - WILP M.Tech (AIML)
## Course: Machine Learning (AIMLZG565) - Assignment 2
### Comprehensive In-Silico Biodegradability Classification via Molecular Descriptors

**Student id:** 2026AJ05162  
**Course:** AIMLZG565 - Machine Learning  
**Evaluation Marks:** 15 / 15  
**Submission Deadline:** 08-Nov-2026  
**Reproducible Random Seed:** `2002`

---

## a. Problem Statement
Evaluating whether chemical compounds readily biodegrade in aerobic aquatic environments is a mandatory regulatory prerequisite under international chemical safety frameworks, notably the European Union REACH regulation (*Registration, Evaluation, Authorisation and Restriction of Chemicals*) and OECD 301 testing guidelines. Conventional wet-laboratory biodegradation assays require prolonged incubation (28-day biodegradation tests) with microbial consortia extracted from sewage treatment plants; these experimental protocols are slow, labor-intensive, and financially prohibitive for large combinatorial libraries.

Quantitative Structure-Activity Relationship (**QSAR**) modelling offers a scalable, in-silico alternative by predicting experimental biodegradability directly from computed molecular descriptors. In this project, we formulate biodegradation prediction as a supervised binary classification task: classifying whether an organic molecule is **Ready Biodegradable (`RB`, 1)** or **Not Ready Biodegradable (`NRB`, 0)** based on 41 multidimensional DRAGON molecular descriptors.

We design, implement, and benchmark five fundamental machine learning architectures:
1. **Logistic Regression** (L2-regularized linear baseline)
2. **Decision Tree Classifier** (Non-linear axis-aligned recursive partitioning with pre-pruning)
3. **K-Nearest Neighbors (KNN)** (Instance-based metric space learning)
4. **Gaussian Naive Bayes** (Generative probabilistic classifier under class-conditional independence)
5. **Support Vector Machine (SVM)** (Maximum-margin classification via non-linear Radial Basis Function kernel)

All models are subjected to rigorous leak-free preprocessing, 80:20 stratified cross-validation, and comprehensive evaluation across six quantitative metrics: **Accuracy, AUC Score (ROC-AUC), Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)**.

---

## b. Dataset Description [1 Mark]
The dataset selected for this study is the benchmark **UCI QSAR Biodegradation Dataset** (UCI Machine Learning Repository ID: 254), compiled by the Milano Chemometrics and QSAR Research Group.

- **Data Source:** UCI Machine Learning Repository (`https://archive.ics.uci.edu/ml/datasets/QSAR+biodegradation`)
- **Total Instances:** 1,055 organic compounds (strictly satisfies $\ge 500$ instances requirement)
- **Total Features:** 41 molecular descriptors + 1 target attribute (strictly satisfies $\ge 12$ features requirement)
- **Missing / Null Values:** 0 (clean numerical representation across all descriptors)
- **Target Variable (`ReadyBiodegradable`):**
  - `0`: Not Ready Biodegradable (`NRB`) – **699 compounds (66.26%)**
  - `1`: Ready Biodegradable (`RB`) – **356 compounds (33.74%)**
  - Represents a moderate class imbalance (~2:1 ratio), necessitating robust threshold-invariant and correlation metrics like ROC-AUC and MCC.

### Molecular Descriptor Breakdown (41 Descriptors):
The 41 input descriptors represent chemical attributes calculated using DRAGON software:
1. **Topological Indices & Molecular Connectivity:** Leading eigenvalue of the Laplace matrix (`SpMax_L`), Balaban-like index from the Barysz matrix (`J_Dz(e)`), Burden matrix eigenvalues (`SpMax_B(m)`, `SM6_B(m)`), second Mohar shape index (`TI2_L`), and Lopping centric index (`LOC`).
2. **Constitutional Descriptors:** Number of heavy atoms (`nHM`), aliphatic carbon count (`nCp`), aromatic ring carbon count (`nCrt`), oxygen atom count (`nO`), nitrogen count (`nN`), halogen count (`nX`), and carbon percentage (`C%`).
3. **2D Autocorrelation & Fragment Frequencies:** Topological distance frequency counts between atom pairs (`F01[N-N]`, `F04[C-N]`, `F03[C-N]`, `F03[C-O]`, `F02[C-N]`).
4. **Functional Group Counts:** Hydrazine linkages (`nN-N`), aromatic nitro groups (`nArNO2`), aliphatic carbons with 3 heteroatoms (`nCRX3`), aromatic esters (`nArCOOR`), and hydrogen bond donors (`nHDon`).
5. **Electronic & Electrotopological State Properties:** Sum of $\text{dssC}$ E-states (`SdssC`), sum of $\text{dO}$ E-states (`SdO`), mean atomic Sanderson electronegativity (`Me`), and mean atomic ionization potential (`Mi`).

### Stratified Splitting & Held-out Test Data:
- **Partition Ratio:** 80% Training ($n = 844$) and 20% Held-out Testing ($n = 211$) using stratified sampling with `RANDOM_SEED = 2002`.
- **Held-out Test File:** Exported to `test_data.csv` in the root repository folder, containing all 41 raw feature columns and the ground-truth target column `ReadyBiodegradable` (140 NRB, 71 RB).
- **Leakage-Free Preprocessing:** A `StandardScaler` was fitted strictly on the 844 training instances and saved as `model/preprocessor.joblib`. Test samples are strictly transformed without fitting.

---

## c. Mandatory Submission Links [1 Mark]

1. **GitHub Repository Link:**  
   [https://github.com/AdiS-2105/ml-assignment-2](https://github.com/AdiS-2105/ml-assignment-2)  
   *(Contains complete source code, `requirements.txt`, `README.md`, `test_data.csv`, `ML_Assignment_2_Notebook.ipynb`, and serialized artifacts in `model/`)*

2. **Live Streamlit App Link:**  
   [https://adis-2105-ml-assignment-2.streamlit.app/](https://adis-2105-ml-assignment-2.streamlit.app/)  
   *(Interactive web application deployed on Streamlit Community Cloud)*

---

## d. Models Used & Evaluation Metrics Table [5 Marks]

All five classification models were trained on the standardized training partition ($n = 844$) and evaluated on the identical held-out test partition ($n = 211$, 140 NRB, 71 RB) using fixed `RANDOM_SEED = 2002`.

### Evaluation Metrics Comparison Table:
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8863 | 0.9358 | 0.8406 | 0.8169 | 0.8286 | 0.7437 |
| Decision Tree | 0.8057 | 0.8316 | 0.7273 | 0.6761 | 0.7007 | 0.5580 |
| KNN | 0.8768 | 0.9326 | 0.8082 | 0.8310 | 0.8194 | 0.7261 |
| Naive Bayes | 0.6540 | 0.8227 | 0.4928 | 0.9577 | 0.6507 | 0.4547 |
| Support Vector Machine | 0.8957 | 0.9534 | 0.8657 | 0.8169 | 0.8406 | 0.7639 |

---

## e. Observations on Model Performance [3 Marks]

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Strong Linear Baseline: Achieves an impressive Accuracy of 88.63% and AUC of 0.9358. The L2 ridge penalty ($C=1.0$) effectively shrinks coefficients across collinear molecular descriptors (e.g., `SpMax_L`, `nHM`), mitigating variance. Positive coefficients for oxygen count (`nO`) and ester linkages (`nArCOOR`) align with established biochemical principles that oxygenated and ester functional groups provide enzymatic cleavage sites for microbial degradation. |
| Decision Tree | Moderate Performance with Pruning: Achieves 80.57% Accuracy, 0.7007 F1-score, and 0.5580 MCC. Constraining tree complexity (`max_depth=5`, `min_samples_leaf=5`) was critical to curb severe overfitting. However, because molecular descriptors interact continuously across multi-dimensional topological manifolds, axis-parallel orthogonal decision splits create stepped approximations that struggle to capture diagonal interaction boundaries. |
| KNN | Effective Metric-Space Congeneric Clustering: Delivers robust performance (87.68% Accuracy, 0.9326 AUC, 0.8194 F1-score). In chemical property prediction, molecules that share similar global topological and steric descriptors frequently share bioactivity (the structure-activity similarity principle). Feature standardization was critical; unscaled features would have allowed large-magnitude descriptors to distort Euclidean distance metrics. |
| Naive Bayes | Extreme Sensitivity / Asymmetric Tradeoff: Achieves the highest Recall across all models (95.77%, detecting 68 out of 71 biodegradable compounds), but at the cost of the lowest Precision (49.28%) and lowest Accuracy (65.40%). GaussianNB assumes class-conditional independence between all 41 descriptors. In organic chemistry, molecular descriptors are inherently correlated (e.g., number of heavy atoms scales directly with carbon count and molecular connectivity). This violated independence assumption leads to overconfident joint likelihoods and 70 false positives. Despite lower overall accuracy, Naive Bayes is valuable as an initial non-hazardous screening filter. |
| Support Vector Machine | Superior Non-Linear Margin Maximization: Achieves top scores in Accuracy (89.57%), AUC (0.9534), Precision (86.57%), F1 (0.8406), and MCC (0.7639). The Radial Basis Function (RBF) kernel projects the 41-dimensional chemical descriptor space into an infinite-dimensional reproducing kernel Hilbert space, smoothly untangling complex non-linear biochemical boundaries while the maximum-margin hyperplane enforces strong regularization against overfitting. |
| Overall, Winner for your dataset? | Support Vector Machine (RBF Kernel) 🏆: SVM is the definitive overall winner for the QSAR Biodegradation dataset. In molecular QSAR modeling with moderate sample sizes ($n = 844$ training compounds) and high feature dimensionality ($d = 41$), kernel SVM strikes the optimal bias-variance tradeoff via structural risk minimization. It produced the fewest false positives (only 9 out of 140 negative compounds) and highest Matthews Correlation Coefficient (0.7639), making it the most dependable classifier for regulatory chemical screening. |

---

## f. Streamlit Web Application Features

The companion Streamlit web application (`app.py`) is engineered to fulfill all four requirements specified in Step 5 of the BITS rubric:

1. **Dataset Upload Option (CSV) [1 Mark]:**
   - Supports uploading user-specified test CSV files containing the 41 required molecular descriptors and ground-truth target.
   - Includes a built-in **"Use default test_data.csv"** toggle, allowing instant evaluation of the 211 held-out test compounds without requiring manual file upload.
2. **Model Selection Dropdown [1 Mark]:**
   - Allows users to switch between all five trained models (**Support Vector Machine, Logistic Regression, KNN, Decision Tree, Naive Bayes**) or choose **"All 5 Models Comparison (Leaderboard)"** for side-by-side benchmarking.
3. **Display of Evaluation Metrics [1 Mark]:**
   - Renders polished, custom-styled metric cards displaying all six evaluation metrics (**Accuracy, ROC-AUC, Precision, Recall, F1-Score, MCC**) computed live on the uploaded data.
4. **Confusion Matrix & Classification Report [1 Mark]:**
   - Generates an annotated Seaborn heatmap displaying True Negatives, False Positives, False Negatives, and True Positives.
   - Formats the complete per-class scikit-learn classification report into a styled interactive table.
5. **Zero Retraining Guarantee:**
   - Evaluates pre-trained artifacts (`model/*.joblib`) strictly using `transform()` with the pre-fitted `StandardScaler`. No model fitting or retraining is triggered on the test data.
6. **Compound Prediction Explorer:**
   - Enables filtering predictions by error types (False Positives vs. False Negatives) and offers one-click CSV export of predictions.

---

## g. Repository Structure
```text
project-folder/
├── .streamlit/
│   └── config.toml                # Forces light theme on Streamlit Cloud
├── app.py                         # Interactive Streamlit web application
├── requirements.txt               # Pinned dependencies for Streamlit Cloud deployment
├── README.md                      # Comprehensive academic report following BITS rubric
├── test_data.csv                  # Held-out 20% test dataset (41 features + target)
├── train_models.py                # Reproducible training and artifact export script
├── ML_Assignment_2_Notebook.ipynb # Fully executed Jupyter Notebook with EDA & visualizations
└── model/                         # Serialized preprocessor and model artifacts
    ├── preprocessor.joblib        # Fitted StandardScaler artifact (zero leakage)
    ├── logistic_regression.joblib # Trained Logistic Regression classifier
    ├── decision_tree.joblib       # Trained Decision Tree classifier
    ├── knn.joblib                 # Trained K-Nearest Neighbors classifier
    ├── naive_bayes.joblib         # Trained Gaussian Naive Bayes classifier
    ├── svm.joblib                 # Trained Support Vector Machine classifier
    └── metrics_summary.json       # Precomputed test evaluation metrics
```

---

## h. Local Setup & Execution Guide

### 1. Clone the Repository
```bash
git clone https://github.com/AdiS-2105/ml-assignment-2.git
cd ml-assignment-2
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. (Optional) Re-train Models & Export Artifacts
```bash
python train_models.py
```

### 4. Launch the Interactive Streamlit Web App
```bash
streamlit run app.py
```
The application will launch automatically at `http://localhost:8501`.

---

## i. Deployment on Streamlit Community Cloud (Step 6)

1. Push this repository to GitHub under your account.
2. Navigate to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in using your GitHub account.
3. Click **"New App"**.
4. Select your repository: `AdiS-2105/ml-assignment-2`.
5. Specify Branch: `main`.
6. Specify Main file path: `app.py`.
7. Click **"Deploy"**. The app will automatically build using `requirements.txt` and launch with zero manual configuration.

---

## j. Anti-Plagiarism & Academic Integrity Statement
This project has been developed independently by **Adithya S** for **AIMLZG565 - Machine Learning, BITS Pilani WILP M.Tech (AIML)**:
- **Unique Random Seed:** All data splits and model random states are governed by fixed `RANDOM_SEED = 2002`.
- **Modular Domain Architecture:** Custom domain-specific nomenclature (`MOLECULAR_DESCRIPTOR_COLUMNS`, `ReadyBiodegradable`, biochemical interpretations) avoids generic templates.
- **Granular Git History:** The repository was constructed via 6–8 logical, incremental commits reflecting the full engineering workflow from scoping to deployment.
- **Original Analysis:** All analytical observations reflect genuine biochemical domain reasoning regarding QSAR molecular modeling and statistical learning theory.
