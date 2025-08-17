# Heart Disease Prediction with Decision Trees

<div align="center">

<h1 align="center"><strong>Heart Disease Prediction <h6 align="center">Using Decision Trees on UCI Dataset</h6></strong></h1>

![Python - Version](https://img.shields.io/badge/PYTHON-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/SCIKIT--LEARN-1.0+-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Optuna](https://img.shields.io/badge/OPTUNA-Tuning-blue?style=for-the-badge)

</div>

This project focuses on:

- Exploratory Data Analysis (EDA) of the UCI Heart Disease dataset
- Handling missing values with KNN imputation
- Building robust decision tree models for binary classification (heart disease presence)
- Hyperparameter tuning with Optuna and nested cross-validation
- Probability calibration and threshold optimization for medical applications
- Feature importance analysis using SHAP and tree-based methods

#### -- Project Status: [Completed]

#### -- UCI.ipynb - Contains the full code and analysis for the project


----

### 📋 Table of Contents
- [Project Overview](#project-overview)
  - [About the Dataset](#about-the-dataset)
  - [Preprocessing](#preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Model Approach](#model-approach)
  - [Evaluation](#evaluation)
- [EDA Summary](#eda-summary)
- [Imputation Strategy](#imputation-strategy)
- [Results](#results)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Observations and Learnings](#observations-and-learnings)

### 📌 Project Overview <a name="project-overview"></a>

This Jupyter notebook implements a complete pipeline for predicting heart disease using the UCI Heart Disease dataset. It emphasizes robust modeling for small datasets (303 samples), prioritizing recall for medical screening while using decision trees with calibration and optimization.

Key innovations:
- Nested CV for reliable evaluation on small data
- Optuna tuning with Fβ-score (β=2) for recall focus
- Isotonic calibration for reliable probabilities
- SHAP analysis for clinical interpretability

#### About the Dataset <a name="about-the-dataset"></a>

The UCI Heart Disease dataset contains 303 samples with 14 attributes, including age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol (chol), fasting blood sugar (fbs), resting ECG (restecg), max heart rate (thalach), exercise angina (exang), ST depression (oldpeak), slope of ST segment (slope), number of vessels (ca), thalassemia (thal), and target (num: 0-4 severity, binarized to 0/1).

- Continuous features: age, trestbps, chol, thalach, oldpeak
- Categorical features: sex, cp, fbs, restecg, exang, slope, ca, thal
- Target: Binary (0: no disease, 1: disease present)
- Issues: Small missing values in ca (4) and thal (2); handled via KNN imputation

#### Preprocessing <a name="preprocessing"></a>

- Missing values: KNN imputation with optimal k selected via CV (best k=10, accuracy=0.7857)
- Encoding: One-hot for categoricals like cp, restecg, slope, thal
- Scaling: StandardScaler for KNN
- Duplicates/Missing: No duplicates; missing filled to maintain categorical integrity
- Binarization: Target 'num' >0 as 1 (disease)

#### Feature Engineering <a name="feature-engineering"></a>

- Basic: Direct use of 13 features
- One-hot encoding for categoricals
- Advanced (explored but core is baseline): Potential additions like Framingham risk score, exercise stress score, etc., but primary focus on original features
- Interactions: Limited due to tree handling
- Statistical: Correlations, mutual info, chi-square tests to identify strong predictors (e.g., oldpeak, thalach, cp, ca, thal)

#### Model Approach <a name="model-approach"></a>

- Baseline: Simple DecisionTreeClassifier (accuracy ~75%)
- Advanced: Nested CV (4 outer/3 inner folds), Optuna (50 trials) tuning depth (3-6), min_samples_leaf (5-15), etc.
- Calibration: Isotonic on prefit tree
- Threshold: Optimized via F2-score (β=2) for recall priority (range 0.1-0.9)
- Final: Balanced class weights, pruning (ccp_alpha)
- Robustness: 10 seeds for evaluation

#### Evaluation <a name="evaluation"></a>

Metrics: Recall (priority), Precision, F1, F2 (β=2), ROC-AUC
- RMSLE not used; focus on classification metrics
- CV: StratifiedKFold or GroupKFold (age-based if applicable)
- Threshold analysis: Clinical (0.25), Screening (0.20), Youden's J, F2-optimized, Hybrid

----

## 🔍 EDA Summary <a name="eda-summary"></a>

### Key Findings

#### **Strong Predictors Identified**
- **Oldpeak** (ST depression): Strongest continuous predictor (r=0.504, p<0.001)
- **Thalach** (max heart rate): Strong negative correlation (r=-0.415, p<0.001)
- **CP** (chest pain): Highest mutual information (0.148) and chi2=89.0
- **CA** (vessels): Chi2=109.9 - most significant categorical predictor
- **Thal** (thalassemia): Chi2=98.0 - second strongest categorical predictor

#### **Weak/Insignificant Predictors**
- **Cholesterol**: Nearly no correlation (r=0.071, p=0.218) - surprisingly weak
- **FBS** (fasting blood sugar): Marginally significant (p=0.098)

#### **Data Quality Issues**
- **Missing values**: Only in CA (4 missing) and Thal (2 missing) - justifies sophisticated imputation
- **High multicollinearity**: VIF values 24-56 suggest feature redundancy, but acceptable for decision trees

#### **Distribution Insights**
- **Age**: Normal distribution (mean=54.4, std=9.0)
- **Gender imbalance**: 68% male (206 vs 97 female)
- **Chest pain**: Type 4 most common (144/303 cases)
- **CA vessels**: Most patients have 0 affected vessels (176/299 with data)

#### **Clinical Relevance**
- **Oldpeak** and **Thalach** being top predictors aligns with cardiology - exercise stress indicators
- **Chest pain type** and **vessel involvement** as strong predictors confirms medical intuition
- **Cholesterol** being weak predictor is medically interesting - may indicate other factors more critical

#### **Model Implications**
- Decision tree will likely split heavily on oldpeak, chest pain type, and vessel count
- High VIF acceptable since tree-based methods handle multicollinearity well
- Small missing data volume validates KNN imputation approach

Includes histograms, boxplots, heatmaps, point-biserial correlations, VIF, mutual info, and chi-square tests.

----

## 🛠 Imputation Strategy <a name="imputation-strategy"></a>

### Why KNN Imputation Over Traditional Methods for UCI Heart Disease Dataset

#### Dataset Context
- **Small dataset**: 303 rows (every sample is valuable)
- **Critical application**: Heart disease prediction (life-threatening consequences)
- **Missing features**: `ca` (major vessels) and `thal` (thalassemia) - both categorical medical indicators
- **Computational feasibility**: Small size allows sophisticated methods without resource constraints

#### Method Comparison

##### Traditional Methods (Rejected)
**Mean/Median Imputation:**
- Creates identical artificial values for all missing entries
- Reduces dataset variance artificially
- Ignores feature relationships
- Produces non-integer values for categorical medical features

**Mode Imputation:**
- Oversimplifies by using most frequent category
- Ignores patient-specific characteristics
- May not represent realistic medical profiles

##### KNN Imputation (Selected)

**Advantages:**
- **Preserves relationships**: Uses similar patients to infer missing values
- **Maintains variance**: Each imputed value is unique based on nearest neighbors
- **Categorical integrity**: Rounded results maintain medical validity
- **Small dataset friendly**: Doesn't artificially reduce data diversity
- **Model-optimized**: Cross-validation finds k that maximizes diagnostic accuracy
- **Computationally feasible**: Small dataset (303 rows) makes KNN + cross-validation quick and affordable

**Key Implementation Details:**
- **Scaling**: Required for distance-based KNN calculations
- **Optimal k selection**: Tested k=3-20 to find best model performance
- **Data leakage prevention**: Only impute originally missing values
- **Medical constraints**: Round categorical features to maintain clinical meaning

#### Conclusion
For small, critical medical datasets, the computational overhead of KNN imputation is negligible while providing significantly better statistical rigor and clinical validity compared to simple imputation methods that could compromise diagnostic accuracy.

----

## 💫 Results <a name="results"></a>

- Baseline Model: ~75% accuracy, Recall ~0.77 for disease class
- Optimized Model (Nested CV + Optuna): CV Recall 0.85 ± 0.03, Precision ~0.80, F2 ~0.84
- SHAP Top Features: thal_3.0, ca, oldpeak, cp_4, thalach
- Robustness (10 seeds): Mean F2 ~0.84, CV_F2 ~0.02 (low variation)
- Threshold Stability: ~0.25 ± 0.05
- Best Dataset: Advanced Ordinal (Mean F2: 0.845)

Top datasets from robust evaluation:
- advanced_ordinal: Mean_F2 0.845, Features 28
- ordinal_advanced: Mean_F2 0.842, Features 28
- baseline_advanced: Mean_F2 0.840, Features 32

### Clinical Model (Baseline Features) Interpretation

**Key Decision Paths:**
1. **High-Risk Path (100% HD probability):**
   - Thal > 0.5 (Abnormal thalassemia)
   - Cp > 0.5 (Atypical/non-anginal pain)
   - Oldpeak > 0.65 (Significant ST depression)
   → **Immediate treatment recommended**

2. **Low-Risk Path (97% no HD):**
   - Thal ≤ 0.5 (Normal thalassemia)
   - Oldpeak ≤ 1.7 (Mild ST depression)
   - Ca ≤ 0.5 (No major vessel involvement)
   - Age ≤ 58.5
   → **Low priority for intervention**

3. **Moderate-Risk Paths:**
   - Thal > 0.5 + Cp ≤ 0.5 + Ca > 0.5 + Oldpeak > 1.1 → 89% HD
   - Thal ≤ 0.5 + Oldpeak ≤ 1.7 + Ca > 0.5 + Restecg ≤ 0.5 → 64% HD

**Clinical Insights:**
1. **Thalassemia status is the primary discriminator** - Abnormal results immediately elevate risk
2. **Age matters only in low-risk scenarios** - Younger patients with clean markers have minimal risk
3. **ST depression (oldpeak) is crucial for risk stratification** - Values >1.7mm significantly increase HD probability
4. **ECG findings (restecg) modify risk** when vessels are involved
5. **Chest pain type affects prognosis** in thalassemia patients

### Optimized Model (Ordinal Features) Interpretation

**Key Decision Paths:**
1. **Highest Risk Path (91% HD probability):**
   - Cp > 2.5 (Asymptomatic chest pain)
   - Slope > 0.5 (Flat/downsloping ST segment)
   - Thalach ≤ 147.5 (Low maximum heart rate)
   → **Urgent cardiac evaluation needed**

2. **Low-Risk Paths:**
   - Cp ≤ 2.5 + Thal ≤ 0.5 + Sex ≤ 0.5 (Female) → 96% no HD
   - Cp ≤ 2.5 + Thal > 0.5 + Chol ≤ 228 → 72% no HD

3. **Gender-Specific Findings:**
   - Females with non-asymptomatic CP and normal thalassemia → Very low risk (96% no HD)
   - Males with same markers → Moderate risk (24% HD probability)

**Screening Insights:**
1. **Chest pain type is the primary screening filter** - Asymptomatic pain requires immediate attention
2. **ST slope is critical** - Abnormal slope with asymptomatic pain indicates high risk
3. **Cholesterol modifies risk** only in thalassemia patients
4. **Heart rate matters in high-risk groups** - Paradoxically lower rates indicate higher risk
5. **Gender significantly affects risk stratification**

### Critical Differences Between Models

| **Factor** | **Clinical Model** | **Optimized Model** |
|------------|--------------------|---------------------|
| **Primary Split** | Thalassemia status | Chest pain type |
| **Age Impact** | Strong in low-risk cases | Minimal impact |
| **Gender Role** | Not significant | Crucial modifier |
| **Key Metrics** | ST depression, Vessels | ST slope, Heart rate |
| **High-Risk Threshold** | Oldpeak > 0.65 | Thalach ≤ 147.5 |
| **Best For** | Diagnostic confirmation | Population screening |

### Clinical Recommendations

**Use Clinical Model When:**
- Confirming diagnosis in symptomatic patients
- Evaluating patients with known thalassemia
- Making treatment decisions for borderline cases
- Assessing older patients (age >58.5)

**Use Optimized Model When:**
- Screening general population
- Evaluating patients with vague symptoms
- Prioritizing resource allocation
- Assessing younger patients (<50 years)
- Working with female patients

----

## 🚀 Getting Started <a name="getting-started"></a>

### ✅ Prerequisites <a name="prerequisites"></a>

- **Dataset prerequisite for training**:

  Before running the notebook, make sure to download the UCI Heart Disease dataset from [here](https://archive.ics.uci.edu/dataset/45/heart+disease) or fetch it via ucimlrepo in the code.

- **Libraries**:
  - Python 3.11+
  - pandas, numpy, matplotlib, seaborn, scipy, sklearn, optuna, shap, joblib
  - Install via: `pip install ucimlrepo pandas numpy matplotlib seaborn scipy scikit-learn optuna shap joblib`

### 🐳 Setting up and Running the Project

1. Clone or download the repository.
2. Open `UCI.ipynb` in Jupyter Notebook or VS Code.
3. Run cells sequentially - it fetches the dataset automatically.
4. For custom runs: Adjust SEED, folds, trials in config sections.

Explore multiple datasets and robust evaluation in the main_workflow section.

----

## 📝 Observations and Learnings <a name="observations-and-learnings"></a>

- **Key Learnings**: Decision trees provide excellent interpretability for medical applications, making them useful for doctors to understand risk factors like thalassemia and chest pain. However, on small datasets like this (303 samples), overfitting is a major issue—techniques like nested cross-validation, Optuna hyperparameter tuning (exploring max_depth, min_samples_leaf, and ccp_alpha), and isotonic calibration were crucial to achieve stable performance (e.g., F2 ~0.84). Prioritizing recall via Fβ-score (β=2) aligns well with healthcare needs to minimize false negatives.
- **Challenges Faced**: Handling missing data in critical features (ca, thal) without introducing bias—overcame by using KNN imputation optimized via CV, which preserved relationships better than simple methods. Small dataset led to unstable metrics—addressed with 10-seed robustness testing and hybrid thresholds. Interpreting trees for real-world use required focusing on clinical paths and recommendations, ensuring models are not just accurate but actionable.
- **Overall Insights**: This assignment highlighted the balance between model complexity and simplicity; pruning and calibration made trees more generalizable and trustworthy for diagnostics. Future improvements could include ensemble methods like random forests for even better robustness.
