<div align=\"center\">

# Carbon Footprint Prediction (India)

[![Author](https://img.shields.io/badge/Author-Prakash%20Ukhalkar-blue.svg)](https://github.com/prakash-ukhalkar)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Latest-orange)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*An advanced Machine Learning and Explainable AI (XAI) pipeline to predict and interpret agricultural carbon emissions in India.*

</div>

---

## Overview

The **India Agri Carbon Prediction** project is a comprehensive research-driven machine learning pipeline aimed at predicting agricultural carbon footprints in India. By leveraging decades of historical data encompassing emissions, energy consumption, and natural resource utilization, this project builds robust predictive models and utilizes Explainable AI (XAI) to extract actionable policy recommendations.

The core pipeline transitions raw, fragmented datasets into a clean temporal format, constructs powerful predictive models using **XGBoost**, and demystifies the results using **SHAP (SHapley Additive exPlanations)**.

## Key Features

- **Automated Data Consolidation:** Merges 32 disparate datasets into 3 coherent thematic masters (Emissions, Energy & Activity, Land & Water Resources).
- **Advanced Feature Engineering:** Incorporates temporal dynamics by engineering lagged variables and momentum indicators, ensuring stationarity for time-series validity.
- **Predictive Modeling:** Deploys Baseline (Linear Regression) and Advanced (XGBoost) algorithms to forecast emission trends accurately.
- **Explainable AI (XAI) & Policy Insights:** Uses SHAP to crack open the \"black-box\" model, revealing high-impact drivers and simulating sustainable policy interventions (Scenario Analysis).

---

## Project Structure

```text
India-Agri-Carbon-Prediction/
│
├── data/                  # Contains raw and consolidated datasets
│   ├── raw/               # Original 32 Excel/CSV files across various themes
│   └── processed/         # Consolidated thematic master panels
│
├── notebooks/             # The core ML and Data Pipeline
│   ├── 01_Initial_Data_Consolidation.ipynb
│   ├── 02_Feature_Engineering_and_Target_Alignment.ipynb
│   ├── 03_Predictive_Modeling_and_Optimization.ipynb
│   └── 04_Interpretability_and_Policy_Insights.ipynb
│
├── models/                # Saved predictive model states
├── results/               # Outputs, SHAP plots, and evaluation metrics
├── scripts/               # Helper python scripts for the pipeline
│
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## Notebook Pipeline Guide

The analysis is broken down into four sequential Jupyter Notebooks to maintain research transparency and modularity:

1. **`01_Initial_Data_Consolidation`**  
   Automates the transformation of 32 noisy CSV/Excel files into three core DataFrames, standardizing \"Fiscal Year\" anomalies to ensure all files speak the same temporal language.

2. **`02_Feature_Engineering_and_Target_Alignment`**  
   Performs Exploratory Data Analysis (EDA). Implements rigorous stationarity testing, correlation analyses, and engineers essential features like historical lags to capture agricultural momentum.

3. **`03_Predictive_Modeling_and_Optimization`**  
   Splits the data temporally to prevent data leakage. Establishes a Linear Regression baseline before training an optimized XGBoost model to predict future emissions accurately.

4. **`04_Interpretability_and_Policy_Insights`**  
   Applies SHAP values to explain *why* the XGBoost model makes specific predictions. Deconstructs the model to attribute importance to factors like renewable energy or irrigation, directly translating mathematical output into actionable policy strategies.

---

## Installation & Setup

Ensure you have **Python 3.10+** installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/prakash-ukhalkar/India-Agri-Carbon-Prediction.git
   cd India-Agri-Carbon-Prediction
   ```

2. **Set up a Virtual Environment (Optional but recommended):**
   ```bash
   python -m venv venv_agri_carbon
   # On Windows:
   venv_agri_carbon\\Scripts\\activate
   # On macOS/Linux:
   source venv_agri_carbon/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebooks:**
   Launch Jupyter Notebook or Jupyter Lab and execute the notebooks in sequential order (01 → 04).
   ```bash
   jupyter notebook
   ```

---

## License

This project is licensed under the [MIT License](LICENSE).

---
<div align=\"center\">
<i>Developed for Sustainable Agricultural Research</i>
</div>
