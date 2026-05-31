# engg2112
This GitHub repository is made for the ML project: Smart Charge AI. It contains three different ML modules following a Pipeline structure:
- Step 1: Classification (RandomForest), Pre-Flight Check for user charging behaviour
- Step 2: Regression (RandomForest), Detailed battery damage per charge cycle.
- Step 3 & 4: Regression (RandomForest), Long-term battery health Forecasting / Linking to financial cost

# Contents:
**Code:**
ML_combine_Step1_Step2.ipynb: Machine Learning Module Step 1 + Step 2 MVP & User Input

final_battery_integrated_step3_step4_PRESENTATION_DATA_v5.py: Machine Learning Module Step 3 + Step 4 & User Input

**Datasets:**
production_ev_battery_data.csv: Dataset used in Step 1 (Tuned & Added Noise Ver.), The original dataset is from [Kaggle](https://www.kaggle.com/datasets/ziya07/ev-battery-charging-data)

step2_physics_data.csv: Dataset used in Step 2 (Synthetic data). The original untuned dataset is from  [BatteryArchive](https://www.batteryarchive.org)

Cell1_SOH_analysis.csv: Dataset used in Step 3, extracted from the Oxford Battery Dataset [Oxford](https://doi.org/10.5287/bodleian:KO2kdmYGg)
