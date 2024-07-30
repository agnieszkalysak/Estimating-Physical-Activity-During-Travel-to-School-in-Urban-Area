# Estimating Physical Activity During Travel to School in Urban Area

This repository contains the code for estimating children PA during commuting to school, with usage of three explainable models: Decision Tree, Random Forest and XGBoost.

## Running

To run the code, you just need simply run Python script with the model of you interest. For example, to check Decision Tree model's results, you should run:

```
python decision_tree.py
```

Inside the script, you need to specify three paths, to **_data_**, **_models_** and **_results_** directories. Inside the **_data_** directory should be placed dataset as a [pickle]([https://github.com](https://docs.python.org/3/library/pickle.html)). The images with results will be placed automatically in the **_restuls_** path. Final models would be saved in **_models_** path, also as pickle serialized object.
