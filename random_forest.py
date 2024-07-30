import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.exceptions import DataConversionWarning
import shap
import matplotlib.pyplot as plt
import warnings
import pickle
import os

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

PATH_TO_DATA = 'data/'
PATH_TO_RESULTS = 'results/selected_data_to_from/random_forest/'
PATH_TO_MODELS = 'models/selected_data/'

with open(os.path.join(PATH_TO_DATA, 'train_test_data.pkl'), 'rb') as file:
    X_train, X_test, y_train, y_test = pickle.load(file)

dt = RandomForestClassifier()

param_grid = {
    'n_estimators': [10, 25, 50],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 7, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'max_features': [None],
    'bootstrap': [True],
    'class_weight': ['balanced_subsample'],
    'max_samples': [None],
    'max_leaf_nodes': [None, 10, 20, 30, 40],
    'min_impurity_decrease': [0, 0.01, 0.005]
}

grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, scoring='balanced_accuracy', verbose=1)
grid_search.fit(X_train, y_train)

best_dt = grid_search.best_estimator_
print(best_dt)

best_dt.fit(X_train, y_train)

y_pred = best_dt.predict(X_test)

importances = best_dt.feature_importances_
indices = np.argsort(importances)
sorted_features = X_train.columns[indices]
sorted_importances = importances[indices]

plt.figure(figsize=(25, 25), dpi=300)
plt.barh(sorted_features, sorted_importances)
plt.xlabel('Feature Importance')
plt.savefig(os.path.join(PATH_TO_RESULTS, 'importances.png'), bbox_inches='tight')
plt.close()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_dt.classes_)
disp.plot(cmap=plt.cm.Blues)
disp.ax_.images[-1].colorbar.remove()
plt.savefig(os.path.join(PATH_TO_RESULTS, 'confusion_matrix.png'), bbox_inches='tight')
plt.close()

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.axis('tight')
table = ax.table(cellText=report_df.values,
                 colLabels=report_df.columns,
                 rowLabels=report_df.index,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
plt.savefig(os.path.join(PATH_TO_RESULTS, 'classification_report.png'), bbox_inches='tight')
plt.close()

f = lambda x: best_dt.predict_proba(x)[:, 1]
med = X_test.median().values.reshape((1, X_test.shape[1]))
explainer = shap.KernelExplainer(f, med)
shap_values = explainer.shap_values(X_test, nsamples=100)
shap_values_exp = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=X_test, feature_names=X_test.columns)

shap_fig = plt.figure(figsize=(25, 45), dpi=300)
shap.plots.beeswarm(shap_values_exp, max_display=200, show=False, plot_size=1)
shap_fig.savefig(os.path.join(PATH_TO_RESULTS, 'shap.png'), bbox_inches='tight')
plt.close(shap_fig)

with open(os.path.join(PATH_TO_MODELS, 'best_rf_selected_data_to_from.pkl'), 'wb') as f:
    pickle.dump(best_dt, f)
