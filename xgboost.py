import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
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
PATH_TO_RESULTS = 'results/'
PATH_TO_MODELS = 'models/'

with open(os.path.join(PATH_TO_DATA, 'train_test_data.pkl'), 'rb') as file:
    X_train, X_test, y_train, y_test = pickle.load(file)

label_mapping = {
    'CAR': 0,
    'WALKING_ONLY': 1,
}

y_train = y_train.map(label_mapping)
y_test = y_test.map(label_mapping)

dt = XGBClassifier()

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.1, 0.5, 1.0],
    'min_child_weight': [1, 5],
    'scale_pos_weight': [1, 2, 3],
    'max_delta_step': [1],
    'colsample_bylevel': [0.6, 0.7, 0.8, 1.0],
    'colsample_bynode': [0.6, 0.7, 0.8, 1.0],
    'tree_method': ['auto']
}

grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, scoring='balanced_accuracy', verbose=1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

best_dt = grid_search.best_estimator_

best_dt.fit(X_train, y_train)

y_pred = best_dt.predict(X_test)

importances = best_dt.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = X_train.columns[indices]
sorted_importances = importances[indices]
top_n = 20
sorted_features = sorted_features[:top_n]
sorted_importances = sorted_importances[:top_n]
sorted_features = sorted_features[::-1]
sorted_importances = sorted_importances[::-1]

plt.figure(figsize=(25, 45), dpi=300)
plt.barh(sorted_features, sorted_importances)
plt.xticks(fontsize=65)
plt.yticks(fontsize=65)
plt.xticks(fontsize=55)
plt.yticks(fontsize=55)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(PATH_TO_RESULTS, 'xgboost_importances.pdf'), bbox_inches='tight')
plt.close()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CAR', 'WALKING_ONLY'])
disp.plot(cmap=plt.cm.Blues)
disp.ax_.images[-1].colorbar.remove()
plt.savefig(os.path.join(PATH_TO_RESULTS, 'xgboost_confusion_matrix.png'), bbox_inches='tight')
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
plt.savefig(os.path.join(PATH_TO_RESULTS, 'xgboost_classification_report.png'), bbox_inches='tight')
plt.close()

f = lambda x: best_dt.predict_proba(x)[:, 1]
med = X_test.median().values.reshape((1, X_test.shape[1]))
explainer = shap.KernelExplainer(f, med)
shap_values = explainer.shap_values(X_test, nsamples=100)
mean_shap_values = np.abs(shap_values).mean(axis=0)
top_indices_shap = np.argsort(mean_shap_values)[-top_n:]
top_shap_values = shap_values[:, top_indices_shap]
top_shap_features = X_test.columns[top_indices_shap]
shap_values_exp = shap.Explanation(values=top_shap_values, base_values=explainer.expected_value, data=X_test.iloc[:, top_indices_shap], feature_names=top_shap_features)
shap_fig = plt.figure(figsize=(25, 45), dpi=300)
shap.plots.beeswarm(shap_values_exp, max_display=top_n, show=False, plot_size=1, s=100, color_bar_label='')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('')
plt.ylabel('')
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=20)
shap_fig.savefig(os.path.join(PATH_TO_RESULTS, 'xgboost_shap.pdf'), bbox_inches='tight')
plt.close(shap_fig)

with open(os.path.join(PATH_TO_MODELS, 'xgboost.pkl'), 'wb') as f:
    pickle.dump(best_dt, f)
