import numpy as np
import pandas as pd

import pickle
import json
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


parent_dir = os.getcwd()

models_dir_path = os.path.join(parent_dir,"models")

data_path = os.path.join(parent_dir,"data")

clf = pickle.load(open(os.path.join(models_dir_path,'model.pkl'),'rb'))

test_data = pd.read_csv(os.path.join(data_path,"interim","test_bow.csv"))

X_test = test_data.iloc[:,0:-1].values
y_test = test_data.iloc[:,-1].values

y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)


metrics_dict={
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'auc':auc
}

reports_dir_path = os.path.join(parent_dir,"reports")

os.makedirs(reports_dir_path,exist_ok=True)

with open(os.path.join(reports_dir_path,'metrics.json'), 'w') as file:
    json.dump(metrics_dict, file, indent=4)