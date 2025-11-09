import numpy as np
import pandas as pd
import pickle
import os

from sklearn.ensemble import GradientBoostingClassifier

parent_dir = os.getcwd()

data_path = os.path.join(parent_dir,"data")

# fetch the data from data/processed
train_data = pd.read_csv(os.path.join(data_path,"interim","train_bow.csv"))

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

# Define and train the XGBoost model

clf = GradientBoostingClassifier(n_estimators=10)
clf.fit(X_train, y_train)

models_dir_path = os.path.join(parent_dir,"models")
os.makedirs(models_dir_path,exist_ok=True)

# save
pickle.dump(clf, open(os.path.join(models_dir_path,'model.pkl'),'wb'))