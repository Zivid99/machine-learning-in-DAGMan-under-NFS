from sklearn.model_selection    import GridSearchCV
from sklearn.linear_model       import LogisticRegression
import pandas as pd
import argparse
import pickle
import time

temp =  argparse.ArgumentParser()

# read data
X_train= pd.read_csv('data/X_train.csv')
y_train= pd.read_csv('data/y_train.csv')

y_train = y_train['HeartDisease']
# y_test = y_test['HeartDisease']

# Logistic Regression
# search for optimun parameters using gridsearch
params = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
          'penalty':['l2','l2'],
          'C':[0.01,0.1,1,10,100],
          'class_weight':['balanced',None]}
logistic_clf = GridSearchCV(LogisticRegression(),param_grid=params,cv=5,n_jobs=-1)

# record start time
start_time = time.time()

## train the classifier
logistic_clf.fit(X_train,y_train)
logistic_clf.best_params_

# record end time 
end_time = time.time()
time_cost = end_time - start_time

# save the model
with open('pkl/lr_clf.pkl','wb') as f:
    pickle.dump((logistic_clf, time_cost), f)