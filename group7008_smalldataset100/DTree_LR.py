from sklearn.model_selection    import GridSearchCV
from sklearn.tree               import DecisionTreeClassifier
from sklearn.linear_model       import LogisticRegression
import pandas   as pd
import argparse 
import pickle
import time

# Parser for command-line options, arguments and sub-commands
temp =  argparse.ArgumentParser()

# Decision Tree Classifier
dtree= DecisionTreeClassifier(random_state=7)
X_train= pd.read_csv('data/X_train.csv')
y_train= pd.read_csv('data/y_train.csv')

# record start time
start_time = time.time()

##grid search for optimum parameters
params = {'criterion': ['gini', 'entropy', 'log_loss'],
          'splitter': ['best', 'random'],
          'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5], 
          'min_samples_leaf':[1,2,3]}
dtree_clf = GridSearchCV(dtree, param_grid=params, n_jobs=-1)
# train the model
dtree_clf.fit(X_train,y_train)
dtree_clf.best_params_ 

# Logistic Regression
# search for optimun parameters using gridsearch
params = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
          'penalty':['l2','l2'],
          'C':[0.01,0.1,1,10,100],
          'class_weight':['balanced',None]}
logistic_clf = GridSearchCV(LogisticRegression(),param_grid=params,cv=10,n_jobs=-1)
logistic_clf.fit(X_train,y_train)
logistic_clf.best_params_

# record end time 
end_time = time.time()
time_cost = end_time - start_time

# save the model
with open('pkl/dtree_lr_clf.pkl','wb') as f:
    pickle.dump((dtree_clf, logistic_clf, time_cost), f)