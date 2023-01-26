from sklearn.model_selection    import GridSearchCV
from sklearn.tree               import DecisionTreeClassifier
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

##grid search for optimum parameters
params = {'criterion': ['gini', 'entropy', 'log_loss'],
          'splitter': ['best', 'random'],
          'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5,6,7,8], 
          'min_samples_leaf':[1,2,3,4,5]}
dtree_clf = GridSearchCV(dtree, param_grid=params, n_jobs=-1)
# dtree_clf = dtree
# record start time
start_time = time.time()

# train the model
dtree_clf.fit(X_train,y_train)
# dtree_clf.best_params_ 

# record end time 
end_time = time.time()
time_cost = end_time - start_time

# save the model
with open('pkl/dtree_clf.pkl','wb') as f:
    pickle.dump((dtree_clf,time_cost), f)
    