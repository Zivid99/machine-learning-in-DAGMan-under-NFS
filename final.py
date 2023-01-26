from sklearn.model_selection    import GridSearchCV
from sklearn.tree               import DecisionTreeClassifier
from sklearn.metrics            import accuracy_score
from sklearn.metrics            import confusion_matrix
from sklearn.metrics            import f1_score
from sklearn.metrics            import classification_report
from sklearn.metrics            import classification_report,roc_auc_score
import pandas   as pd
import argparse
import joblib 
import pickle
import time

X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')
y_test = y_test['HeartDisease']

with open("pkl/dtree_clf.pkl", "rb") as f:
    dtree_clf, lr_time = pickle.load(f)
dtree_predict = dtree_clf.predict(X_test)
dtree_accuracy = accuracy_score(y_test,dtree_predict)
cm=confusion_matrix(y_test,dtree_predict)
dtree_f1 = f1_score(y_test, dtree_predict)
probs = dtree_clf.predict_proba(X_test)
probs = probs[:, 1]
dtree_auc = roc_auc_score(y_test, probs)
print('#################Decision Trees################')
print(f"Using Decision Trees we get an accuracy of {round(dtree_accuracy*100,2)}%")
print(classification_report(y_test,dtree_predict))
print(f'f1 score: {round(dtree_f1*100,2)}%')
print(f'auc_roc: {round(dtree_auc*100,2)}%')
print('================================================')
print(f"Time cost: ", lr_time)
print()
with open("pkl/lr_clf.pkl", "rb") as f:
    logistic_clf, dtree_time = pickle.load(f)
# make predictions
logistic_predict = logistic_clf.predict(X_test)
log_accuracy = accuracy_score(y_test,logistic_predict)
cm=confusion_matrix(y_test,logistic_predict)
logistic_f1 = f1_score(y_test, logistic_predict)
logistic_auc = roc_auc_score(y_test, logistic_predict)
print('##################Logistic Regression###################')
print(f"Logistic Regression:")

print(f"Using logistic regression we get an accuracy of {round(log_accuracy*100,2)}%")
print(classification_report(y_test,logistic_predict))
print(f'f1 score: {round(logistic_f1*100,2)}%')
print(f'auc_roc: {round(logistic_auc*100,2)}%')
print('========================================================')
print(f"Time cost: ", dtree_time)
print()
print(f'DTree time + LR time: ', dtree_time+lr_time)


with open("pkl/dtree_lr_clf.pkl", "rb") as f:
    dtree2_clf, logistic2_clf, mix_time = pickle.load(f)
# predictions
dtree_predict = dtree2_clf.predict(X_test)
dtree_accuracy = accuracy_score(y_test,dtree_predict)
cm=confusion_matrix(y_test,dtree_predict)
dtree_f1 = f1_score(y_test, dtree_predict)
probs = dtree2_clf.predict_proba(X_test)
probs = probs[:, 1]
dtree_auc = roc_auc_score(y_test, probs)
print('')
print('########Decision Trees in mix mode#########')
print(f"Using Decision Trees we get an accuracy of {round(dtree_accuracy*100,2)}%")
print(classification_report(y_test,dtree_predict))
print(f'f1 score: {round(dtree_f1*100,2)}%')
print(f'auc_roc: {round(dtree_auc*100,2)}%')
# print('=============================================')
# print(f"Time cost: ", mix_time)
print()

# make predictions
logistic_predict = logistic2_clf.predict(X_test)
log_accuracy = accuracy_score(y_test,logistic_predict)
cm=confusion_matrix(y_test,logistic_predict)
logistic_f1 = f1_score(y_test, logistic_predict)
logistic_auc = roc_auc_score(y_test, logistic_predict)

print('#######Logistic Regression in mix mode########')
print(f"Logistic Regression:")
print(f"Using logistic regression we get an accuracy of {round(log_accuracy*100,2)}%")
print(classification_report(y_test,logistic_predict))
print(f'f1 score: {round(logistic_f1*100,2)}%')
print(f'auc_roc: {round(logistic_auc*100,2)}%')
print('==================================================')
print(f"Time cost in tatal: ", mix_time)
print()