import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score

# Quickly plot ROC Curve and calculate AUC score for several algorithms to determine the best model


# Import Data
df = pd.DataFrame.from_csv('credit_dollars.csv')
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_ypred = knn.predict(X_test)
knn_proba = knn.predict_proba(X_test)[:,1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_proba)
knn_auc = auc(fpr_knn, tpr_knn)
knn_f1 = f1_score(y_test, knn_ypred)

# SVM
vector = svm.SVC(kernel='linear', C=0.5)
vector.fit(X_train, y_train)
vector_ypred = vector.predict(X_test)
vector_proba = vector.predict_proba(X_test)[:,1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, vector_proba)
svm_auc = auc(fpr_svm, tpr_svm)
svm_f1 = f1_score(y_test, vector_ypred)


# Decision Tree
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_ypred = tree.predict(X_test)
tree_proba = tree.predict_proba(X_test)[:,1]
fpr_tree, tpr_tree, _ = roc_curve(y_test, tree_proba)
tree_auc = auc(fpr_tree, tpr_tree)
tree_f1 = f1_score(y_test, tree_ypred)

# Random Forest
forest = RandomForestClassifier(n_estimators=100, max_features=20)
forest.fit(X_train, y_train)
forest_ypred = forest.predict(X_test)
forest_proba = forest.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, forest_proba)
forest_auc = auc(fpr_rf, tpr_rf)
forest_f1 = f1_score(y_test, forest_ypred)

# Gradient Boosting
grad = GradientBoostingClassifier()
grad.fit(X_train, y_train)
grad_ypred = grad.predict(X_test)
grad_proba = grad.predict_proba(X_test)[:,1]
fpr_gb, tpr_gb, _ = roc_curve(y_test, grad_proba)
grad_auc = auc(fpr_gb, tpr_gb)
grad_f1 = f1_score(y_test, grad_ypred)


print('F1 SCORES',
'\n',
'KNN:',knn_f1,
'\n',
'TREE:',tree_f1,
'\n',
'FOREST:',forest_f1,
'\n',
'GRAD:',grad_f1)


print('AUC SCORES',
'\n',
'KNN:',knn_auc,
'\n',
'TREE:',tree_auc,
'\n',
'FOREST:',forest_auc,
'\n',
'GRAD:',grad_auc)


# Gradient Boosting seems to work significantly better, so we'll tune the model using Gradient Boosting going forward

pylab.figure(figsize=(10,10))
pylab.plot(fpr_knn, tpr_knn, label='knn')
pylab.plot(fpr_svm, tpr_svm, label='svm-lin')
pylab.plot(fpr_tree, tpr_tree, label='decision tree')
pylab.plot(fpr_rf, tpr_rf, label='random forest')
pylab.plot(fpr_gb, tpr_gb, label='gradient boosting')
pylab.plot([0,1],[0,1], linestyle='dashed')
pylab.xlabel('FPR', labelpad=10)
pylab.ylabel('TPR',rotation=0, labelpad=15)
pylab.legend(loc='upper left')
pylab.title('ROC Curves')
pylab.savefig('roc.png')
