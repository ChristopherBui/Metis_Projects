import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
import itertools

# Import Data
df = pd.DataFrame.from_csv('credit_dollars.csv')
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

# Create Confusion Matrix Plotter
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=pylab.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    pylab.imshow(cm, interpolation='nearest', cmap=cmap)
    pylab.title(title)
    pylab.colorbar()
    tick_marks = np.arange(len(classes))
    pylab.xticks(tick_marks, classes, rotation=45)
    pylab.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pylab.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pylab.tight_layout()
    pylab.ylabel('True label')
    pylab.xlabel('Predicted label')



# GridSearchCV For Each Model

# Decision Tree
tree = DecisionTreeClassifier()
max_depth = [5,10,15,None]
max_features = [10,15,17,None]

tree_param = {'max_depth':max_depth, 'max_features':max_features}

tree_grid = GridSearchCV(tree, param_grid=tree_param, cv=5, scoring='f1')

tree_grid.fit(X,y)
tree_ypred = tree_grid.predict(X_test)


tree_yproba = tree_grid.predict_proba(X_test)[:,1]
fpr_tree, tpr_tree, _ = roc_curve(y_test, tree_yproba)

# Get Metric Scores; Decision Tree
print('best score: ',tree_grid.best_score_)
print(tree_grid.best_params_)
print('precision:',precision_score(y_test, tree_ypred),'\n','recall:',recall_score(y_test, tree_ypred),'\n','accuracy:',accuracy_score(y_test,tree_ypred),'\n','auc:',auc(fpr_tree,tpr_tree))

# Plot Decision Tree Confusion Matrix
cnf_tree = confusion_matrix(y_test, tree_ypred, labels=None)
plot_confusion_matrix(cnf_tree, title='Decision Tree CM', classes=tree_grid.classes_)
pylab.savefig('cnf_tree.png')



# Random Forest
rf = RandomForestClassifier()
rf_param = {}

rf_grid = GridSearchCV(rf, param_grid=rf_param, cv=5, scoring='f1')

rf_grid.fit(X,y)
rf_ypred = rf_grid.predict(X_test)


rf_yproba = rf_grid.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_yproba)

# Get Metric Scores; Random Forest
print('best score: ',rf_grid.best_score_)
print(rf_grid.best_params_)
print('precision:',precision_score(y_test, rf_ypred),'\n','recall:',recall_score(y_test, rf_ypred),'\n','accuracy:',accuracy_score(y_test,rf_ypred),'\n','auc:',auc(fpr_rf,tpr_rf))


# Plot Decision Tree Confusion Matrix
cnf_rf = confusion_matrix(y_test, rf_ypred, labels=None)
plot_confusion_matrix(cnf_rf, title='Random Forest CM', classes=rf_grid.classes_)
pylab.savefig('cnf_rf.png')

# Gradient Boosting is the best model, so it was implemented using a separate python script
