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


# GridSearchCV For Gradient Boosting
grad = GradientBoostingClassifier()
grad_param = {'n_estimators':[300], 'max_depth':[3], 'max_features':[None]}

grad_grid = GridSearchCV(grad, param_grid=grad_param, cv=5, scoring='f1')

grad_grid.fit(X,y)
grad_ypred = grad_grid.predict(X_test)


grad_yproba = grad_grid.predict_proba(X_test)[:,1]
fpr_grad, tpr_grad, _ = roc_curve(y_test, grad_yproba)

# Find Best Feature Contributing To The Model
flist = grad_grid.best_estimator_.feature_importances_
bfi = np.where(flist==max(flist))
print(flist)
print('bfi:',bfi)

# Get Metric Scores; Gradient Boosting
print('best score: ',grad_grid.best_score_)
print(grad_grid.best_params_)
print('precision:',precision_score(y_test, grad_ypred),'\n','recall:',recall_score(y_test, grad_ypred),'\n','accuracy:',accuracy_score(y_test,grad_ypred),'\n','auc:',auc(fpr_grad,tpr_grad))

# Plot Confusion Matrix
cnf_grad = confusion_matrix(y_test, grad_ypred, labels=None)
plot_confusion_matrix(cnf_grad, title='Gradient Boosting CM', classes=grad_grid.classes_)
pylab.savefig('cnf_grad.png')

pylab.figure(figsize=(10,10))
pylab.plot(fpr_grad, tpr_grad, label='gradient boosting')
pylab.xlabel('FPR', labelpad=10)
pylab.ylabel('TPR',rotation=0, labelpad=15)
pylab.legend(loc='upper left')
pylab.title('Gradient Boosting Tuned')
pylab.savefig('gbroctuned.png')

# Implementing thresholding with various threshold values

def threshold(threshold):
    threshold = threshold
    pred_proba = grad_grid.predict_proba(X_test)
    pred = (pred_proba[:,1]>=threshold).astype('int')

    # return new metric scores after thesholding
    print('f1-2:',f1_score(y_test,pred),'\n','p2:',precision_score(y_test, pred),'\n','re2:',recall_score(y_test, pred),'\n','ac2:',accuracy_score(y_test, pred),'\n','auc2:',auc(fpr_grad,tpr_grad))

# Example run with threshold = 0.3
threshold(0.3)
