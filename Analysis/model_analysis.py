import sys
sys.path.append("/Users/agatorano/Code/METIS/Census_Insurance_Prediction/Create_Database")
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import random

from clean_data import *

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.learning_curve import learning_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sqlalchemy import create_engine


def get_data():

    '''
    get data from csv
    '''

    i = 0
    data = pd.DataFrame()
    for chunk in pd.read_csv('../ss13pusa.csv', chunksize=5000, header=0):
        if i == 10:
            break
        data = pd.concat([data, chunk])
        i += 1

    data = clean_chunk(data)
    data['HICOV'] = data.HICOV.map(lambda x: 1 if x == 1 else 0)

    y = data.HICOV
    X = data.drop('HICOV', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=4444)

    return X_train, X_test, y_train, y_test


def test_random_forest_n_estimators_parameter(params):
    '''
    identify optimum number of estimaters
    for random forest
    '''

    x = y = z = []
    for param in params:
        x.append(param)
        model = RandomForestClassifier(n_estimators=param)
        fitted = model.fit(X_train, y_train)
        y.append(accuracy_score(y_train, fitted.predict(X_train)))
        z.append(accuracy_score(y_test, fitted.predict(X_test)))
    tr, = plt.plot(x, y)
    ts, = plt.plot(x, z)
    plt.legend((tr, ts), ('Training Accuracy', 'Test Accuracy'), loc='best')
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy')
    plt.title("Random Forest Accuracy vs Estimator Parameter")
    plt.savefig("n_estimators.png")


def test_random_forest_max_depth_parameter(params):
    '''
    identify max depth parameter
    for random forest
    '''
    x = y = z = []
    for param in params:
        x.append(param)
        model = RandomForestClassifier(max_depth=param)
        fitted = model.fit(X_train, y_train)
        y.append(accuracy_score(y_train, fitted.predict(X_train)))
        z.append(accuracy_score(y_test, fitted.predict(X_test)))
    tr, = plt.plot(x, y)
    ts, = plt.plot(x, z)
    plt.legend((tr, ts), ('Training Accuracy', 'Test Accuracy'), loc='best')
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.title("Random Forest Accuracy vs Max Depth")
    plt.savefig("rf_max_depth.png")


def test_random_forest_min_samples_split_parameter(params):
    '''
    identify min sample split
    for random forest
    '''
    x = y = z = []
    for param in params:
        x.append(param)
        model = RandomForestClassifier(min_samples_split=param)
        fitted = model.fit(X_train, y_train)
        y.append(accuracy_score(y_train, fitted.predict(X_train)))
        z.append(accuracy_score(y_test, fitted.predict(X_test)))
    tr, = plt.plot(x, y)
    ts, = plt.plot(x, z)
    plt.legend((tr, ts), ('Training Accuracy', 'Test Accuracy'), loc='best')
    plt.xlabel('min_samples_split')
    plt.ylabel('Accuracy')
    plt.title("Random Forest Accuracy vs Min Sample Split")
    plt.savefig("rf_min_sample.png")


def test_logistic_regression_c_parameter(params):
    x = y = z = []
    for param in params:
        x.append(param)
        model = LogisticRegression(C=param)
        fitted = model.fit(X_train, y_train)
        y.append(accuracy_score(y_train, fitted.predict(X_train)))
        z.append(accuracy_score(y_test, fitted.predict(X_test)))
    tr, = plt.plot(x, y)
    ts, = plt.plot(x, z)
    plt.legend((tr, ts), ('Training Accuracy', 'Test Accuracy'), loc='best')
    plt.xlabel('C')
    plt.ylabel('Accuracy')
    plt.title("Logistic Accuracy vs. C")
    plt.savefig("logistic_c_param.png")


def plot_learning_curve(model, type_model):
    m, train_scores, valid_scores = learning_curve(estimator=model,
                                                   X=X_train, y=y_train.ravel(),
                                                   train_sizes=np.linspace(0.1, 1.0, 80))

    train_cv_err = np.mean(train_scores, axis=1)
    test_cv_err = np.mean(valid_scores, axis=1)
    tr, = plt.plot(m, train_cv_err)
    ts, = plt.plot(m, test_cv_err)
    plt.legend((tr, ts), ('training error', 'test error'), loc='best')
    plt.title('Learning Curve %s' % type_model)
    plt.xlabel('Data Points')
    plt.ylabel('Accuracy')
    plt.savefig("%s_Learning_Curve.png" % type_model.replace(" ", ""))


def plot_ROC_curve(model, X_train, X_test, y_train, y_test):
    """Function to plot an ROC curve
    Compute ROC curve and ROC area for each class
    Learn to predict each class against the other
    """

    print model
    print "*************************** Model Metrics *********************************"
    print 'Accuracy: %s' % cross_val_score(model, X_train, y_train, scoring='accuracy', cv=5).mean()
    print 'Precision: %s' % cross_val_score(model, X_train, y_train, scoring='precision', cv=5).mean()
    print 'Recall: %s' % cross_val_score(model, X_train, y_train, scoring='recall_weighted', cv=5).mean()
    print 'F1: %s' % cross_val_score(model, X_train, y_train, scoring='f1', cv=5).mean()

    fitted = model.fit(X_train, y_train)
    try:
        y_score = fitted.predict_proba(X_test)[:,1]
    except:
        y_score = fitted.decision_function(X_test)

    # Confusion matrix
    print "********************* Normalized Confusion Matrix *************************"
    cm = confusion_matrix(y_test, fitted.predict(X_test))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')

    # Classification Report
    print "********************* Classification Report********************************"    
    print classification_report(y_test, fitted.predict(X_test))

    print "********************* ROC Curve *******************************************"

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def get_top_features_rf(X_train, X_test, y_train, y_test):
    '''
    plot the top 10 features as
    assigned by the random forest
    '''

    model = RandomForestClassifier(n_estimators=5, max_depth=12, min_samples_split=5)
    model.fit(X_train, y_train)
    model.predict(X_test)
    l = [(feat, lab) for lab, feat in zip(X.columns, model.feature_importances_)]
    l = sorted(l, reverse=True)
    l[:10]
    ax = k.plot(kind='bar', alpha=0.7, title="Top Features")
    ax.grid(False)
    fig = ax.get_figure()
    fig.savefig('top_features.pdf')


def main():

    X_train, X_test, y_train, y_test = get_data()
    test_random_forest_n_estimators_parameter(range(1, 20))
    test_random_forest_max_depth_parameter(range(1, 20))
    test_random_forest_min_samples_split_parameter(range(2, 20))
    test_logistic_regression_c_parameter([.001, .01, .1, 1, 10])
    plot_learning_curve(RandomForestClassifier(n_estimators=5, max_depth=12, min_samples_split=5), "Random Forest")
    plot_learning_curve(LogisticRegression(C=0.01), "Logistic Regression")
    plot_learning_curve(DecisionTreeClassifier(min_samples_split=100), "Decison Tree")
    plot_learning_curve(BernoulliNB(), "Bournouli")
    get_top_features_rf(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
