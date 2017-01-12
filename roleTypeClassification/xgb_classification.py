
import pandas as pd
import numpy
import pickle
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold, train_test_split
import xgboost as xgb

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from general.filePath import *
# from utils import visualization


# IRMAS - 30
THRESHOLD = 20

labels = None


def feature_preprocessing(datafile):

    # import some data to play with
    data = pd.DataFrame.from_csv(datafile)

    # the class label, either dan or laosheng
    y = data['class']

    # feature matrix, axis 0: observation, axis 1: feature dimension
    X = data.drop(['class'], axis=1).values

    return X, y

def feature_scaling_train(X):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    pickle.dump(scaler,open(path.join(scaler_path,'feature_scaler.pkl'),'wb'))
    # X = scaler.transform(X)

    return

def feature_scaling_test(X):
    scaler = pickle.load(open(path.join(scaler_path,'feature_scaler.pkl'),'r'))
    X = scaler.transform(X)

    return X

def buildEstimators(mode):
    if mode == 'train' or mode == 'cv':
        # best parameters got by gridsearchCV, best score: 1
        estimators = [('anova_filter', SelectKBest(f_classif, k='all')),
                      ('xgb', xgb.XGBClassifier(learning_rate=0.1,n_estimators=300,max_depth=3))]
        clf = Pipeline(estimators)
    elif mode == 'test':
        clf = pickle.load(open(join(classifier_path,"xgb_classifier.plk"), "r"))
    return clf

def imputerLabelEncoder_train(X,y):
    imputer = preprocessing.Imputer()
    X = imputer.fit_transform(X)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    return X,y,imputer,le

def imputerLabelEncoder_test(X,y):
    imputer = pickle.load(open(join(classifier_path,"inputer.plk"),'r'))
    X = imputer.fit_transform(X)

    le = pickle.load(open(join(classifier_path,"le.plk"), "r"))
    y = le.fit_transform(y)
    return X,y

def imputer_run(X):
    imputer = pickle.load(open(join(classifier_path,"inputer.plk"),'r'))
    X = imputer.fit_transform(X)
    return X

def save_results(y_test, y_pred, labels, fold_number=0):
    pickle.dump(y_test, open("y_test_fold{number}.plk".format(number=fold_number), "w"))
    pickle.dump(y_pred, open("y_pred_fold{number}.plk".format(number=fold_number), "w"))
    print classification_report(y_test, y_pred)
    print confusion_matrix(y_test, y_pred)
    print "Micro stats:"
    print precision_recall_fscore_support(y_test, y_pred, average='micro')
    print "Macro stats:"
    print precision_recall_fscore_support(y_test, y_pred, average='macro')
    try:
        visualization.plot_confusion_matrix(confusion_matrix(y_test, y_pred),
                                            title="Test CM fold{number}".format(number=fold_number),
                                            labels=labels)
    except:
        pass


def train_test(clf, X, y, labels):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    save_results(y_test, y_pred, labels)

def train_evaluate_stratified(clf, X, y, labels):
    skf = StratifiedKFold(y, n_folds=10)
    for fold_number, (train_index, test_index) in enumerate(skf):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        save_results(y_test, y_pred, labels, fold_number)


def grid_search(clf, X, y):
    params = dict(anova_filter__k=[50, 100, 'all'],
                  xgb__max_depth=[3, 5, 10], xgb__n_estimators=[50, 100, 300, 500],
                  xgb__learning_rate=[0.05, 0.1])
    gs = GridSearchCV(clf, param_grid=params, n_jobs=4, cv=10, verbose=2)
    gs.fit(X, y)

    print "Best estimator:"
    print gs.best_estimator_
    print "Best parameters:"
    print gs.best_params_
    print "Best score:"
    print gs.best_score_

    y_pred = gs.predict(X)
    y_test = y

def train_save(clf, X, y, le, inputer):
    clf.fit(X, y)
    pickle.dump(clf, open(join(classifier_path,"xgb_classifier.plk"), "w"))
    pickle.dump(le, open(join(classifier_path,"le.plk"), "w"))
    pickle.dump(inputer, open(join(classifier_path,"inputer.plk"), "w"))

def prediction(clf, X, y):
    y_pred = clf.predict(X)
    y_test = y
    print classification_report(y_test, y_pred)
    # print confusion_matrix(y_test, y_pred)
    print "Micro stats:"
    print precision_recall_fscore_support(y_test, y_pred, average='micro')
    print "Macro stats:"
    print precision_recall_fscore_support(y_test, y_pred, average='macro')

if __name__ == "__main__":
    datafile,dataset = [],[]
    datafile.append(sys.argv[1])
    datafile.append(sys.argv[2])
    dataset.append(sys.argv[3])
    dataset.append(sys.argv[4])
    mode=sys.argv[5]

    X0, y0 = feature_preprocessing(datafile[0])
    X1, y1 = feature_preprocessing(datafile[1])

    X = numpy.vstack((X0,X1))
    y = numpy.hstack((y0,y1))

    if mode == 'train' or mode == 'cv':
        feature_scaling_train(X)

    X = feature_scaling_test(X)

    if mode == 'train' or mode == 'cv':
        X,y,imputer,le = imputerLabelEncoder_train(X,y)
    elif mode == 'test':
        X,y = imputerLabelEncoder_test(X,y)

    # print X,y
    clf = buildEstimators(mode)

    if mode == 'cv':
        grid_search(clf,X,y)
    elif mode == 'train':
        train_save(clf, X, y, le, imputer)
    elif mode == 'test':
        prediction(clf, X, y)
