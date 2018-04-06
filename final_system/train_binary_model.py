import sklearn
import pickle
import os, sys

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV

from sklearn.utils import shuffle
sys.path.append('../../basic')
sys.path.append('../../basic/training')
sys.path.append('../../feature_extraction')

import train_utils

DATADIR = './data'
MODELDIR = './saved_models'
assert os.path.exists(DATADIR)
assert os.path.exists(MODELDIR)

label_mapping = {'none': 0,
                'do':1,
                'reason': 2,
                'fr': 3,
                'severity_type': 4,
                'manner/route': 5,
                'adverse': 6,
                'du': 7
                }

def train_models(X, y):
    """
    Tries several classifications, saves cross-validated scores.
    """
    f = open('binary_model_scores.txt', 'w')
    clfs = [LogisticRegression, DecisionTreeClassifier, MultinomialNB,
    RandomForestClassifier, SVC]
    for classifier in clfs:
        clf = classifier()
        print("Training {}".format(classifier))
        pred = cross_val_predict(clf, X, y)
        score = classification_report(y, pred)
        print(score)
        f.write('{}\n'.format(clf))
        f.write(score)
        f.write('\n\n')

    f.close()




def get_positive_preds(y_pred, pos='any'):
    """
    Gets the indices for any data points that were predicted to have a relation.
    """
    idxs = []
    for i, y_ in enumerate(y_pred):
        if y_ == pos:
            idxs.append(i)
    return idxs

def train_eval_cross_val(X, y):
    """
    Trains and validates a model using cross-validation.
    """
    clf = RandomForestClassifier(max_depth = None,
                            max_features = None,
                            min_samples_leaf = 2,
                            min_samples_split = 2,
                            n_estimators = 10,
                            n_jobs = 3)

    pred = cross_val_predict(clf, X, y)
    score = classification_report(y, pred)
    print(score)
    with open('rf_baseline_binary_model_cross_val_scores.txt', 'w') as f:
        f.write(score)
    return clf


def train_model(X, y):
    """
    Trains a model using the entire training data.
    Saves the trained model.
    No evaluation.
    """
    clf = RandomForestClassifier(max_depth = None,
                            max_features = None,
                            min_samples_leaf = 2,
                            min_samples_split = 2,
                            n_estimators = 10,
                            n_jobs = 3)
    clf.fit(X, y)
    outpath = os.path.join(MODELDIR, 'rf_binary_model.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump(clf, f)
    print("Saved at {}".format(outpath))
    return clf


def main():
    inpath = os.path.join(DATADIR, data_file)
    with open(inpath, 'rb') as f:
        feat_dicts, relats, X, y, = pickle.load(f)

    shuffle(feat_dicts, relats, X ,y)
    #X = X[:10, :]
    #y = y[:10]
    print("X: {}".format(X.shape))
    print("y: {}".format(len(y)))
    print(set(y))
    y = np.array(y)
    print(y[:10])


    train_eval_cross_val(X, y)

    train_model(X, y)

    exit()





if __name__ == '__main__':
    data_file = 'binary_data.pkl'
    main()
