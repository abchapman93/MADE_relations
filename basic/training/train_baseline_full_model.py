import sklearn
import pickle
import os, sys

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.utils import shuffle

import train_utils


DATADIR = os.path.join('..', '..', 'data')
MODELDIR = os.path.join('..', '..', 'saved_models')
assert os.path.exists(DATADIR)
assert os.path.exists(MODELDIR)
sys.path.append('..')
sys.path.append('../../feature_extraction')

LABEL_MAPPING = {'none': 0,
                'do':1,
                'reason': 2,
                'fr': 3,
                'severity_type': 4,
                'manner/route': 5,
                'adverse': 6,
                'du': 7
                }


def filter_by_idx(idxs, *args):
    arrs = args
    z = list(zip(*arrs))
    z = [z[i] for i in idxs]
    return zip(*z)

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
    with open('rf_full_baseline_model_cross_val_scores.txt', 'w') as f:
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
    outpath = os.path.join(MODELDIR, 'rf_full_baseline_model.pkl')

    with open(outpath, 'wb') as f:
        pickle.dump(clf, f)
    print("Saved full classifier at {}".format(outpath))
    return clf


def main():
    inpath = os.path.join(DATADIR, data_file)
    with open(inpath, 'rb') as f:
        feat_dicts, relats, X, y, vectorizer, full_feature_selector = pickle.load(f)
        #feat_dicts, relats, X, y, _, _ = pickle.load(f) #TODO: get rid of the _'s
    shuffle(feat_dicts, relats, X ,y)
    #X = X[:10000, :]
    #y = y[:10000]
    print("X: {}".format(X.shape))
    print("y: {}".format(len(y)))
    print(len(feat_dicts))
    print(len(relats))
    # Filter out any examples that the filtering classifier said had no relation
    #with open('bin_yes_preds.pkl', 'rb') as f:
    #    idxs = pickle.load(f)
    #X = X[idxs]
    #feat_dicts, relats, y = filter_by_idx(idxs, feat_dicts, relats, y)
    print("X: {}".format(X.shape))
    print("y: {}".format(len(y)))
    print(len(feat_dicts))
    print(len(relats))
    y = np.array(y)
    print(y[:10])

    train_eval_cross_val(X, y)
    train_model(X, y)
    exit()





if __name__ == '__main__':
    data_file = 'baseline_full_data.pkl'
    main()
