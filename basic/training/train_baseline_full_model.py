import sklearn
import pickle
import os, sys

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict

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

def transform_features(X):
    with open(os.path.join(MODELDIR, 'full_chi2.pkl'), 'rb') as f:
        chi2 = pickle.load(f)
    X = chi2.transform(X)
    print("New X shape: {}".format(X.shape))
    return X

def filter_by_idx(idxs, *args):
    arrs = args
    z = list(zip(*arrs))
    z = [z[i] for i in idxs]
    return zip(*z)

def main():
    inpath = os.path.join(DATADIR, data_file)
    with open(inpath, 'rb') as f:
        feat_dicts, relats, X, y, _, _ = pickle.load(f)
    del _
    #X = X[:10000, :]
    #y = y[:10000]
    print("X: {}".format(X.shape))
    print("y: {}".format(len(y)))
    print(len(feat_dicts))
    print(len(relats))
    # Filter out any examples that the filtering classifier said had no relation
    with open('bin_yes_preds.pkl', 'rb') as f:
        idxs = pickle.load(f)
    #X = X[idxs]
    #feat_dicts, relats, y = filter_by_idx(idxs, feat_dicts, relats, y)
    print("X: {}".format(X.shape))
    print("y: {}".format(len(y)))
    print(len(feat_dicts))
    print(len(relats))
    y = np.array(y)
    print(y[:10])

    #clf = LinearRegression()
    #clf = SVC()
    clf = RandomForestClassifier(max_depth = None,
                            max_features = None,
                            min_samples_leaf = 2,
                            min_samples_split = 2,
                            n_estimators = 10,
                            n_jobs = 3)
    pred = cross_val_predict(clf, X, y)
    score = classification_report(y, pred)
    print(score)

    # Save some examples of errors
    with open(os.path.join(DATADIR, 'annotated_documents.pkl'), 'rb') as f:
        docs = pickle.load(f)
    train_utils.save_errors('binary_errors.txt', y, pred, feat_dicts, relats, docs)

    # Save the model
    # TODO: Do you have to do something special because of cross-validation?
    model_file = os.path.join(MODELDIR, 'full_baseline.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
    print("Saved non-binary classifier at {}".format(model_file))
    exit()

    # Save the predictions
    outpath = os.path.join(DATADIR, 'filtered_data_lexical.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump((X_filter, y_filter), f)
    print("Saved filtered data at {}".format(outpath))
    print("{}, {}".format(X_filter.shape, y_filter.shape))
    exit()

    print("Training")
    clf.fit(X, y)
    pred = clf.predict(X)
    pred = [int(p) for p in pred]
    print(pred)
    score = classification_report(y, pred)
    print(score)





if __name__ == '__main__':
    data_file = 'full_lexical_data.pkl'
    main()
