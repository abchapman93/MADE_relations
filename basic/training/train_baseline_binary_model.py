import sklearn
import pickle
import os, sys

import numpy as np

from sklearn.linear_model import LinearRegression
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

label_mapping = {'none': 0,
                'do':1,
                'reason': 2,
                'fr': 3,
                'severity_type': 4,
                'manner/route': 5,
                'adverse': 6,
                'du': 7
                }

def get_positive_preds(y_pred, pos='any'):
    """
    Gets the indices for any data points that were predicted to have a relation.
    """
    idxs = []
    for i, y_ in enumerate(y_pred):
        if y_ == pos:
            idxs.append(i)
    return idxs

def main():
    inpath = os.path.join(DATADIR, data_file)
    with open(inpath, 'rb') as f:
        feat_dicts, relats, X, y, _, _ = pickle.load(f)
    del _
    #X = X[:10, :]
    #y = y[:10]
    print("X: {}".format(X.shape))
    print("y: {}".format(len(y)))
    print(set(y))

    #X = X
    #y = y


    # Transform y
    #y = [int(0) if label == 'none' else int(1)  for label in y_non_bin]
    y = np.array(y)
    print(y[:10])

    #X = transform_features(X)

    #clf = LinearRegression()
    clf = SVC()
    pred = cross_val_predict(clf, X, y)
    score = classification_report(y, pred)
    print(score)

    yes_preds = get_positive_preds(pred, pos='any')
    with open('bin_yes_preds.pkl', 'wb') as f:
        pickle.dump(yes_preds, f)
    print("Saved indices of predicted relations")

    # Save some examples of errors
    with open(os.path.join(DATADIR, 'annotated_documents.pkl'), 'rb') as f:
        docs = pickle.load(f)
    train_utils.save_errors('binary_errors.txt', y, pred, feat_dicts, relats, docs)

    # Save the model
    # TODO: Do you have to do something special because of cross-validation?
    model_file = os.path.join(MODELDIR, 'filter_baseline.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
    print("Saved binary classifier at {}".format(model_file))
    exit()

    print("Training")
    clf.fit(X, y)
    pred = clf.predict(X)
    pred = [int(p) for p in pred]
    print(pred)
    score = classification_report(y, pred)
    print(score)





if __name__ == '__main__':
    data_file = 'binary_lexical_data.pkl'
    main()
