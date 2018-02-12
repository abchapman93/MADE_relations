import sklearn
import pickle
import os, sys

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict

DATADIR = os.path.join('..', '..', 'data')
MODELDIR = os.path.join('..', '..', 'saved_models')
assert os.path.exists(DATADIR)
assert os.path.exists(MODELDIR)

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

def main():
    inpath = os.path.join(DATADIR, data_file)
    with open(inpath, 'rb') as f:
        X, y = pickle.load(f)
    #X = X[:10000, :]
    #y = y[:10000]
    print("X: {}".format(X.shape))
    print("y: {}".format(len(y)))

    #X = X
    #y = y


    # Transform y
    #y = [LABEL_MAPPING[label]  for label in y]
    y = np.array(y)
    print(y[:10])

    #clf = LinearRegression()
    clf = SVC()
    pred = cross_val_predict(clf, X, y)
    score = classification_report(y, pred)
    print(score)
    # Save the model
    # TODO: Do you have to do something special because of cross-validation?
    model_file = os.path.join(MODELDIR, 'filter_baseline.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
    print("Saved binary classifier at {}".format(model_file))
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
    data_file = sys.argv[1]
    main()
