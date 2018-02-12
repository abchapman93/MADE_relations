import sklearn
import pickle
import os, sys

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict

DATADIR = os.path.join('..', '..', 'data')
assert os.path.exists(DATADIR)

label_mapping = {'none': 0,
                'do':1,
                'reason': 2,
                'fr': 3,
                'severity_type': 4,
                'manner/route': 5,
                'adverse': 6,
                'du': 7
                }

def main():
    inpath = os.path.join(DATADIR, data_file)
    with open(inpath, 'rb') as f:
        X, y = pickle.load(f)
    print("X: {}".format(X.shape))
    print("y: {}".format(len(y)))

    X = X
    y = y

    # Transform y
    if binary == 'f': # all the classes
        y = [label_mapping[label] for label in y]
    elif binary == 'b': # yes or no
        y = [int(0) if label == 'none' else int(1)  for label in y]
    y = np.array(y)
    print(y)

    #clf = LinearRegression()
    clf = SVC()
    pred = cross_val_predict(clf, X, y)
    score = classification_report(y, pred)
    print(score)
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
    try:
        binary = sys.argv[2]
    except IndexError:
        binary = 'f'
    main()
