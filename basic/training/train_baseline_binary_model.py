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

def train_models(X, y):
    f = open('binary_model_scores.txt', 'w')
    clfs = [LogisticRegression, DecisionTreeClassifier, MultinomialNB,
    RandomForestClassifier, SVC]
    for classifier in clfs:
        clf = classifier()
        print("Training {}".format(classifier))
        pred = cross_val_predict(clf, X, y)
        score = classification_report(y, pred)
        print(score)

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
    #train_models(X, y)
    clf = SVC()
    clf_name = 'SVC'
    param_grid = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
            ]
    learned_parameters = train_utils.grid_search(X, y, clf, param_grid)
    with open('best_{}_params.pkl'.format(clf_name), 'wb') as f:
        pickle.dump(parameters, f)
    exit()

    #clf = SVC(C = learned_parameters['C'], 'kernel')
    clf = RandomForestClassifier(max_depth = None,
                            max_features = None,
                            min_samples_leaf = 2,
                            min_samples_split = 2,
                            n_estimators = 10,
                            n_jobs = 3)

    pred = cross_val_predict(clf, X, y)
    score = classification_report(y, pred)
    print(score)
    exit()

    yes_preds = get_positive_preds(pred, pos='any')
    with open('bin_yes_preds.pkl', 'wb') as f:
        pickle.dump(yes_preds, f)
    print("Saved indices of predicted relations")

    # Save some examples of errors
    with open(os.path.join(DATADIR, 'annotated_documents.pkl'), 'rb') as f:
        docs = pickle.load(f)
    train_utils.save_errors('binary', y, pred, feat_dicts, relats, docs)

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
