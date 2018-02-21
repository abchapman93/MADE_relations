import sklearn
import pickle
import os, sys

import numpy as np

from scipy.sparse import hstack

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
    inpath1 = os.path.join(DATADIR, data_file1)
    with open(inpath1, 'rb') as f:
        feat_dicts, relats, X_lex, y_lex, _, _ = pickle.load(f)
        y_lex = np.array(y_lex)
    del _
    inpath2 = os.path.join(DATADIR, data_file2)
    print(inpath2)
    with open(inpath2, 'rb') as f:
        feat_dicts, relats, X_clus, y_clus, _, _ = pickle.load(f)
        y_clus = np.array(y_clus)
    print(feat_dicts[0])
    del _
    X = hstack([X_lex, X_clus])
    assert np.array_equal(y_lex, y_clus)
    y = y_lex
    print("X: {}".format(X.shape))
    print("y: {}".format(len(y)))
    print(set(y))
    # Transform y
    #y = [int(0) if label == 'none' else int(1)  for label in y_non_bin]
    y = np.array(y)
    print(y[:10])

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
    data_file1 = 'full_lexical_data.pkl'
    data_file2 = 'full_cluster_data.pkl'
    main()
