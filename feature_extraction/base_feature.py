"""
This module contains classes that
    1) Take feature dictionaries and vectorize them
    2) Use Chi2 to select K best and can transform X arrays or
    write the feature names and scordes
"""
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer

import os
import numpy as np
import pickle

from sklearn.feature_selection import SelectKBest, chi2

DATADIR = os.path.join('..', '..', 'data')
MODELDIR = os.path.join('..', '..', 'saved_models')

class MyFeatureSelector(object):
    def __init__(self, vectorizer, k=1000):
        self.ch2 = SelectKBest(chi2, k)
        self.k = k
        self.vectorizer = vectorizer

    def fit(self, X, y):
        """
        Fits to a feature matrix.
        """
        self.ch2.fit(X, y)

    def transform(self, X):
        """
        Transforms a feature matrix
        """
        return self.ch2.transform(X)

    def fit_transform(self, X, y):
        return self.ch2.fit_transform(X, y)

    def write_feature_names(self, outpath):
        with open(outpath, 'w') as f:
            f.write('\n'.join(self.vectorizer.get_feature_names()))

    def write_feature_scores(self, outpath):
        feature_names = self.vectorizer.get_feature_names()
        # (index, score)
        top_ranked = [(index, score) for (index, score)
                        in enumerate(self.ch2.scores_)]

        # Sort by score
        top_ranked = list(sorted(top_ranked, key=lambda x:x[1], reverse=True))

        # Use the original index to find the feature name
        feature_names_and_scores = ['{}\t{}'.format(feature_names[idx], score)
                    for (idx, score) in top_ranked]

        f = open(outpath, 'w')
        f.write('\n'.join(feature_names_and_scores))
        print("Saved scores at {}".format(outpath))
        f.close()

class BaseFeatureExtractor(object):
    def __init__(self):
        pass

    def extract_features(self, relation, doc):
        return np.empty(0)
