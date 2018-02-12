from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer

class BaseFeatureExtractor(object):
    def __init__(self):
        pass

    def extract_features(self, relation, doc):
        return np.empty(0)

class FeatureVectorCreator(object):
    """
    Takes a dictionary of features from 1 or more FeatureExtractors
    and returns a sparse vector
    """

    def __init__(self, *args):
        self.vectorizer = DictVectorizer(sparse=True, sort=True)
        self.possible_values = set()
        pass


    def transform_feature_vector(self, feature_dict):
        """
        Takes the dictionary of feature names and values for a single instance.
        Returns a sparse vector where the indices of the vector are valued by the values.
        """
        sparse_vector_dict = {} # Will start with dicts of indices and values
        for feature_name, v in feature_dict.items():
            if isinstance(v, int):# Really just a single value
                sparse_vector_dict['{}'.format(feature_name)] = v
                continue

            # Or it's a string value, such as <first entity type>:<DRUG>
            elif isinstance(v, str):
                feature_value_name = "{}:<{}>".format(feature_name, v)
                sparse_vector_dict[feature_value_name] = 1

            # Otherwise it's a list of tokens we can iterate through
            else:
                for value in v:
                    feature_value_name = "{}:<{}>".format(feature_name, value)
                    sparse_vector_dict[feature_value_name] = 1
        return sparse_vector_dict


    def fit_transform(self, d):
        X = self.vectorizer.fit_transform(d)
        return X


    def vectorize(self, d):
        """
        Takes a list of dictonairies. Returns a sparse vector
        """
        try:
            X = self.vectorizer.transform(d)
        except AttributeError as e:
            raise AttributeError("Vectorizer has not been fit yet. Call fit_transform(d)")
        return X
