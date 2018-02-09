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
        self.vectorizer = DictVectorizer(sparse=True)
        self.possible_values = set()
        pass

    def create_vectorizer(self, *args):
        d = {}
        feature_extractors = args
        for feature_extractor in feature_extractors:
            for feature_enum in self.enum_feature_values(feature_extractor):
                if feature_enum in d:
                    pass
                else:
                    d[feature_enum] = 0
        self.vectorizer.fit([d])

    def enum_feature_values(self, feature_extractor):
        L = []
        for feature_name, v in feature_extractor.all_features_values.items():
            if isinstance(v, int): # this is a single-column feature
                L.append(feature_name)
            elif isinstance(v, list): # This is a list of possible values
                for value in v:
                    L.append("{}:<{}>".format(feature_name, value))
            else:
                raise TypeError("Something's wrong: {}:<{}>".format(feature_name, v))
        # Set possible values
        self.possible_values = set(L)
        return L





    def create_feature_vector_indices(self, feature_extractors):
        """
        Converts tokens, pos tags, labels, etc. to indexes
        TODO: should take an optional number of feature extractors
        """
        curr_idx = 0
        #  this will be a mapping from feature type and value to an index
        feature_idxs = defaultdict(dict) # {feature_name: feature_value: index}

        feature_dict = {}
        for feature_extractor in feature_extractors:
            for feature_name, feature_values in feature_extractor.all_features_values.items():
                feature_dict[feature_name] = feature_values

        # This will be a mapping from names to ranges
        feature_index_ranges = {}

        # This will keep track of the indices
        feature_start_idx = curr_idx

        feature_name_dict = {}

        for i, (feature_name, feature_values) in enumerate(feature_dict.items()):
            if isinstance(feature_values, list):
                for feature_value in feature_values:
                    feature_idxs[feature_name][feature_value] = curr_idx
                    curr_idx += 1
                    # If this is the last value for this feature type, set the range
                    if i == len(feature_values):
                        feature_end_idx = curr_idx
                        feature_index_ranges[feature_name] = (feature_start_idx, feature_end_idx)
                    feature_name_dict['{}:{}'.format(feature_name, feature_value)] = feature_value

            elif isinstance(feature_values, int): # This is just a binary feature
                feature_idxs[feature_name][0] = curr_idx
                curr_idx += 1
                feature_end_idx = curr_idx
                feature_index_ranges[feature_name] = (feature_start_idx, feature_end_idx)
                # Now reset the start index
                feature_start_idx = curr_idx
                feature_name_dict['{}'.format(feature_name, feature_values)] = feature_values
            else:
                raise TypeError("Something's wrong: name: {} type:{}".format(feature_name, type(feature_values)))

        print(feature_name_dict); exit()

        feature_index_ranges = {}
        for feature_name in feature_idxs.keys():
            idxs = [i for (v, i) in feature_idxs[feature_name].items()]
            feature_index_ranges[feature_name] = (min(idxs), max(idxs))

        print(feature_idxs.keys())
        print(feature_index_ranges)

        self.feature_idxs = feature_idxs
        self.feature_index_ranges = feature_index_ranges
        return

        exit()


        print(feature_idxs)



        # First, unroll all possible features into index spaces
        # This should the be number of ngrams in the vocabulary
        # This defines the indices in feature_values for ngrams before the relation
        features_grams_before = {gram: idx + curr_idx for (gram, idx) in self.vocab.items()}
        for gram in feature_dict['grams_before']:
            feature_idx = features_grams_before[gram]
            feature_values[feature_idx] += 1
            print(gram)
        print(feature_values)
        print([x for x in feature_values.items() if x[1] > 0])
        exit()
        #features_grams_before = [curr_idx + x for x in range(len(self.vocab))]
        print(features_grams_before)
        exit()
        pass

    def transform_feature_vector(self, feature_dict):
        """
        Takes the dictionary of feature names and values for a single instance.
        Returns a sparse vector where the indices of the vector are valued by the values.
        """
        sparse_vector_dict = {} # Will start with dicts of indices and values
        for feature_name, v in feature_dict.items():
            print(feature_name)
            if isinstance(v, int):# Really just a single value
                #idx = self.feature_idxs[feature_name][0]

                # If this is an unknown feature, we don't want to add it
                if feature_name not in self.possible_values:
                    print(self.possible_values); exit()
                    continue
                sparse_vector_dict['{}'.format(feature_name)] = v
                continue
            # Otherwise it's a list of tokens we can iterate through
            for value in v:
                print(value)
                # Get the index
                feature_value_name = "{}:<{}>".format(feature_name, value)
                if feature_value_name not in self.possible_values: # Call it OOV
                    print(feature_value_name); exit()
                    feature_value_name = "{}:<OOV>".format(feature_name)
                sparse_vector_dict[feature_value_name] = 1
                #idx = self.feature_idxs[feature_name][value.lower()] # NOTE: Lower-casing to match in the vocab
                #sparse_vector_dict[idx] = 1# value.lower() # TODO: Change to 1
        return sparse_vector_dict


    def vectorize(self, d):
        """
        Takes a list of dictonairies. Returns a sparse vector
        """

        X = self.vectorizer.transform(d)
        return X

    #def transform_feature_vector(self, feature_dict):
    #    """
    #    Transforms a dictionary so that all keys are made up of key-value concatenations
    #    and the values are actual feature values.
    #    ie., {'in_same_sentence': 1, 'ngrams_before': ['hello', 'world']}
    #        => {'in_same_sentence': 1, 'ngrams_before:hello': 1, 'ngrams_before:world': 1}
    #    """
