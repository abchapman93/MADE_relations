from collections import defaultdict

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
        pass

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

        for i, (feature_name, feature_values) in enumerate(feature_dict.items()):
            if isinstance(feature_values, list):
                for feature_value in feature_values:
                    feature_idxs[feature_name][feature_value] = curr_idx
                    curr_idx += 1
                    # If this is the last value for this feature type, set the range
                    if i == len(feature_values):
                        feature_end_idx = curr_idx
                        feature_index_ranges[feature_name] = (feature_start_idx, feature_end_idx)

            elif isinstance(feature_values, int): # This is just a binary feature
                feature_idxs[feature_name][0] = curr_idx
                curr_idx += 1
                feature_end_idx = curr_idx
                feature_index_ranges[feature_name] = (feature_start_idx, feature_end_idx)
                # Now reset the start index
                feature_start_idx = curr_idx
            else:
                raise TypeError("Something's wrong: name: {} type:{}".format(feature_name, type(feature_values)))

        feature_index_ranges = {}
        for feature_name in feature_idxs.keys():
            idxs = [i for (v, i) in feature_idxs[feature_name].items()]
            feature_index_ranges[feature_name] = (min(idxs), max(idxs))

        print(feature_idxs.keys())
        print(feature_index_ranges)
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
