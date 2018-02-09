"""
This script generates lexical features including:
    - n-grams before and after each annotation in the relation
    - pos features
    - number of entities between the two relation entities
"""
import os, sys
from collections import defaultdict
import pickle
from nltk import ngrams as nltk_ngrams

sys.path.append(os.path.join('..', 'basic'))
import base_feature


DATADIR = os.path.join('..', 'data') # the processed data
MADE_DIR = os.path.join(os.path.expanduser('~'), 'Box Sync', 'NLP_Challenge', 'MADE-1.0') # the original MADE data

class LexicalFeatureExtractor(base_feature.BaseFeatureExtractor):
    """
    ngram_window - the length of ngrams to include in the vocabulary.
    context_window - the number of ngrams to include before and after the entity.
    """
    def __init__(self, ngram_window=(1, 1), context_window=(2, 2), vocab=None):
        self.ngram_window = ngram_window
        if min(ngram_window) < 1 or max(ngram_window) > 3:
            raise NotImplementedError("Ngram Window must be between one and 3")
        self.context_window = context_window
        self._unfiltered_vocab = vocab # Contains unigrams-trigrams
        self.vocab = self.create_vocab_idx(vocab) # Only contains ngrams defined by context_window
        self.tokens = [gram for (gram, idx) in self.vocab.items() if len(gram.split()) == 1] # Only unigrams
        self.pos = {} # Will eventually contain mapping for POS tags

        self.all_features_values = self.create_base_features()


    def create_vocab_idx(self, vocab):
        """
        Creates a mapping from integers to the terms in vocab.
        Only includes the ngrams whose length are equal to or less than ngram_window.
        """
        vocab_idx = {}
        min_length, max_length = self.ngram_window
        for i, gram in enumerate(vocab.keys()):

            if (min(self.ngram_window) > len(gram.split()) or
                        len(gram.split()) > max(self.ngram_window)):
                continue
            vocab_idx[gram] = i
        return vocab_idx

    def create_base_features(self):
        """
        Enumerates possible feature values from the vocab, as well as an OOV value.
        Any features that are binary should only get one index and are encoded as 0.
        """
        # This will be a dictionary that contains all possible values for each feature
        all_features_values = {
            'same_sentence': 0,
            'num_tokens_between': 0,
            'grams_between': ['OOV'] + list(self.vocab),
            'grams_before': ['OOV'] + list(self.vocab),
            'grams_after': ['OOV'] + list(self.vocab),
            'pos_grams_between': ['OOV'],
            'pos_grams_before': ['OOV'],
            'pos_grams_after': ['OOV']

            }
        return all_features_values
    def create_feature_dict(self, relat, doc):
        """
        Takes a RelationAnnotation and an AnnotatedDocument.
        Returns the a dictionary containing the defined lexical features.
        """

        lex_features = {}

        # Binary feature: Are they in the same sentence?
        in_same_sentence = doc.in_same_sentence(relat.get_span())

        # Get the number of tokens between
        num_tokens_between = len(self.get_grams_between(relat, doc, (1, 1)))
        # Get all tokens in between
        grams_between = self.get_grams_between(relat, doc, self.ngram_window)
        grams_before = self.get_grams_before(relat, doc)
        grams_after = self.get_grams_after(relat, doc)


        lex_features['same_sentence'] = in_same_sentence
        lex_features['num_tokens_between'] = num_tokens_between
        lex_features['grams_between'] = grams_between
        lex_features['grams_before'] = grams_before
        lex_features['grams_after'] = grams_after

        return lex_features







    def get_grams_between(self, relat, doc, ngram_window):
        """
        Returns the N-grams between the two entities connected in relat
        """
        grams = []
        spans = relat.spans
        start = spans[0][1]
        end = spans[1][0]
        start, end = sorted((start, end))
        tokens_in_span = doc.get_text_at_span((start, end))
        for n in set(ngram_window):
            # Now sort the ngrams so that it doesn't matter what order they occur in
            grams = grams + [' '.join(sorted(tup)) for tup in list(nltk_ngrams(tokens_in_span, n))]
        return grams


    def get_grams_before(self, relat,doc):
        """
        Returns the n-grams before the first entity.
        """
        grams = []
        offset = relat.span[0]
        tokens_before = doc.get_tokens_before_or_after(offset, delta=-1, n=self.context_window[0])
        for n in set(self.ngram_window):
            grams = grams + [' '.join(sorted(tup)) for tup in list(nltk_ngrams(tokens_before, n))]
        return grams

    def get_grams_after(self, relat, doc):
        """
        Returns the n-grams after the final entity.
        """
        grams = []
        offset = relat.span[1]
        tokens_after = doc.get_tokens_before_or_after(offset, delta=1, n=self.context_window[1])
        for n in set(self.ngram_window):
            grams = grams + [' '.join(sorted(tup)) for tup in list(nltk_ngrams(tokens_after, n))]
        return grams


    def get_tokens_in_window(self, relat, doc):
        """
        """
        print(relat, doc)
        # Get the text between the two entities
        start, end = relat.get_span()
        print(doc.in_same_sentence((start, end)))
        exit()

    def get_pos_tags(self):
        pass



    def __repr__(self):
        return "LexicalFeatureExtractor Ngram Window: {} Vocab: {} terms".format(
                self.ngram_window, len(self.vocab))


def main():
    # Load in data
    #print(list(nltk_ngrams('This is a sentence'.split(), 2))); exit()
    with open(os.path.join(DATADIR, 'generated_train.pkl'), 'rb') as f:
        relats = pickle.load(f)
    with open(os.path.join(DATADIR, 'annotated_documents.pkl'), 'rb') as f:
        docs = pickle.load(f)
    print("Loaded {} relations and {} docs".format(len(relats), len(docs)))
    with open(os.path.join(DATADIR, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    feature_extractor = LexicalFeatureExtractor(ngram_window=(1, 2), vocab=vocab)



    feat_dicts = {} # mapping from relation to feature dictionary
    # Create features for each relation



    vector_creator = base_feature.FeatureVectorCreator()
    feature_vector_idxs = vector_creator.create_feature_vector_indices([feature_extractor])

    print(feature_extractor)
    for i, relat in enumerate(relats[1:]):

        #if relat.span[1] - relat.span[0] > 500:
        #    print(i)
        #    break
        #continue
        doc = docs[relat.file_name]
        span = relat.span
        feature_dict = feature_extractor.create_feature_dict(relat, doc)
        feat_dicts[relat] = feature_dict
        for feature_name, feature_values in feature_dict.items():
            if isinstance(feature_values, list):
                all_lex_features[feature_name].extend(feature_values)

        break

    # Now that we have all possible values for each feature,
    # we can assign each feature-value pair an index
    vector_creator = base_feature.FeatureVectorCreator()
    feature_vector = vector_creator.create_feature_vector_indices(all_lex_features)




if __name__ == '__main__':
    main()
