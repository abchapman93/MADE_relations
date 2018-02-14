"""
This script generates lexical features including:
    - n-grams before and after each annotation in the relation
    - pos features
    - number of entities between the two relation entities
"""
import os, sys
from collections import defaultdict
import re
from random import shuffle
import pickle
from nltk import ngrams as nltk_ngrams

from sklearn.feature_extraction import DictVectorizer

sys.path.append(os.path.join('..', 'basic'))
import base_feature
from feature_utils import save_example_feature_dict

import create_vocab

DATADIR = os.path.join('..', 'data') # the processed data
MODELDIR = os.path.join('..', 'saved_models')
MADE_DIR = os.path.join(os.path.expanduser('~'), 'Box Sync', 'NLP_Challenge', 'MADE-1.0') # the original MADE data


ENTITY_TYPES_MAPPING = {
    'DRUG': 0, 'DOSE': 1,
    'INDICATION': 2, 'FREQUENCY': 3,
    'SSLIF': 4, 'ADE': 5,
    'ROUTE': 6, 'DURATION': 7,
    'SEVERITY': 8
}


class LexicalFeatureExtractor(base_feature.BaseFeatureExtractor):
    """
    ngram_window - the length of ngrams to include in the vocabulary.
    context_window - the number of ngrams to include before and after the entity.
    """
    def __init__(self, ngram_window=(1, 1), context_window=(2, 2),
                vocab=None, pos_vocab=None, min_vocab_count=5, min_pos_count=5):
        self.ngram_window = ngram_window
        if min(ngram_window) < 1 or max(ngram_window) > 3:
            raise NotImplementedError("Ngram Window must be between one and 3")
        self.context_window = context_window
        self.min_vocab_count = min_vocab_count
        self.min_pos_count = min_pos_count

        # Set vocab and POS vocab
        self._unfiltered_vocab = vocab # Contains unigrams-trigrams, no count threshold
        self._unfiltered_pos_vocab = pos_vocab

        self.vocab = self.create_vocab(vocab, min_vocab_count) # Only contains ngrams defined by context_window
        #print(self.vocab); exit()
        self.pos_vocab =  self.create_vocab(pos_vocab, min_pos_count)
        #self.tokens = [gram for (gram, idx) in self.vocab.items() if len(gram.split()) == 1] # Only unigrams
        self.pos = {} # Will eventually contain mapping for POS tags

        #self.all_features_values = self.create_base_features()


    def create_vocab(self, vocab, thresh=5):
        """
        Creates a mapping from integers to the terms in vocab.
        Only includes the ngrams whose length are equal to or less than ngram_window.
        """
        # Preprocess to transform numbers, etc.
        #vocab = {self.normalize_grams(gram): vocab[gram] for gram in vocab}
        vocab_idx = {}
        min_length, max_length = self.ngram_window
        for i, gram in enumerate(vocab.keys()):
            # Normalize the gram
            #gram = self.normalize_grams(gram)
            # If it doesn't meet the count threshold, skip it
            if vocab[gram] < thresh:
                continue


            # If it's too big or too small, don't include it
            if (min(self.ngram_window) > len(gram.split()) or
                        len(gram.split()) > max(self.ngram_window)):
                continue
            vocab_idx[gram] = vocab[gram]
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
            'pos_grams_between': ['OOV'] + list(self.pos_vocab),
            #'pos_grams_before': ['OOV'] + list(self.pos_vocab),
            #'pos_grams_after': ['OOV'] + list(self.pos_vocab),
            'first_entity_type': 0,#list(ENTITY_TYPES_MAPPING.values()),
            'second_entity_type': 0,#list(ENTITY_TYPES_MAPPING.values()),

            }
        return all_features_values

    def create_feature_dict(self, relat, doc):
        """
        Takes a RelationAnnotation and an AnnotatedDocument.
        Returns the a dictionary containing the defined lexical features.
        """

        lex_features = {}

        # Binary feature: Are they in the same sentence?
        lex_features['same_sentence'] = doc.in_same_sentence(relat.get_span())
        # Get the number of tokens between
        # NOTE: only unigrams
        lex_features['num_tokens_between'] = len(self.get_grams_between(relat, doc, ngram_window=(1, 1)))
        # Get all tokens in between
        lex_features.update({
            'grams_between:<{}>'.format(v): 1 for v in self.get_grams_between(relat, doc)
            })
        lex_features.update({
            'grams_before:<{}>'.format(v): 1 for v in self.get_grams_before(relat, doc)
            })
        lex_features.update({
            'grams_after:<{}>'.format(v): 1 for v in self.get_grams_after(relat, doc)
            })
        print(lex_features); exit()
        grams_after = self.get_grams_after(relat, doc)

        tags_between = self.get_grams_between(relat, doc, seq='tags')
        num_sentences_overlap = len(doc.get_sentences_overlap_span(relat.get_span()))

        entities_between = self.get_entities_between(relat, doc)
        num_entities_between = len(entities_between)
        types_entities_between = set([e.type for e in entities_between])



        entity_type1 = relat.entity_types[0].upper()
        entity_type2 = relat.entity_types[1].upper()

        lex_features['same_sentence'] = in_same_sentence
        lex_features['num_tokens_between'] = num_tokens_between

        # Transform lists into binary key-value pairs
        for list_of_values in (grams_between, grams_before, grams_after,
            tags_between, entity_type, types_entities_between):
            pass
        lex_features['grams_between'] = grams_between
        lex_features['grams_before'] = grams_before
        lex_features['grams_after'] = grams_after

        lex_features['pos_grams_between'] = tags_between

        lex_features['first_entity_type'] = entity_type1#ENTITY_TYPES_MAPPING[entity_type1]
        lex_features['second_entity_type'] = entity_type2#ENTITY_TYPES_MAPPING[entity_type2]
        lex_features['num_sentences_overlap'] = num_sentences_overlap

        lex_features['num_entities_between'] = num_entities_between
        lex_features['types_entities_between'] = types_entities_between
        #assert len(set(lex_features.keys()).difference(set(self.all_features_values.keys()))) == 0
        #assert len(set(self.all_features_values.keys()).difference(set(lex_features.keys()))) == 0



        return lex_features


    def get_grams_between(self, relat, doc, seq='tokens', ngram_window=None):
        """
        Returns the N-grams between the two entities connected in relat.
        Represents it as OOV if it's not in the vocabulary.
        Returns a unique set.
        """

        if seq == 'tokens':
            vocab = self.vocab
        elif seq == 'tags':
            vocab = self.pos_vocab
        else:
            raise ValueError("Must specify seq: {}".format(seq))

        if not ngram_window:
            ngram_window = self.ngram_window

        all_grams = []
        spans = relat.spans
        start = spans[0][1]
        end = spans[1][0]
        start, end = sorted((start, end))
        tokens_in_span = doc.get_tokens_or_tags_at_span((start, end), seq)
        # NOTE: lower-casing the ngrams, come back to this if you want to encode the casing
        tokens_in_span = [token.lower() for token in tokens_in_span]
        for n in range(ngram_window[0], ngram_window[1] + 1):
            # Now sort the ngrams so that it doesn't matter what order they occur in
            grams = list(nltk_ngrams(tokens_in_span, n))
            grams = self.sort_ngrams(grams)# + [' '.join(sorted(tup)) for tup in list(nltk_ngrams(tokens_in_span, n))]
            all_grams.extend(set(grams))
        all_grams = [self.normalize_grams(x) for x in set(all_grams)]
        all_grams = [x if x in vocab else 'OOV' for x in all_grams]
        return set(all_grams)


    def get_grams_before(self, relat,doc, seq='tokens', ngram_window=None):
        """
        Returns the n-grams before the first entity.
        """
        if seq == 'tokens':
            vocab = self.vocab
        elif seq == 'tags':
            vocab = self.pos_vocab
        if not ngram_window:
            ngram_window = self.ngram_window

        all_grams = []
        offset = relat.span[0]
        tokens_before = doc.get_tokens_or_tags_before_or_after(offset, delta=-1,
            n=self.context_window[0], seq=seq, padding=True)
        tokens_before = [token.lower() for token in tokens_before]
        for n in range(ngram_window[0], ngram_window[1] + 1):
            grams = list(nltk_ngrams(tokens_before, n))
            grams = self.sort_ngrams(grams)# + [' '.join(sorted(tup)) for tup in list(nltk_ngrams(tokens_in_span, n))]
            all_grams.extend(set(grams))
            #grams = grams + [' '.join(sorted(tup)) for tup in list(nltk_ngrams(tokens_before, n))]
        all_grams = [self.normalize_grams(x) for x in set(all_grams)]
        all_grams = [x if x in vocab else 'OOV' for x in all_grams]
        return set(all_grams)

    def get_grams_after(self, relat, doc, seq='tokens', ngram_window=None):
        """
        Returns the n-grams after the final entity.
        """
        if seq == 'tokens':
            vocab = self.vocab
        elif seq == 'tags':
            vocab = self.pos_vocab
        if not ngram_window:
            ngram_window = self.ngram_window

        all_grams = []
        offset = relat.span[1]
        tokens_after = doc.get_tokens_or_tags_before_or_after(offset, delta=1,
                                        n=self.context_window[1], seq=seq)
        tokens_after = [token.lower() for token in tokens_after]
        for n in range(ngram_window[0], ngram_window[1] + 1):
            grams = list(nltk_ngrams(tokens_after, n))
            grams = self.sort_ngrams(grams)# + [' '.join(sorted(tup)) for tup in list(nltk_ngrams(tokens_in_span, n))]
            all_grams.extend(set(grams))
            #grams = grams + [' '.join(sorted(tup)) for tup in list(nltk_ngrams(tokens_after, n))]
        all_grams = [self.normalize_grams(x) for x in set(all_grams)]
        all_grams = [x if x in vocab else 'OOV' for x in all_grams]
        return set(all_grams)

    def sort_ngrams(self, ngrams):
        return [' '.join(sorted(tup)) for tup in ngrams]

    def normalize_grams(self, ngram_string):
        """
        Normalizes the values in a string of joined ngrams
        """
        # Substitute numbers
        return create_vocab.normalize_grams(ngram_string)

    def get_pos_tags(self):
        pass

    def get_entities_between(self, relat, doc):
        """
        Returns a list of entities that occur between entity1 and entity2
        """
        offset, end = relat.get_span()
        overlapping_entities = []
        # Index the entity in doc by span
        offset_to_entity = {entity.span[0]: entity for entity in doc.get_annotations()
                    if entity.id not in (
                        relat.annotation_1.id, relat.annotation_2.id)
                        }

        while offset < end:
            if offset in offset_to_entity:
                overlapping_entities.append(offset_to_entity[offset])
            offset += 1

        return overlapping_entities




        span = relat.get_span

    def __repr__(self):
        return "LexicalFeatureExtractor Ngram Window: {} Vocab: {} terms".format(
                self.ngram_window, len(self.vocab))


def main():
    # Load in data
    inpath = os.path.join(DATADIR, 'training_documents_and_relations.pkl')
    with open(inpath, 'rb') as f:
        docs_, relats = pickle.load(f)

    docs = {doc.file_name: doc for doc in docs_}
    print("Loaded {} docs and {} relations".format(len(docs), len(relats)))
    shuffle(relats)
    with open(os.path.join(DATADIR, 'vocab.pkl'), 'rb') as f:
        vocab, pos_vocab = pickle.load(f)




    feature_extractor = LexicalFeatureExtractor(context_window=(2, 2),
                            ngram_window=(1, 3), vocab=vocab, pos_vocab=pos_vocab,
                            min_vocab_count=20, min_pos_count=20)
    vector_creator = base_feature.FeatureVectorCreator()

    feat_dicts = [] # mappings of feature names to values
    y = [] # the relation types
    for i, relat in enumerate(relats):
        doc = docs[relat.file_name]
        span = relat.span

        # This returns a dictionary with lists of values
        feature_dict = feature_extractor.create_feature_dict(relat, doc)
        # You now have to flatten it so that it's a single-dimension dictionary
        feature_vect_dict = vector_creator.transform_feature_vector(feature_dict)
        feat_dicts.append(feature_vect_dict)
        if i == 0:
            print(feature_dict.keys())
            print(feature_dict['first_entity_type'])
        if i % 10 == 0:
            pass
            #break
        #   print(feature_dict)
#
        #   save_example_feature_dict(feature_dict, relat, doc)
           #exit()
           #break
        if i % 100 == 0 and i > 0:
            print("{}/{}".format(i, len(relats)))

        y.append(relat.type)


    X = vector_creator.fit_transform(feat_dicts)
    #print(X)
    print(X.shape)
    print(len(y))

    # Let's save the names of the feature names
    feature_names = vector_creator.vectorizer.get_feature_names()
    print(type(feature_names))
    with open('lexical_feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))

    # Now pickle the data
    outpath = os.path.join(DATADIR, 'data_lexical.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump((X, y), f)

    print("Saved data at {}".format(outpath))
    print()

    # Finally, save the vectorizer
    with open(os.path.join(MODELDIR, 'lex_dict_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vector_creator.vectorizer, f)


    print("Made it to the end")



if __name__ == '__main__':
    main()
