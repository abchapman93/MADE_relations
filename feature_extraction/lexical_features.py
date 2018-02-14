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


def normalize_grams(ngram_string):
    """
    Normalizes the values in a string of joined ngrams
    """
    # Substitute numbers
    ngram_string = re.sub('[\d]+|one|two|three|four|five|six|seven|eight|nine|ten', '<NUMBER>', ngram_string)
    return ngram_string



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

        # Get the number of tokens between
        # NOTE: only unigrams
        lex_features['num_tokens_between'] = len(self.get_grams_between(relat, doc, ngram_window=(1, 1)))
        # Get all tokens/POS tags in between
        # Create one feature for each ngram/tag
        lex_features.update({
            'grams_between:<{}>'.format(v): 1 for v in self.get_grams_between(relat, doc)
            })
        lex_features.update({
            'grams_before:<{}>'.format(v): 1 for v in self.get_grams_before(relat, doc)
            })
        lex_features.update({
            'grams_after:<{}>'.format(v): 1 for v in self.get_grams_after(relat, doc)
            })

        lex_features.update({
            'tags_between:<{}>'.format(v): 1 for v in self.get_grams_between(relat, doc, seq='tags')
            })
        lex_features.update({
            'tags_before:<{}>'.format(v): 1 for v in self.get_grams_before(relat, doc, seq='tags')
            })
        lex_features.update({
            'tags_after:<{}>'.format(v): 1 for v in self.get_grams_after(relat, doc, seq='tags')
            })

        # Get features for information about entities/context between
        # Binary feature: Are they in the same sentence?
        lex_features['same_sentence'] = doc.in_same_sentence(relat.get_span())

        # One binary feature for every type of entity between
        entities_between = self.get_entities_between(relat, doc)
        # TODO: Maybe change this to a count
        lex_features.update({
            'entities_between:<{}>'.format(v.type.upper()): 1 for v in entities_between
            })
        lex_features['num_entities_between'] = len(entities_between)

        lex_features['num_sentences_overlap'] = len(doc.get_sentences_overlap_span(relat.get_span()))

        # Features for types of the entities
        lex_features['first_entity_type:<{}>'.format(relat.entity_types[0].upper())] = 1
        lex_features['second_entity_type:<{}>'.format(relat.entity_types[1].upper())] = 1
        lex_features['entity_types_concat'] = '<=>'.join(sorted(['<{}>'.format(et.upper()) for et in
            (relat.entity_types[0].upper(), relat.entity_types[1].upper())
        ]))


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
        span1, span2 = relat.spans
        # Fixed this: get the start and span of the middle, not of the entire relation
        _, start, end, _ = sorted(span1 +span2)
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
        return normalize_grams(ngram_string)

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
    #shuffle(relats)
    with open(os.path.join(DATADIR, 'vocab.pkl'), 'rb') as f:
        vocab, pos_vocab = pickle.load(f)




    feature_extractor = LexicalFeatureExtractor(context_window=(2, 2),
                            ngram_window=(1, 3), vocab=vocab, pos_vocab=pos_vocab,
                            min_vocab_count=20, min_pos_count=20)

    feat_dicts = [] # mappings of feature names to values
    y = [] # the relation types
    #for i, relat in enumerate(relats[6972:]):
    for i, relat in enumerate(relats):
        #if relat.file_name != '1_1044':
        #    continue
        #else:
        #    if str(relat) == "'Percocet':'3', Drug:Dose, type=do":
        #        print("Found it!")
        #        print(i)
        #        print(relat)
        #        exit()
        #    continue

        doc = docs[relat.file_name]
        span = relat.span

        feature_dict = feature_extractor.create_feature_dict(relat, doc)
        feat_dicts.append(feature_dict)
        y.append(relat.type)
        #if i % 10 == 0:
        #    if i != 0:
        #        break
           #save_example_feature_dict(feature_dict, relat, doc)
           #exit()
        if i % 100 == 0 and i > 0:
            print("{}/{}".format(i, len(relats)))
            #break


    print(feat_dicts[0])
    #for d in feat_dicts:
    #    for v in d.values():
    #        assert isinstance(v, int)
    # Create feature vectors
    vectorizer = DictVectorizer(sparse=True, sort=True)

    X = vectorizer.fit_transform(feat_dicts)
    print(X)
    print(X.shape)
    print(len(y))

    k= 100

    ## Now do some feature selection and transformation
    binary_feature_selector = base_feature.MyFeatureSelector(vectorizer, k=k)
    y_bin = ['any' if y_ != 'none' else 'none' for y_ in y]
    X_bin = binary_feature_selector.fit_transform(X, y_bin)
    print(X_bin.shape)
    # Save feature names and scores
    binary_feature_selector.write_feature_scores('binary_lex_feature_scores.txt')
    # Pickle data for training and error analysis
    binary_data = (feat_dicts, relats, X_bin, y_bin, vectorizer, binary_feature_selector)
    outpath = '../data/binary_lexical_data.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump(binary_data, f)
    print("Saved binary data at {}".format(outpath))
#
    # To avoid running out of memory
    del X_bin
    del y_bin
    del binary_feature_selector

    # Now do the same for non-binary
    full_feature_selector = base_feature.MyFeatureSelector(vectorizer, k=k)
    X_full = full_feature_selector.fit_transform(X, y)
    print(X_full.shape)
    full_feature_selector.write_feature_scores('full_lex_feature_scores.txt')
    full_data = (feat_dicts, relats, X_full, y, vectorizer, full_feature_selector)
    outpath = '../data/full_lexical_data.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump(full_data, f)
    print("Saved non-binary data at {}".format(outpath))



    print("Made it to the end")



if __name__ == '__main__':
    main()
