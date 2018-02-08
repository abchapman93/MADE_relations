"""
This script generates lexical features including:
    - n-grams before and after each annotation in the relation
    - pos features
    - number of entities between the two relation entities
"""
import os, sys
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
        self._unfiltered_vocab = vocab
        self.vocab = self.create_vocab_idx(vocab)

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
            vocab_idx[i] = gram
        return vocab_idx

    def create_feature_vector(self, relat, doc):
        """
        Takes a RelationAnnotation and an AnnotatedDocument.
        Returns the defined lexical features.
        """
        print(relat)
        # Binary feature: Are they in the same sentence?
        print("Same sentence?")
        in_same_sentence = doc.in_same_sentence(relat.get_span())
        print(in_same_sentence)
        # Get all tokens in between
        grams_between = self.get_grams_between(relat, doc)
        print("Grams between")
        print(grams_between)
        print("Grams before")
        grams_before = self.get_grams_before(relat, doc)
        print(grams_before)
        print("Grams after")
        grams_after = self.get_grams_after(relat, doc)
        print(grams_after)

        exit()
        self.get_tokens_in_window(relat, doc)

    def get_grams_between(self, relat, doc):
        """
        Returns the N-grams between the two entities connected in relat
        """
        grams = []
        spans = relat.spans
        start = spans[0][1]
        end = spans[1][0]
        start, end = sorted((start, end))
        tokens_in_span = doc.get_text_at_span((start, end))
        for n in set(self.ngram_window):
            grams = grams + [' '.join(tup) for tup in list(nltk_ngrams(tokens_in_span, n))]
        return grams


    def get_grams_before(self, relat,doc):
        """
        Returns the n-grams before the first entity.
        """
        grams = []
        offset = relat.span[0]
        tokens_before = doc.get_tokens_before_or_after(offset, delta=-1, n=self.context_window[0])
        for n in set(self.ngram_window):
            grams = grams + [' '.join(tup) for tup in list(nltk_ngrams(tokens_before, n))]
        return grams

    def get_grams_after(self, relat, doc):
        """
        Returns the n-grams after the final entity.
        """
        grams = []
        offset = relat.span[1]
        tokens_after = doc.get_tokens_before_or_after(offset, delta=1, n=self.context_window[1])
        for n in set(self.ngram_window):
            grams = grams + [' '.join(tup) for tup in list(nltk_ngrams(tokens_after, n))]
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

    # Create features for each relation
    feature_extractor = LexicalFeatureExtractor(ngram_window=(1, 2), vocab=vocab)
    print(feature_extractor)
    for i, relat in enumerate(relats[1:]):

        #if relat.span[1] - relat.span[0] > 500:
        #    print(i)
        #    break
        #continue
        doc = docs[relat.file_name]
        span = relat.span
        feature_vector = feature_extractor.create_feature_vector(relat, doc)
        print(feature_vector)
        break




if __name__ == '__main__':
    main()
