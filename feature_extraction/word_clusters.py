import os, sys
from random import shuffle

from nltk import ngrams as nltk_ngrams
from sklearn.feature_extraction import DictVectorizer

import pickle

import base_feature
sys.path.append(os.path.join('..', 'basic'))

CLUSTERS_FILE = os.path.join(os.path.expanduser('~'), 'Box Sync', 'NLP_Challenge', 'Data_Resources', 'Word_Clusters', 'WordClusters_K500_BatchKmeans_wikipedia-pubmed-and-PMC-w2v.pickle')
DATADIR = os.path.join('..', 'data') # the processed data


from collections import Counter

class WordClusterExtractor(base_feature.BaseFeatureExtractor):
    def __init__(self, vocab, clusters, ngram_window=(1, 1), context_window=(2, 2), min_vocab_count=5):
        self.ngram_window = ngram_window
        self.context_window = context_window
        self._unfiltered_vocab = vocab
        self.vocab = self.create_vocab(vocab, min_vocab_count, ngram_window)
        self.clusters = clusters # Dictionary mapping words to


    def create_feature_dict(self, relat, doc):
        cluster_features = {}

        cluster_features.update({
                'clusters_in_entity_1:<{}>'.format(c): 1 for c in
                self.get_clusters_in_entity(relat.annotation_1, doc)
                            })

        cluster_features.update({

            'clusters_in_entity_2:<{}>'.format(c): 1 for c in
            self.get_clusters_in_entity(relat.annotation_2, doc)
        })

        cluster_features.update({
            'clusters_in_between:<{}>'.format(c): 1 for c in
            self.get_clusters_between(relat, doc)
        })

        cluster_features.update({
            'clusters_before:<{}>'.format(c): 1 for c in
            self.get_clusters_before(relat, doc)
        })

        cluster_features.update({
            'clusters_after:<{}>'.format(c): 1 for c in
            self.get_clusters_after(relat, doc)
        })





        return cluster_features



    def get_clusters_in_entity(self, entity, doc):
        tokens = doc.get_tokens_or_tags_at_span(entity.span, 'tokens')
        cluster_tokens = [str(self.clusters[t]) if t in self.clusters else 'OOV' for t in tokens]
        clusters = []
        for n in range(self.ngram_window[0], self.ngram_window[1] + 1):
            grams = list(nltk_ngrams(cluster_tokens, n))
            grams = self.sort_ngrams(grams)
            clusters.extend(set(grams))
        return set(clusters)

    def get_clusters_between(self, relat, doc):
        clusters = []
        span1, span2 = relat.spans
        _, start, end, _ = sorted(span1 +span2)
        tokens_in_span = doc.get_tokens_or_tags_at_span((start, end), 'tokens')
        tokens_in_span = [token.lower() for token in tokens_in_span]
        cluster_tokens = [str(self.clusters[t]) if t in self.clusters else 'OOV' for t in tokens_in_span]

        for n in range(self.ngram_window[0], self.ngram_window[1] + 1):
            grams = list(nltk_ngrams(cluster_tokens, n))
            grams = self.sort_ngrams(grams)
            clusters.extend(set(grams))

        return set(clusters)


    def get_clusters_before(self, relat, doc):
        clusters = []
        offset = relat.span[0]
        tokens_before = doc.get_tokens_or_tags_before_or_after(offset, delta=-1,
            n=self.context_window[0], seq='tokens', padding=True)
        tokens_before = [token.lower() for token in tokens_before]
        cluster_tokens = [str(self.clusters[t]) if t in self.clusters else 'OOV' for t in tokens_before]

        for n in range(self.ngram_window[0], self.ngram_window[1] + 1):
            grams = list(nltk_ngrams(cluster_tokens, n))
            grams = self.sort_ngrams(grams)
            clusters.extend(set(grams))

        return set(clusters)

    def get_clusters_after(self, relat, doc):
        clusters = []
        offset = relat.span[1]
        tokens_after = doc.get_tokens_or_tags_before_or_after(offset, delta=1,
            n=self.context_window[1], seq='tokens', padding=True)
        tokens_after = [token.lower() for token in tokens_after]
        cluster_tokens = [str(self.clusters[t]) if t in self.clusters else 'OOV' for t in tokens_after]

        for n in range(self.ngram_window[0], self.ngram_window[1] + 1):
            grams = list(nltk_ngrams(cluster_tokens, n))
            grams = self.sort_ngrams(grams)
            clusters.extend(set(grams))

        return set(clusters)



    def sort_ngrams(self, ngrams):
        return [' '.join(sorted(tup)) for tup in ngrams]



def main():
    with open(os.path.join(DATADIR, 'training_documents_and_relations.pkl'), 'rb') as f:
        docs, relats = pickle.load(f)

    docs = {doc.file_name: doc for doc in docs}

    with open(CLUSTERS_FILE, 'rb') as f:
        clusters = pickle.load(f)
    with open(os.path.join(DATADIR, 'vocab.pkl'), 'rb') as f:
        vocab, _ = pickle.load(f)

    feature_extractor = WordClusterExtractor(vocab, clusters,
            ngram_window=(2, 2), context_window=(2, 2))

    feat_dicts = [] # mappings of feature names to values
    y = [] # the relation types
    for i, relat in enumerate(relats):

        doc = docs[relat.file_name]

        feature_dict = feature_extractor.create_feature_dict(relat, doc)
        feat_dicts.append(feature_dict)
        y.append(relat.type)
        if i % 100 == 0 and i > 0:
            print("{}/{}".format(i, len(relats)))
    print(relat)
    print(relat.get_example_string(doc))
    print(feat_dicts[0])

    vectorizer = DictVectorizer(sparse=True, sort=True)

    X = vectorizer.fit_transform(feat_dicts)
    print(X)
    print(X.shape)
    print(len(y))

    k= 100

    ## Now do some feature selection and transformation
    #binary_feature_selector = base_feature.MyFeatureSelector(vectorizer, k=k)
    #y_bin = ['any' if y_ != 'none' else 'none' for y_ in y]
    #X_bin = binary_feature_selector.fit_transform(X, y_bin)
    #print(X_bin.shape)
    ## Save feature names and scores
    #binary_feature_selector.write_feature_scores('binary_cluster_feature_scores.txt')
    ## Pickle data for training and error analysis
    #binary_data = (feat_dicts, relats, X_bin, y_bin, vectorizer, binary_feature_selector)
    #outpath = '../data/binary_cluster_data.pkl'
    #with open(outpath, 'wb') as f:
    #    pickle.dump(binary_data, f)
    #print("Saved binary data at {}".format(outpath))
#
    ## To avoid running out of memory
    #del X_bin
    #del y_bin
    #del binary_feature_selector

    # Now do the same for non-binary
    full_feature_selector = base_feature.MyFeatureSelector(vectorizer, k=k)
    X_full = full_feature_selector.fit_transform(X, y)
    print(X_full.shape)
    full_feature_selector.write_feature_scores('full_cluster_feature_scores.txt')
    full_data = (feat_dicts, relats, X_full, y, vectorizer, full_feature_selector)
    outpath = '../data/full_cluster_data.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump(full_data, f)
    print("Saved non-binary data at {}".format(outpath))


if __name__ == '__main__':
    main()
