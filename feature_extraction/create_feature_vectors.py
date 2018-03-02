# import os, sys
# from collections import defaultdict
# import re
# from random import shuffle
# import pickle
# from nltk import ngrams as nltk_ngrams
#
# import spacy
# import networkx as nx
#
#
#
# from sklearn.feature_extraction import DictVectorizer
#
# sys.path.append(os.path.join('..', 'basic'))
# import base_feature
# from base_feature import BaseFeatureExtractor
# from feature_utils import save_example_feature_dict
#
# DATADIR = os.path.join('..', 'data') # the processed data
#
#
# import os, sys
# from collections import defaultdict
# import re
# from random import shuffle
# import pickle
# from nltk import ngrams as nltk_ngrams
#
# from sklearn.feature_extraction import DictVectorizer
#
# sys.path.append(os.path.join('..', 'basic'))
# import base_feature
# from base_feature import BaseFeatureExtractor
# from feature_utils import save_example_feature_dict
#
# DATADIR = os.path.join('..', 'data') # the processed data
# MODELDIR = os.path.join('..', 'saved_models')
# MADE_DIR = os.path.join(os.path.expanduser('~'), 'Box Sync', 'NLP_Challenge', 'MADE-1.0') # the original MADE data


from lexical_features import *
from dependency import *




def main():
    nlp = spacy.load('en_core_web_sm')
    # Load in data
    inpath = os.path.join(DATADIR, 'training_documents_and_relations.pkl')
    with open(inpath, 'rb') as f:
        docs, relats = pickle.load(f)

    print("Loaded {} docs and {} relations".format(len(docs), len(relats)))
    #shuffle(relats)
    with open(os.path.join(DATADIR, 'vocab.pkl'), 'rb') as f:
        vocab, pos_vocab = pickle.load(f)




    feature_extractor = LexicalFeatureExtractor(context_window=(2, 2),
                            ngram_window=(1, 3), vocab=vocab, pos_vocab=pos_vocab,
                            min_vocab_count=20, min_pos_count=20)

    feat_dicts = [] # mappings of feature names to values
    y = [] # the relation types
    for i, relat in enumerate(relats):

        doc = docs[relat.file_name]

        feature_dict = feature_extractor.create_feature_dict(relat, doc)
        # NOTE: Adding dependency and constituent features
        feature_dict.update(create_dep_and_const_features(relat, doc, nlp))
        feat_dicts.append(feature_dict)
        y.append(relat.type)
        if i % 100 == 0 and i > 0:
            print("{}/{}".format(i, len(relats)))
            #break


    print(feat_dicts[0])
    # Create feature vectors
    vectorizer = DictVectorizer(sparse=True, sort=True)

    X = vectorizer.fit_transform(feat_dicts)
    print("Binary data")
    print(X)
    print(X.shape)
    print(len(y))

    k= 5000

    ## Now do some feature selection and transformation
    binary_feature_selector = base_feature.MyFeatureSelector(vectorizer, k=k)
    y_bin = ['any' if y_ != 'none' else 'none' for y_ in y]
    X_bin = binary_feature_selector.fit_transform(X, y_bin)
    print(X_bin.shape)
    # Save feature names and scores
    binary_feature_selector.write_feature_scores('binary_full_feature_scores.txt')
    # Pickle data for training and error analysis
    binary_data = (feat_dicts, relats, X_bin, y_bin)
    outpath = '../data/binary_full_data.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump(binary_data, f)
    print("Saved binary data at {}".format(outpath))
    print(X_bin.shape)

#
    # To avoid running out of memory
    del X_bin
    del y_bin

    # Now do the same for non-binary
    full_feature_selector = base_feature.MyFeatureSelector(vectorizer, k=k)
    X_full = full_feature_selector.fit_transform(X, y)
    print(X_full.shape)
    full_feature_selector.write_feature_scores('full_full_feature_scores.txt')
    full_data = (feat_dicts, relats, X_full, y, vectorizer, full_feature_selector) # TODO: Change this
    outpath = '../data/full_full_data.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump(full_data, f)
    print("Saved non-binary data at {}".format(outpath))
    print(X_full.shape)

    # Save feature extractor for use in prediction
    outpath = '../data/lex_feature_extractors.pkl'
    items = (feature_extractor, binary_feature_selector, full_feature_selector)
    with open(outpath, 'wb') as f:
        pickle.dump(items, f)
    print("Saved LexicalFeatureExtractor, binary feature selector and full feature selector at {} ".format(outpath))



    print("Made it to the end")

if __name__ == '__main__':
    main()
