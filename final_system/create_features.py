import sys

sys.path.append('../../feature_extraction')
sys.path.append('../../basic')
from lexical_features import *


DATADIR = './data'

def main():
    # Load in data
    inpath = os.path.join(DATADIR, 'all_training_documents_and_relations.pkl')
    with open(inpath, 'rb') as f:
        docs, relats = pickle.load(f)

    print("Loaded {} docs and {} relations from {}".format(len(docs), len(relats), inpath))
    #shuffle(relats)
    with open(os.path.join(DATADIR, 'vocab.pkl'), 'rb') as f:
        vocab, pos_vocab = pickle.load(f)




    feature_extractor = LexicalFeatureExtractor(context_window=(2, 2),
                            ngram_window=(1, 3), vocab=vocab, pos_vocab=pos_vocab,
                            min_vocab_count=20, min_pos_count=20)

    feat_dicts = [] # mappings of feature names to values
    y = [] # the relation types
    k= 1000
#     for i, relat in enumerate(relats):
#         if i % 100 == 0 and i > 0:
#             print("{}/{}".format(i, len(relats)))
#
#         doc = docs[relat.file_name]
#
#         feature_dict = feature_extractor.create_feature_dict(relat, doc)
#
#         feat_dicts.append(feature_dict)
#         y.append(relat.type)
#
#     print(feat_dicts[0])
#     # Create feature vectors
#     vectorizer = DictVectorizer(sparse=True, sort=True)
#
#     X = vectorizer.fit_transform(feat_dicts)
#     print("Binary data")
#     print(X)
#     print(X.shape)
#     print(len(y))
#
#
#     ## Now do some feature selection and transformation
#     binary_feature_selector = base_feature.MyFeatureSelector(vectorizer, k=k)
#     y_bin = ['any' if y_ != 'none' else 'none' for y_ in y]
#     X_bin = binary_feature_selector.fit_transform(X, y_bin)
#     print(X_bin.shape)
#     # Save feature names and scores
#     binary_feature_selector.write_feature_scores('binary_lex_feature_scores.txt')
#     # Pickle data for training and error analysis
#     binary_data = (feat_dicts, relats, X_bin, y_bin)
#     outpath = 'data/binary_data.pkl'
#     with open(outpath, 'wb') as f:
#         pickle.dump(binary_data, f)
#     print("Saved binary data at {}".format(outpath))
#     print(X_bin.shape)
# #
# ##
#     # To avoid running out of memory
#     del X_bin
#     del y_bin

    # NOTE: TO START IN MIDDLE

    with open('data/binary_data.pkl', 'rb') as f:
       feat_dicts, relats, X_bin, y_bin = pickle.load(f)
    del X_bin
    print ("Loaded {} feat_dicts and {} relats".format(len(feat_dicts), len(relats)))
    y = [r.type for r in relats]
    k = 1000
    vectorizer = DictVectorizer(sparse=True, sort=True)
    X = vectorizer.fit_transform(feat_dicts)
    binary_feature_selector = base_feature.MyFeatureSelector(vectorizer, k=k)
    X_bin = binary_feature_selector.fit_transform(X, y_bin)
    del X_bin

    # Now do the same for non-binary
    full_feature_selector = base_feature.MyFeatureSelector(vectorizer, k=k)
    X_full = full_feature_selector.fit_transform(X, y)
    print(X_full.shape)
    full_feature_selector.write_feature_scores('full_lex_feature_scores.txt')
    full_data = (feat_dicts, relats, X_full, y, vectorizer, full_feature_selector) # TODO: Change this
    outpath = 'data/full_data.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump(full_data, f)
    print("Saved non-binary data at {}".format(outpath))
    print(X_full.shape)

    # Save feature extractor for use in prediction
    outpath = 'data/full_feature_extractors.pkl'
    items = (feature_extractor, binary_feature_selector, full_feature_selector)
    with open(outpath, 'wb') as f:
        pickle.dump(items, f)
    print("Saved LexicalFeatureExtractor, binary feature selector and full feature selector at {} ".format(outpath))



    print("Made it to the end")

if __name__ == '__main__':
    main()
