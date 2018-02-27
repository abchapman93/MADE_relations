import os, sys
import numpy as np
import pickle

# Import modules from this package
sys.path.append('..')
sys.path.append('../basic')
sys.path.append('../feature_extraction')
from feature_extraction.lexical_features import LexicalFeatureExtractor
from basic import made_utils

FEATURE_EXTRACTOR_FILE = os.path.join('..', 'data', 'lex_feature_extractors.pkl')
assert os.path.exists(FEATURE_EXTRACTOR_FILE)

BINARY_MODEL_FILE = os.path.join('..', 'saved_models', 'rf_lex_binary_model.pkl')
FULL_MODEL_FILE = os.path.join('..', 'saved_models', 'rf_lex_full_model.pkl')

for f in (BINARY_MODEL_FILE, FULL_MODEL_FILE):
    assert os.path.exists(f)

def read_in_data(datadir):
    """
    Reads in data using BioC parser.
    Pairs all possible relations as defined in made_utils.pair_annotations_in_doc
    """
    reader = made_utils.TextAndBioCParser(datadir=datadir)
    docs = reader.read_texts_and_xmls(-1, include_relations=False)
    # Now pair up all possible annotations
    for doc in docs.values():
        possible_relations = made_utils.pair_annotations_in_doc(doc, max_sent_length=3)
        doc.add_relations(possible_relations)
    return docs

def load_models_and_feature_extractors():
    """
    Loads in pre-trained classifiers and feature extractors
    that will transform possible relations into feature vectors.
    Returns:
        bin_clf - a model trained for binary classification
        full-clf - a model trained for full classification
        feature_extractor - Takes a relation and a doc and returns a feature dictionary
        binary_feature_selector - Takes a feature dictionary and returns a 100-column array with the 100 best features for binary classification
        full_feature_selector - Same as above, returns a matrix for the full classification
    """
    with open(BINARY_MODEL_FILE, 'rb') as f:
        bin_clf = pickle.load(f)
    with open(FULL_MODEL_FILE, 'rb') as f:
        full_clf = pickle.load(f)

    with open(FEATURE_EXTRACTOR_FILE, 'rb') as f:
        feature_extractor, binary_feature_selector, full_feature_selector = pickle.load(f)
    return bin_clf, full_clf, feature_extractor, binary_feature_selector, full_feature_selector


def get_non_zero_preds(pred):
    """
    Returns all of the indices where a relation is predicted to be 'any'
    """
    return [i for i in range(len(pred)) if pred[i][0] != 'n']


def filter_by_idx(idxs, *args):
    arrs = args
    z = list(zip(*arrs))
    z = [z[i] for i in idxs]
    to_return = list(zip(*z))
    if len(to_return) == 1:
        return to_return[0]
    return to_return


def main():
    # First, read in the texts and xmls as AnnotatedDocuments
    if cached: # Read in the files from the last run
        with open('../data/evaluation_annotated_docs.pkl', 'rb') as f:
            docs = pickle.load(f)
    else: # Read in the files directly and pickle them for next time
        docs = read_in_data(datadir)
        with open('../data/evaluation_annotated_docs.pkl', 'wb') as f:
            pickle.dump(docs, f)
    print(len(docs))

    # Load in the models
    (bin_clf, full_clf, feature_extractor,
     binary_feature_selector, full_feature_selector) = load_models_and_feature_extractors()

    # Now for each document, predict which relations are true
    for i, (file_name, doc) in enumerate(docs.items()):
        if i % 10 == 0 and i > 0:
            print("{}/{}".format(i, len(docs)))
        # First, binary clf
        possible_relations = doc.get_relations()
        if not len(possible_relations):
            continue

        feat_dicts = [feature_extractor.create_feature_dict(r, doc)
                        for r in possible_relations]


        X_bin = binary_feature_selector.vectorizer.transform(feat_dicts)
        X_bin = binary_feature_selector.transform(X_bin)
        pred_bin = bin_clf.predict(X_bin)

        # Tried using a lower threshold for binary classifier, but gave a very small bump in recall and hurt precision
        # [P('any'), P('none')]
        #pred_bin = np.empty(X_bin.shape[0], dtype=str)
        #probs_bin = bin_clf.predict_proba(X_bin)[:, 0] # Probability that they are 'any'
        #pred_bin[probs_bin >= 0.4] = 'any'
        #pred_bin[probs_bin < 0.4] = 'none'
        #pred_bin[1] = 'none' # Debugging purposes
##
        # Now filter out
        yes_idxs = get_non_zero_preds(pred_bin)
        possible_relations, feat_dicts = filter_by_idx(yes_idxs,
                                        possible_relations, feat_dicts)
        if not len(possible_relations):
            continue
        X_bin = X_bin[yes_idxs,:]

        # Now predict with the full classifier
        X_full = full_feature_selector.vectorizer.transform(feat_dicts)
        X_full = full_feature_selector.transform(X_full)
        pred_full = full_clf.predict(X_full)

        # Now filter out any that were predicted to be 'none'
        # We'll call this true relations
        yes_idxs = get_non_zero_preds(pred_full)
        pred_full, possible_relations = filter_by_idx(yes_idxs, pred_full, possible_relations)
        for label, relat in zip(pred_full, possible_relations):
            relat.type = label
        doc.relations = possible_relations

    # Now write bioc xmls
    for doc in docs.values():
        doc.to_bioc_xml('output')

        #print(doc.relations)
        #print(doc.get_relations()); exit()


    # For now, let's just make sure this is doing what we want
    with open('example_output.txt', 'w') as f:
        for i, (file_name, doc) in enumerate(docs.items()):
            f.write('{}\n\n'.format(file_name))
            f.write('\n'.join([r.get_example_string(doc) for r in
                                doc.get_relations()]))
            for r in doc.get_relations():
                f.write(str(r) + '\n')
                f.write('annotation1: {} \t annotation2: {}\n'.format(r.annotation_1.id, r.annotation_2.id))
            f.write('\n\n-----------\n\n')





if __name__ == '__main__':
    #datadir = sys.argv[1] #
    datadir = os.path.join('..', 'data', 'heldout_xmls')
    assert os.path.exists(datadir)
    cached = '--cached' in sys.argv
    main()
