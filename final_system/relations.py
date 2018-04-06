import os, sys
import numpy as np
import pickle
from collections import defaultdict




sys.path.append('../../basic')
sys.path.append('../../basic/training')
sys.path.append('../../feature_extraction')

import train_utils

from lexical_features import LexicalFeatureExtractor
DATADIR = './data/test_data'
MODEL_DIR = './saved_models'
#OUTDIR = 'output_run_1/annotations'
OUTDIR = 'output_run_2/annotations'
assert os.path.exists(DATADIR)
assert os.path.exists(MODEL_DIR)
assert os.path.exists(OUTDIR)

import made_utils


def read_in_data(datadir, num_docs=-1):
    """
    Reads in data using BioC parser.
    Pairs all possible relations as defined in made_utils.pair_annotations_in_doc
    """
    reader = made_utils.TextAndBioCParser(datadir=datadir)
    docs = reader.read_texts_and_xmls(num_docs, include_relations=False)
    # Now pair up all possible annotations
    for doc in docs.values():
        possible_relations = made_utils.pair_annotations_in_doc(doc, max_sent_length=1)
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


def limit_by_string_concat(possible_relations, pred_full, pred_probs):

    new_pred_full = []
    concat_strs = []


    # Now, let's go through document and prevent any 'duplicate' edges
    # Where two entities with the exact same text are given the exact same relation
    # For now, we'll say that the first relation should be considered true
    doc_cache = {} # This will map prediction types to instances  and probabilities

    for i in range(len(possible_relations)):
        relat = possible_relations[i]
        r_pred = pred_full[i]
        concat = '{}:{}'.format(relat.annotation_1.text, relat.annotation_2.text)
        concat_strs.append(concat)
        if r_pred == 'none':
            new_pred_full.append(r_pred)
            continue
        #if r_pred not in ('do', 'du', 'fr', 'manner/route'):
        #    new_pred_full.append(r_pred)
        #    continue
        r_prob = max(pred_probs[i])

        if r_pred not in doc_cache:
            doc_cache[r_pred] = defaultdict(list)

        if concat in doc_cache[r_pred]: # This means we've already seen an instance like this and need to check the probabilities

            # Check the probabilities
            to_replace = [] # The indices for values that should be replaced with 'none'
            for (other_prob, other_idx) in doc_cache[r_pred][concat]:
                if r_prob > other_prob: # This one is more likely and we should keep it
                    to_replace.append(other_idx)
                else: # This one is less likely and we should replace it
                    r_pred = 'none'
                    num_replaced += 1
                    #new_pred_full.append(r_pred)

            # Append the new prediction
            new_pred_full.append(r_pred)
            # Replace the ones that are less likely
            for other_idx in to_replace:
                num_replaced += 1
                new_pred_full[other_idx] = 'none'

        else: # This means this is the first exact relation instance
            new_pred_full.append(r_pred)
        if r_pred != 'none':
            doc_cache[r_pred][concat].append((r_prob, i)) # Add to the cache

    #for i in range(len(pred_full)):
    #    if pred_full[i] != new_pred_full[i]:
    #        print(pred_full[i], new_pred_full[i])

    print(num_replaced)
    #if num_replaced:
    #    for i in range(len(pred_full)):
    #        print(concat_strs[i], pred_full[i], new_pred_full[i])
    pred_full = new_pred_full


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
    #if len(to_return) == 1:
    #    return to_return[0]
    return to_return


def main():
    # First, read in the texts and xmls as AnnotatedDocuments
    if cached: # Read in the files from the last run
        with open('data/evaluation_annotated_docs.pkl', 'rb') as f:
            docs = pickle.load(f)
    else: # Read in the files directly and pickle them for next time
        docs = read_in_data(datadir, -1)
        with open('data/evaluation_annotated_docs.pkl', 'wb') as f:
            pickle.dump(docs, f)
    print(len(docs))

    # Load in the models
    (bin_clf, full_clf, feature_extractor,
     binary_feature_selector, full_feature_selector) = load_models_and_feature_extractors()


    # Now for each document, predict which relations are true
    for idx, (file_name, doc) in enumerate(docs.items()):

        print(file_name)
        print("{}/{}".format(idx, len(docs)))
        if idx % 10 == 0 and idx > 0:
            print("{}/{}".format(idx, len(docs)))
        # First, binary clf
        possible_relations = doc.get_relations()
        if not len(possible_relations):
            continue

        relat_offsets = []
        for i, r in enumerate(possible_relations):
            anno1, anno2 = r.get_annotations()
            relat_offsets.append((anno1.start_index, anno2.start_index))

        feat_dicts = []
        print(len(possible_relations))
        for r in possible_relations:
            #if idx == 14:
            #    print('{}/{}'.format(i, len(possible_relations)))
            feat_dict = feature_extractor.create_feature_dict(r, doc)
            feat_dicts.append(feat_dict)
        if idx == 0:
            print(feat_dicts)

        # [feature_extractor.create_feature_dict(r, doc)
                        # for r in possible_relations]

        # Predict with the binary classifier
        X_bin = binary_feature_selector.vectorizer.transform(feat_dicts)
        X_bin = binary_feature_selector.transform(X_bin)
        pred_bin = bin_clf.predict(X_bin)
#
        # Now filter out
        yes_idxs = get_non_zero_preds(pred_bin)
        if not len(yes_idxs):
            continue
        possible_relations, relat_offsets, feat_dicts = filter_by_idx(yes_idxs,
                                        possible_relations, relat_offsets, feat_dicts)

        # Now predict with the full classifier
        X_full = full_feature_selector.vectorizer.transform(feat_dicts)
        X_full = full_feature_selector.transform(X_full)
        pred_full = full_clf.predict(X_full)

        # Now filter out any that were predicted to be 'none'
        # We'll call this true relations
        yes_idxs = get_non_zero_preds(pred_full)
        if not len(yes_idxs):
            continue
        pred_full, possible_relations = filter_by_idx(yes_idxs, pred_full, possible_relations)
        for label, relat in zip(pred_full, possible_relations):
            relat.type = label
        doc.relations = possible_relations
    # Now write bioc xmls
    num_written = 0
    for doc in docs.values():
        num_written += 1
        print(num_written)
        doc.to_bioc_xml(OUTDIR)
    print("Saved {} files at {}".format(len(docs), "output/annotations"))

        #print(doc.relations)
        #print(doc.get_relations()); exit()





if __name__ == '__main__':
    #datadir = sys.argv[1]
    datadir = DATADIR

    assert os.path.exists(datadir)
    BINARY_MODEL_FILE = os.path.join(MODEL_DIR, 'rf_binary_model.pkl')
    FULL_MODEL_FILE = os.path.join(MODEL_DIR, 'rf_full_model.pkl')
    FEATURE_EXTRACTOR_FILE = os.path.join( 'data', 'full_feature_extractors.pkl')
    for f in (BINARY_MODEL_FILE, FULL_MODEL_FILE, FEATURE_EXTRACTOR_FILE):
        print(f)
        assert os.path.exists(f)
    cached = '-cached' in sys.argv

    main()
