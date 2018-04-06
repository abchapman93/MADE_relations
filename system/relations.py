import argparse
import os, sys
import numpy as np
import pickle
from collections import defaultdict


import train_utils
from lexical_features import LexicalFeatureExtractor
import made_utils


MODEL_DIR = './saved_models'
assert os.path.exists(MODEL_DIR)


# Paths to the pickled models and feature extractors
BINARY_MODEL_FILE = os.path.join(MODEL_DIR, 'rf_binary_model.pkl')
FULL_MODEL_FILE = os.path.join(MODEL_DIR, 'rf_full_model.pkl')
FEATURE_EXTRACTOR_FILE = os.path.join(MODEL_DIR, 'full_feature_extractors.pkl')
for f in (BINARY_MODEL_FILE, FULL_MODEL_FILE, FEATURE_EXTRACTOR_FILE):
    assert os.path.exists(f)



def read_in_data(datadir, num_docs=-1):
    """
    Reads in data using BioC parser.
    Pairs all possible relations as defined in made_utils.pair_annotations_in_doc
    """
    reader = made_utils.TextAndBioCParser(datadir=datadir)
    print("Reading in documents and annotations, tokenizing")
    docs = reader.read_texts_and_xmls(num_docs, include_relations=False)
    # Now pair up all possible annotations
    print("Pairing possible relations in text")
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
    #if len(to_return) == 1:
    #    return to_return[0]
    return to_return


def main():
    # First, read in the texts and xmls as AnnotatedDocuments
    docs = read_in_data(datadir, -1)
    print("Loaded in {} annotated documents".format(len(docs)))

    # Load in the models
    (bin_clf, full_clf, feature_extractor,
     binary_feature_selector, full_feature_selector) = load_models_and_feature_extractors()


    # Now for each document, predict which relations are true
    for idx, (file_name, doc) in enumerate(docs.items()):

        print("{}/{}".format(idx, len(docs)))

        # Candidate relations
        possible_relations = doc.get_relations()

        # If there are no possible relations, skip to the next document
        if not len(possible_relations):
            continue

        relat_offsets = []
        for i, r in enumerate(possible_relations):
            anno1, anno2 = r.get_annotations()
            relat_offsets.append((anno1.start_index, anno2.start_index))

        feat_dicts = []
        for r in possible_relations:
            feat_dict = feature_extractor.create_feature_dict(r, doc)
            feat_dicts.append(feat_dict)

        # Predict with the binary classifier
        X_bin = binary_feature_selector.vectorizer.transform(feat_dicts)
        X_bin = binary_feature_selector.transform(X_bin)
        pred_bin = bin_clf.predict(X_bin)
#
        # Now filter out any that we predict don't have a relation
        yes_idxs = get_non_zero_preds(pred_bin)
        if not len(yes_idxs):
            continue
        possible_relations, relat_offsets, feat_dicts = filter_by_idx(yes_idxs,
                                        possible_relations, relat_offsets, feat_dicts)

        # Now predict with the full classifier
        X_full = full_feature_selector.vectorizer.transform(feat_dicts)
        X_full = full_feature_selector.transform(X_full)
        pred_full = full_clf.predict(X_full)

        # Again filter out any that we don't think have any relations
        yes_idxs = get_non_zero_preds(pred_full)
        if not len(yes_idxs):
            continue
        # We'll call the remaining true relations
        pred_full, possible_relations = filter_by_idx(yes_idxs, pred_full, possible_relations)
        for label, relat in zip(pred_full, possible_relations):
            relat.type = label
        doc.relations = possible_relations
    # Now write bioc xmls
    for doc in docs.values():
        doc.to_bioc_xml(outdir)
    print("Saved {} files at {}".format(len(docs), os.path.join(outdir, 'annotations')))

        #print(doc.relations)
        #print(doc.get_relations()); exit()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read in input BIOC files and write out predicted relations.')
    parser.add_argument('--datadir',
            help='Relative path to the data directory containing subfolders annotations/ and corpus/',
            dest='datadir',
            required=True)
    parser.add_argument('--outdir',
            help='Relative path to the data directory to write out new bioc.xml files.',
            dest='outdir',
            required=True)
    args = parser.parse_args()
    datadir = os.path.join(os.path.abspath('.'), args.datadir)
    outdir = os.path.join(os.path.abspath('.'), args.outdir)
    print("Data will be read in from {} and saved to {}".format(datadir, outdir))

    assert os.path.exists(datadir)
    assert os.path.exists(outdir)
    main()
