"""
This script generates lexical features including:
    - n-grams before and after each annotation in the relation
    - pos features
    - number of entities between the two relation entities
"""
import os, sys
import pickle

sys.path.append(os.path.join('..', 'basic'))
import base_feature


DATADIR = os.path.join('..', 'data') # the processed data
MADE_DIR = os.path.join(os.path.expanduser('~'), 'Box Sync', 'NLP_Challenge', 'MADE-1.0') # the original MADE data


def main():
    # Load in data
    with open(os.path.join(DATADIR, 'generated_train.pkl'), 'rb') as f:
        relats = pickle.load(f)
    with open(os.path.join(DATADIR, 'annotated_documents.pkl'), 'rb') as f:
        docs = pickle.load(f)
    print("Loaded {} relations and {} docs".format(len(relats), len(docs)))


if __name__ == '__main__':
    main()
