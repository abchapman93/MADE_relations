import os, sys
import pickle

sys.path.append('..')
sys.path.append('../basic')
from basic import annotation

def main():
    with open(CACHED_DATA_FILE, 'rb') as f:
        docs = pickle.load(f)

    # Iterate through the files in annotation and find annotations that don't line up
    


if __name__ == '__main__':
    try:
        ANNOTATION_DIR=sys.argv[1]
        PREDICTION_DIR=sys.argv[2]
        TEXT_DIR=sys.argv[3]
    except IndexError:
        ANNOTATION_DIR = '../data/heldout_xmls/annotations'
        PREDICTION_DIR = './output'
        TEXT_DIR = '../data/heldout_xmls/corpus'
    CACHED_DATA_FILE = '../data/evaluation_annotated_docs.pkl'

    main()
