"""
This script will create RelationAnnotation objects out of every possible annotation pair in the data documents.
"""
from collections import defaultdict
import random
import os
import sys
import pickle

sys.path.append('..')
import made_utils
import annotation
from train_utils import *


def sample_negative_examples(relations, neg_prop=1.0):
    """
    Takes a list of Relationannotations and
    neg_prop, a float that specifies the proportion of negative
    to positive examples.

    In the future, a more sophisticated method of sampling might be used,
    ie., sampling by the probability of the Annotation types in the nodes.
    """
    pos_relations = []
    neg_relations = []
    for relat in relations:
        if relat.type == 'none':
            neg_relations.append(relat)
        else:
            pos_relations.append(relat)

    pos_size = len(pos_relations)
    neg_sample_size = int(neg_prop * pos_size)

    neg_sample = random.sample(neg_relations, neg_sample_size)
    print("{} positive relations, {} negative relations".format(len(pos_relations),
                len(neg_sample)))
    return pos_relations + neg_sample


def write_doc_debug(doc, outdir):
    # For debugging, let's look in-depth at one tokenized document
    outpath = os.path.join(outdir, 'annotated_document.txt')
    with open(outpath, 'w') as f:
        f.write(doc.file_name )
        f.write('\n')
        f.write(str(doc))
        f.write('\n')
        f.write(doc.text)
        f.write('\n')
        f.write("Sentences:")
        for sent in doc.get_sentences():
            f.write(str(sent))
            f.write('\n')
        f.write("Tokenized words:{}".format('\n'.join(doc.get_tokens())))
        exit()
        f.write('\n')
        f.write('\n')
        f.write("Tokenized words: {}".format(doc._tokens))


def main():
    # First, read in data as a dictionary
    docs = made_utils.read_made_data()
    doc = docs[list(docs.keys())[0]]
    # Load in legal edges
    legal_edges = load_legal_edges()
    # Now generate all possible relation annotations
    #all_possible_relations = create_all_relations(docs, legal_edges)
    all_relations = []
    for fname, doc in docs.items():
        all_relations.extend(pair_annotations_in_doc(doc, legal_edges))

    #relation_types = defaultdict(int)
    #for relat in all_relations:
    #    relation_types[relat.type] += 1

    sample_relats = sample_negative_examples(all_relations, neg_prop=1.0)

    outpath = os.path.join(outdir, 'generated_train.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump(sample_relats, f)
    print("Saved {} training examples".format(len(sample_relats)))

    # Save the documents as well
    outpath = os.path.join(outdir, 'annotated_documents.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump(docs, f)
    print("Saved {} documents at {}".format(len(docs), outpath))
    with open(os.path.join(outdir, 'string_train.txt'), 'w') as f:
        f.write('\n'.join([str(r) for r in random.sample(sample_relats, 100)]))

if __name__ == '__main__':
    outdir = os.path.join('..', '..', 'data')
    assert os.path.exists(outdir)
    assert os.path.isdir(outdir)
    main()
