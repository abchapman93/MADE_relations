"""
This script will take all of the training data and convert it into AnnotatedDocuments
with true and fake relations in order to train with the entire set.
"""

import os, sys
import pickle
import random

sys.path.append('../../basic')
import made_utils


DATADIR = 'data'

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
    print("Original Distribution: {} positive relations, {} negative relations".format(
                len(pos_relations),
                len(neg_relations)))
    print("{} positive relations, {} negative relations".format(len(pos_relations),
                len(neg_sample)))
    return pos_relations + neg_sample


def main():
    reader = made_utils.TextAndBioCParser()
    if cached:
        with open('all_training_documents_no_relations', 'rb') as f:
            docs = pickle.load(f)
    else:
        docs = reader.read_texts_and_xmls()
        with open('all_training_documents_no_relations', 'wb') as f:
            pickle.dump(docs, f)
    relations = []

    for i, doc in enumerate(docs.values()):
        # Add Fake relations for training
        print(i)
        all_relations = made_utils.pair_annotations_in_doc(doc, max_sent_length=3)
        relations += doc.get_relations()
#
    outpath = os.path.join(DATADIR, 'all_training_documents_and_relations.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump((docs, relations), f)

    #with open('data/non_sampled_training_documents_and_relations.pkl', 'rb') as f:
    #    docs, all_relations = pickle.load(f)
#
    #for doc in docs.values():
    #    doc.relations = []
#
    #print("{} docs and {} relations".format(len(docs), len(all_relations)))

    # Sample negative examples
    all_relations = sample_negative_examples(all_relations, 2.0)
    for i, relat in enumerate(all_relations):
        if i % 100 == 0:
            print("{}/{}".format(i, len(all_relations)))
        r_doc = docs[relat.file_name]
        r_doc.relations.append(relat)
    outpath = 'data/all_training_documents_and_relations.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump((docs, all_relations), f)


    print("Saved {} documents and {} relations at {}".format(len(docs), len(all_relations), outpath))



if __name__ == '__main__':
    cached = '-cached' in sys.argv
    assert os.path.exists(DATADIR)
    main()
