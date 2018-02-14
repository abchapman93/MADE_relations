import os, sys
import re
import pickle
from collections import defaultdict
sys.path.append('..')
import made_utils, annotation

DATADIR = os.path.join('..', '..', 'data')


def save_errors(outpath, y, y_pred, feat_dicts, relats, docs):
    errors = defaultdict(list)
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            continue
        relat = relats[i]
        doc = docs[relat.file_name]
        true_type = y[i]
        pred_type = y_pred[i]
        example_string = relat.get_example_string(doc)

        string = 'TRUE TYPE: {} ------ PRED TYPE: {}\n'.format(true_type, pred_type)
        string += '{}\n'.format(doc.file_name)
        string += '{}\n\n'.format(str(relat))
        string += 'CONTEXT STRING: {}\n\n'.format(example_string)
        string += 'FEATURE DICTIONARY: \n{}'.format(str(feat_dicts[i]))

        string += '\n\n-----------------------------------\n\n'
        errors[true_type].append(string)
    for true_type, strings in errors.items():
        fname = '{}_errors.txt'.format(re.sub('/', '-', true_type))
        with open(fname, 'w') as f:
            f.write('\n'.join(strings))
        print("Saved {} error examples".format(true_type))

    pass


def define_legal_edges(thresh=1):
    """
    This function iterates through all of the annotations in the training data
    and defines legal edges as being between any two entity types
    that appear connected by any relation in the training data above some threshold.

    For example, if a threshold is set at 10, and it is found that there are only
    6 Relations connecting 'drug => drug', this is defined as illegal.
    """
    edges = defaultdict(int) # count of entity => entity edges
    entities = [] # all possible entities (nodes)
    # Dictionary mapping filenames to AnnotatedDocuments
    docs = made_utils.read_made_data()

    for i, doc in enumerate(docs.values()):
        for relation in doc.relations:
            edge = relation.entity_types
            entity_type1, entity_type2 = edge
            # If the two entity edges are the same, exclude them
            if entity_type1 == entity_type2:
                # Include this with a count of 0, so we can have a feature for it w/ real annotations
                edges[edge] = 0
                continue
            # Add one to this edge
            edges[edge] += 1
            # Append this entity
            entities.extend(edge)
    entities = set(entities)

    # Now add all edges that weren't seen
    # These will have a value of 0
    for entity in entities:
        for other in entities:
            edges[(entity, other)] += 0

    with open(os.path.join(DATADIR, 'edge_counts.pkl'), 'wb') as f:
        pickle.dump(edges, f)

    print("Saved edge counts")
    return [x for x in edges.keys() if edges[x] >= thresh]


def load_legal_edges(thresh=1):
   """
   Reads a pickled dictionary of edge counts.
   Returns a list of edges with a count > thresh
   """
   with open(os.path.join(DATADIR, 'edge_counts.pkl'), 'rb') as f:
       edges = pickle.load(f)
   print("Loaded legal edges")
   return [x for x in edges.keys() if edges[x] >= thresh]


def pair_annotations_in_doc(doc, legal_edges=[], max_sent_length=2):
    """
    Takes a single AnnotatedDocument that contains annotations.
    All annotations that have a legal edge between them
    and are have an overlapping sentence length <= max_sent_length,
        ie., they are in either the same sentence or n adjancent sentences,
    are paired to create RelationAnnotations.
    Takes an optional list legal_edges that defines which edges should be allowed.

    Returns a list of RelationAnnotations.
    """
    true_annotations = doc.get_annotations()
    true_relations = doc.get_relations()
    generated_relations = []
    edges = defaultdict(list)


    # Map all annotation_1's to annotation_2's
    # in order to identify all positive examples of relations
    # If this is testing data, it may not actually have these
    for relat in true_relations:
        anno1, anno2 = relat.get_annotations()
        edges[anno1.id].append(anno2.id)

    for anno1 in true_annotations:
        for anno2 in true_annotations:

            # Don't pair the same annotation with itself
            if anno1.id == anno2.id:
                continue

            if anno1.span == anno2.span:
                continue

            # Don't generate paris that have already been paried
            if anno2.id in edges[anno1.id]:
                continue

            # Exclude illegal relations
            if len(legal_edges) and (anno1.type, anno2.type) not in legal_edges:
                continue

            # Check the span between them, make sure it's either 1 or 2
            start1, end1 = anno1.span
            start2, end2 = anno2.span
            sorted_spans = list(sorted([start1, end1, start2, end2]))
            span = (sorted_spans[0], sorted_spans[-1])
            overlapping_sentences = doc.get_sentences_overlap_span(span)
            if len(overlapping_sentences) > max_sent_length:
                continue

            # If they haven't already been paired, pair them
            else:
                generated_relation = annotation.RelationAnnotation.from_null_rel(
                    anno1, anno2, doc.file_name
                )
                generated_relations.append(generated_relation)
    relations = true_relations + generated_relations
    return relations

if __name__ == '__main__':
    define_legal_edges()
