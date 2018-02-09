import os, sys
import pickle
from collections import defaultdict
sys.path.append('..')
import made_utils, annotation

DATADIR = os.path.join('..', '..', 'data')


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


def pair_annotations_in_doc(doc, legal_edges=[]):
    """
    Takes a single AnnotatedDocument that contains annotations.
    All annotations that have a legal edge between them are paired
    to create RelationAnnotations.
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
            if anno1.id == anno2.id:
                continue
            if len(legal_edges) and (anno1.type, anno2.type) not in legal_edges:
                continue
            elif anno2.id not in edges[anno1.id]:
                generated_relation = annotation.RelationAnnotation.from_null_rel(
                    anno1, anno2, doc.file_name
                )
                generated_relations.append(generated_relation)
    relations = true_relations + generated_relations
    return relations

if __name__ == '__main__':
    define_legal_edges()
