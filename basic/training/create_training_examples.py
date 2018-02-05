"""
This script will create RelationAnnotation objects out of every possible annotation pair in the data documents.
"""
from collections import defaultdict
import os
import sys
import pickle

sys.path.append('..')
import made_utils
import annotation


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

    with open('edge_counts.pkl', 'wb') as f:
        pickle.dump(edges, f)

    print("Saved edge counts")
    return [x for x in edges.keys() if edges[x] >= thresh]

def load_legal_edges(thresh=1):
    """
    Reads a pickled dictionary of edge counts.
    Returns a list of edges with a count > thresh
    """
    with open('edge_counts.pkl', 'rb') as f:
        edges = pickle.load(f)
    print("Loaded legal edges")
    return [x for x in edges.keys() if edges[x] >= thresh]


def create_all_relations(docs, legal_edges):
    """
    Takes a dictionary of AnnotatedDocuments.
    Returns a list of all possible relations between entity annotations in those documents.
    These consist of either true relations that are found in RelationAnnotation objects
    or in artifical, "negative" samples that were generated between two non-related entities.
    """
    relations = defaultdict(list)
    for fname, doc in docs.items():
        true_annotations = doc.get_annotations()
        true_relations = doc.get_relations()
        connected_annotations = defaultdict(list) # mapping of tuples (id, id): RelationAnnotation
        # Map all annotation_1's to annotation_2's
        # in order to identify all positive examples of relations
        for relat in true_relations:
            anno1, anno2 = relat.get_annotations()
            connected_annotations[anno1.id].append(anno2.id)
        # Now create negative relations between each anno and all other anno's
        fake_relations = []
        for anno in true_annotations:
            for other_anno in true_annotations:
                if anno.id == other_anno.id:
                    continue
                if (anno.type, other_anno.type) not in legal_edges:
                    continue
                elif other_anno.id not in connected_annotations[anno.id]:
                    fake_relation = annotation.RelationAnnotation.from_null_rel(anno, other_anno)
                    fake_relations.append(fake_relation)


        for r in true_relations: # append doc so we can use it later
            relations[r.type].append((r,doc))
        for fr in fake_relations:
            relations[fr.type].append((fr, doc))

    return relations
    print(relations); exit()
    annos = doc.get_annotations()


def main():
    # First, read in data as a dictionary
    docs = made_utils.read_made_data()
    # Load in legal edges
    legal_edges = load_legal_edges()
    # Now generate all possible relation annotations
    all_possible_relations = create_all_relations(docs, legal_edges)
    for rel_type in all_possible_relations.keys():
        print("{}: {} relations".format(rel_type, len(all_possible_relations[rel_type])))
    with open('generated_train.pkl', 'wb') as f:
        pickle.dump(all_possible_relations, f)
    print("Saved generated training examples")


if __name__ == '__main__':
    main()
    exit()
    #legal_edges = define_legal_edges()
    legal_edges = load_legal_edges()
    print(legal_edges)
    #main()
