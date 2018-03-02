
import os, sys
from collections import defaultdict
import re
from random import shuffle
import pickle
from nltk import ngrams as nltk_ngrams

import spacy
import networkx as nx



from sklearn.feature_extraction import DictVectorizer

sys.path.append(os.path.join('..', 'basic'))
import base_feature
#from base_feature import BaseFeatureExtractor
from feature_utils import save_example_feature_dict

DATADIR = os.path.join('..', 'data') # the processed data


def get_sents_with_annos(relat, doc, window=(3, 3)):
    """
    Returns the sentence that contains a given annotation.
    Replaces the text of the annotations with a tag <ENTITY-TYPE>
    """
    anno1, anno2 = relat.get_annotations()
    sorted_entities = list(sorted((anno1, anno2), key=lambda x:x.span[0]))
    sorted_spans = list(sorted((anno1.span, anno2.span), key=lambda x:x[0]))
    sents = doc.get_sentences_overlap_span(relat.span)
    tokens = []
    # Step back some window
    offset = sorted_spans[0][0]

    while offset not in doc._sentences:
        offset -= 1
        if offset < 0:
            break
        if offset in doc._tokens:
            tokens.insert(0, doc._tokens[offset].lower())

    # Now add an entity
    tokens.append('{}'.format(sorted_entities[0].type.upper()))

    # Now add all the tokens between them
    offset = sorted_spans[0][1]
    end = sorted_spans[1][0]
    while offset < end:
        if offset in doc._tokens:
            tokens.append(doc._tokens[offset].lower())
        offset += 1
    # Now add the second entity
    tokens.append('{}'.format(sorted_entities[1].type.upper()))

    # Now add a window on the right
    current_length = len(tokens)
    offset = sorted_spans[1][1]
    while offset not in doc._sentences:
        if offset > max(doc._tokens.keys()):
            break
        if offset in doc._tokens:
            tokens.append(doc._tokens[offset].lower())
        offset += 1


    return ' '.join(tokens)

    for sent in sents:
        text += ' '.join(sent)
        text = re.sub(anno1.text, '<>'.format(anno.type).upper(), text)
        text = re.sub(anno2.text, '<>'.format(anno.type).upper(), text)


def get_shortest_path(sent, relat, doc):
    """
    Returns the shortest path between the two annotations in relat.
    """
    start_node = None
    end_node = None
    links = []

    # Identify edges
    for token in doc:
        out_node = token
        in_node = token.head
        links.append((out_node, in_node))
        if token.text == relat.annotation_1.type.upper():
            start_node = token
        elif token.text == relat.annotation_2.type.upper():
            end_node = token

    # If this is a self-loop, return {}
    if start_node == end_node:
        return []
    # Create the graph
    graph = nx.Graph(links)
    try:
        path = nx.shortest_path(graph, source=start_node, target=end_node)
    except:
        path = []
    return path


def create_dep_path_features(path):
    path_full = '-'.join(['{}/{}'.format(t.text, t.dep_) for t in path])
    path_text = '-'.join(['{}'.format(t.text) for t in path])
    path_dep = '-'.join(['{}'.format(t.dep_) for t in path])
    path_length = len(path)
    feat_dict = {'path_full:{}'.format(path_full): 1,
                'path_text:{}'.format(path_text): 1,
                'path_dep:{}'.format(path_dep): 1,
                'path_length': path_length,
            }
    if path_length == 2:
        feat_dict['const_neighbors'] = 1
    return feat_dict


def find_lca(out_node, in_node):
    for ancestor in out_node.ancestors:
        if ancestor in in_node.ancestors:
            return {'lca': ancestor.text}
    return {}


def create_dep_and_const_features(relat, doc, nlp):
    if relat.annotation_1.type == relat.annotation_2.type:
        return {}
    if not doc.in_same_sentence(relat.get_span()):
        feat_dict = {'path_full:OOS': 1, 'path_text:OOS': 1, 'path_dep:OOS': 1}
        return feat_dict

    feat_dict = {}
    sent = get_sents_with_annos(relat, doc)
    nlp_doc = nlp(sent)
    path = get_shortest_path(sent, relat, nlp_doc)
    if path == []:
        return {}

    # Now format the path info
    feat_dict.update(create_dep_path_features(path))
    feat_dict.update(find_lca(path[0], path[-1]))
    return feat_dict






def main():
    nlp = spacy.load('en_core_web_sm')
    inpath = os.path.join(DATADIR, 'training_documents_and_relations.pkl')
    with open(inpath, 'rb') as f:
        docs, relations = pickle.load(f)

    feat_dicts = []
    for i, (rpt_id, doc) in enumerate(docs.items()):
        for relat in doc.get_relations():
            if relat.annotation_1.type == relat.annotation_2.type:
                feat_dicts.append({})
                continue
            if not doc.in_same_sentence(relat.get_span()):
                feat_dict = {'path_full:OOS': 1, 'path_text:OOS': 1, 'path_dep:OOS': 1}
                feat_dicts.append(feat_dict)
                continue
            feat_dict = {}
            sent = get_sents_with_annos(relat, doc)
            nlp_doc = nlp(sent)
            path = get_shortest_path(sent, relat, nlp_doc)
            if path == []:
                feat_dicts.append({})
                continue
            # Now format the path info
            feat_dict.update(create_dep_path_features(path))
            feat_dict.update(find_lca(path[0], path[-1]))
            feat_dicts.append(feat_dict)




if __name__ == '__main__':
    main()
