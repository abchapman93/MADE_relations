from collections import defaultdict

from basic import annotation, made_utils
from basic.annotation import *

reader = made_utils.TextAndBioCParser()
docs = reader.read_texts_and_xmls(num_docs=-1)

info = defaultdict(int)
entity_types = defaultdict(int)
relation_types = defaultdict(list)

type_to_type_edges = defaultdict(int) # entity => entity
type_has_out_edge_to = defaultdict(dict)

for i, doc in enumerate(docs.values()):
    #if i % 100 == 0:
    #    print(i)
        # How many documents, entities, and relations?
    info['num_docs'] += 1
    info['num_entities'] += len(doc.bioc_annotations)
    info['num_relations'] += len(doc.bioc_relations)

    # How many distinct type -> type relations?
    for relation in doc.relations:
        type1, type2 = relation.entity_types
        entity_types[type1] += 1
        entity_types[type2] += 1
        relation_types[relation.type].append(' => '.join((type1, type2)))

        type_to_type_edges[' => '.join((type1, type2))] += 1

        if type2 in type_has_out_edge_to[type1]:
            type_has_out_edge_to[type1][type2] += 1
        else:
            type_has_out_edge_to[type1][type2] = 1

info['num_entity_types'] = len(entity_types)
info['num_relationship_types'] = len(type_to_type_edges)
# What types are never connected?

types_not_connected = []


# {do: {
#   'Drug => Dose': 5150,
#    'Dose => Drug': 27
#}}
type_dists = {}
for key, edges in relation_types.items():
    type_dists[key] = defaultdict(int)
    for edge in edges:
        type_dists[key][edge] += 1

for type1 in entity_types:
    for type2 in entity_types:
        if (type1, type2) not in type_to_type_edges:
            types_not_connected.append(' => '.join((type1, type2)))
        if (type2, type1) not in type_to_type_edges:
            types_not_connected.append(' => '.join((type2, type1)))

#print(info)
#print(type_to_type_edges)
#print(type_has_out_edge_to)

string = "# of documents: {num_docs}\nTotal # of entities: {num_entities}\nTotal # relations: {num_relations}\n# unique entity types: {num_entity_types}\n# unique relations: {num_relationship_types}\n\n".format(**info)
string += "Most common relation types: {}".format('\n\t'.join(str(tup) for tup in
                    type_dists.items()))
#string += "Most common edges: \n\t{}".format('\n\t'.join([str(tup) for tup in
#                                    sorted(type_to_type_edges.items(), key=lambda x:x[1], reverse=True)]))


string += "\n\n"
string += "Most common entities: \n\t{}".format('\n\t'.join(
                    [str(tup) for tup in
                    sorted(entity_types.items(), key=lambda x:x[1], reverse=True)]
                ))
string += "\n\n"
string += "Impossible edges: \n\t{}".format('\n\t'.join(set(types_not_connected)))
#print(string)

with open('exploration_results.txt', 'w') as f:
    f.write(string)
