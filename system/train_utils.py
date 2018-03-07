import os, sys
import re
import pickle
from collections import defaultdict
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
sys.path.append('..')
import made_utils, annotation

DATADIR = os.path.join('..', '..', 'data')

def create_rfc():
    return RandomForestClassifier(
        criterion='entropy', max_depth=None, max_features=None, min_samples_leaf=1,
        min_samples_split=1
    )

def train_grid_search(X, y):
    # Transform y
    #y = [int(0) if label == 'none' else int(1)  for label in y_non_bin]

    #X = transform_features(X)

    #clf = LinearRegression()
    #train_models(X, y)
    clf = RandomForestClassifier()
    clf_name = 'RFC'
    param_grid = {
        'criterion': ['entropy', 'gini'],
        'max_depth': [1, 3, 5, 10, 25, 50, 75, 100, None],
        'min_samples_leaf': [1, 2, 3, 5, 10, 25, 100],
        'min_samples_split': [1, 2, 3, 5, 10, 25, 100],
        'max_features': [1, 3, 15, 10, 100, "sqrt", "log2", None]
            }

    # Smaller set
    param_grid = {
        'criterion': ['entropy', 'gini'],
        'max_depth': [1, 50, None],
        'min_samples_leaf': [1, 25, 100],
        'min_samples_split': [2, 25, 100],
        'max_features': [1, 100, "log2", None]
            }

    learned_parameters = grid_search(X, y, clf, param_grid)
    print(learned_parameters)
    return learned_parameters

def grid_search(X, y, clf, parameters):
    grid = GridSearchCV(clf, parameters, n_jobs=3, cv=3, verbose=1)
    print("doing grid search")
    fit_model = grid.fit(X, y)
    learned_parameters = fit_model.best_params_

    print(learned_parameters)

    return learned_parameters


def correct_preds_by_edges(relats, y, y_pred):

    EDGE_MAPPINGS = {
        '<DRUG>--<DOSE>': 'do',
        '<DRUG>--<INDICATION>':  'reason',
        '<DRUG>--<FREQUENCY>': 'fr',
        '<SSLIF>--<SEVERITY>': 'severity_type',
        '<ADE>--<SEVERITY>': 'severity_type',
        '<INDICATION>--<SEVERITY>': 'severity_type',
        '<DRUG>--<ROUTE>': 'manner/route',
        '<DRUG>--<ADE>': 'adverse',
        '<DRUG>--<DURATION>': 'du'
    }

    # Identify possible errors
    for i in range(len(relats)):
        if y[i] == y_pred[i]:
            continue





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
        fname = '{}_{}_errors.txt'.format(re.sub('/', '-', true_type), outpath)
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


def pair_annotations_in_doc(doc, legal_edges=[], max_sent_length=3):
    return made_utils.pair_annotations_in_doc(doc, legal_edges, max_sent_length)




if __name__ == '__main__':
    define_legal_edges()
