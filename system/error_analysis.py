import os, sys
import re
import numpy as np
import pandas as pd
import pickle

sys.path.append('..')
sys.path.append('../basic')
from basic import annotation, made_utils

TRUTH_DIR = '../data/heldout_xmls/'

def main():
    #with open(CACHED_DATA_FILE, 'rb') as f:
    #    docs = pickle.load(f)

    class_names = list(sorted(['adverse', 'do','du', 'fr', 'manner/route', 'none', 'reason','severity_type'])) # This will map each type of truth label to each instance of a mix-up
    # This will map predicted classes to examples of errors
    confusions = {name: [] for name in class_names}

    confusion_matrix = np.zeros((8, 8))
    confusion_matrix = pd.DataFrame(confusion_matrix, columns=class_names, index=class_names)

    # Read in the saved false positives and false negatives
    fn_df = pd.read_csv('saved_false_negatives.txt', sep='\t')
    fp_df = pd.read_csv('saved_false_positives.txt', sep='\t')


    # Read in the truth files
    truth_reader = made_utils.TextAndBioCParser(TRUTH_DIR)
    truth_docs = truth_reader.read_texts_and_xmls()


    print(fp_df.head())
    # False positives
    # For each file, identify the true span -> span relationships and types
    #for filename, doc in truth_docs.items():
    num_samples = 0
    for filename in set(fp_df.filename):
        if filename.split('.')[0] not in truth_docs:
            continue
        doc = truth_docs[filename.split('.')[0]]

        # Subset all of the false positives to look at ones in this document
        rpt_df = fp_df[fp_df.filename == str(filename)]
        num_samples += len(rpt_df)

        # Map the annotation spans to the Annotations in the document
        annotations = doc.get_annotations()
        annotation_map = {a.start_index: a for a in annotations}
        # Map the false positive relations from span to the two annotations in the predicted relation
        pred_span_map = {(offset1, offset2): (annotation_map[offset1], annotation_map[offset2], pred_type)  for
                        (offset1, offset2, pred_type) in zip(rpt_df['offset_1'], rpt_df['offset_2'], rpt_df['pred_type'])}

        # Map the true relation spans to the relation itself
        truth_span_map = {(relation.annotation_1.start_index, relation.annotation_2.end_index): relation for relation in doc.get_relations()}

        # Now check if there was a relation between these two spans and we got the type wrong
        for span in pred_span_map.keys():
            #print(span)
            anno1, anno2, pred_label = pred_span_map[span]
            if span in truth_span_map: # this means that there is a relation, but we gave it the wrong label
                true_label = truth_span_map[span]
            else: # this means that there shouldn't have been any relation
                true_label = 'none'

            confusion_matrix[true_label][pred_label] += 1

            # Create a new Relation item to show our mistake
            pred_relation = annotation.RelationAnnotation(None, anno1, anno2, filename.split('.')[0], type=true_label)
            confusions[pred_label].append(pred_relation)

    #  Now go through and save examples of the errors
    for class_name in class_names:
        errors = sorted(confusions[class_name], key=lambda x:x.file_name)
        print(class_name, len(errors))
        if len(errors) == 0:
            continue
        f = open('error_analysis/fp/{}_errors.txt'.format(re.sub('/', '-', class_name)), 'w')
        for r in errors:
            doc = truth_docs[r.file_name]
            example_string = r.get_example_string(doc)
            f.write(r.file_name + '\n')
            f.write(str(r) + '\t{}:{}\t'.format(r.annotation_1.start_index, r.annotation_2.start_index) + '\n')
            f.write('Annotations:\t{}\t\t:{}\n'.format(str(r.annotation_1), str(r.annotation_2)))
            f.write(example_string + '\n')
            f.write('\n----------------------------\n')

        f.close()

    print(confusion_matrix)
    confusion_matrix.to_csv('error_analysis/fp/confusion_matrix.tsv', sep='\t')






        # Go through the false positives
        #for ()







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
