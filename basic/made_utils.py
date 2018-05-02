"""
This module will contain helper classes and functions for parsing
the data found in annotations.
"""
import bioc
import os, glob, sys
sys.path.append('.')
import annotation
from collections import defaultdict
#import annotation

def read_made_data(num_docs=-1):
    """
    Returns a dictionary where the keys are filenames and
    the values are AnnotatedDocument objects.
    """
    reader = TextAndBioCParser()
    docs = reader.read_texts_and_xmls(num_docs)
    return docs


class TextAndBioCParser(object):
    def __init__(self, datadir=''):
        """
        datadir: the path leading to the directory containing 'corpus' and 'annotations'.

        """
        self.datadir = datadir
        if self.datadir == '':
            self.datadir = os.path.join(os.path.expanduser('~'), 'Box Sync', 'NLP_Challenge', 'MADE-1.0')

        assert os.path.exists(self.datadir)
        pass


    def read_texts_and_xmls(self, num_docs=-1, include_relations=True):
        """
        Reads corresponding text and annotation files
        that are located in self.datadir
        num_docs: number of docs to read in. If left as default, will read all docs.
        """
        annotated_docs = {} # {file_name: (text, [annotation_1, ..., annotation_n])}
        text_files = glob.glob(os.path.join(self.datadir, 'corpus', '*'))
        num_failed=0
        for i, file_path in enumerate(text_files):
            if i % 100 == 0:
                print("{}/{}".format(i, num_docs if num_docs != -1 else len(text_files)))
            if i == num_docs:
                break
            file_name = os.path.basename(file_path)
            try:
                text, annotations, relations = self.read_text_and_xml(file_name)
            except:
                print("{} failed".format(file_path))
                continue
                raise e
                num_failed += 1
                print(num_failed, i+1)
                continue
            if not include_relations: # If we're just reading in data for evaluation, we don't want any gold standard relations
                relations=[]

            try:
                annotated_docs[file_name] = annotation.AnnotatedDocument(file_name, text, annotations, relations,)
            except Exception as e:
                raise e
                pass
        return annotated_docs

    def read_text_and_xml(self, file_name):
        """
        Reads a single text file and its corresponding xml.
        """
        return [self.read_text_file(file_name)] + self.read_bioc_xml(file_name)


    def read_text_file(self, file_name):
        """
        Reads the text in a n
        """
        fullpath = os.path.join(self.datadir, 'corpus', file_name)
        with open(fullpath) as f:
            text = f.read()
        return text

    def read_bioc_xml(self, file_name):
        """
        Parses a bioc xml file.
        Returns a list of of lists of:
            annotations
            relations
        """
        full_path = os.path.join(self.datadir, 'annotations', file_name+".bioc.xml")
        bioc_reader = bioc.BioCReader(full_path)
        bioc_reader.read()
        doc = bioc_reader.collection.documents[0]
        passage = doc.passages[0]
        # NOTE: chaning annos and relats to lists
        annos = [anno for anno in passage.annotations]
        relations = [relat for relat in passage.relations]
        #annos = {anno.id: anno for anno in passage.annotations}
        #relations = {relat.id: relat for relat in passage.relations}
        return [annos, relations]
        #annos = [EntityAnnotatio]
        return {'annotations': annos,
               'relations': relations}

def pair_annotations_in_doc(doc, legal_edges=[], max_sent_length=3):
    """
    Takes a single AnnotatedDocument that contains annotations.
    All annotations that have a legal edge between them
    and are have an overlapping sentence length <= max_sent_length,
        ie., they are in either the same sentence or n adjancent sentences,
    are paired to create RelationAnnotations.
    Takes an optional list legal_edges that defines which edges should be allowed.

    Returns a list of new RelationAnnotations with annotation type 'none'.
    """
    if legal_edges == []:
        legal_edges = [('Drug', 'Route'),
                         ('Drug', 'Indication'),
                         ('SSLIF', 'Severity'),
                         ('Drug', 'Dose'),
                         ('Drug', 'Frequency'),
                         ('Drug', 'Duration'),
                         ('Drug', 'ADE'),
                         ('ADE', 'Severity'),
                         ('Indication', 'Severity'),
                         ('SSLIF', 'ADE')]
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
    # relations = true_relations + generated_relations
    # NOTE: Found a bug here , this should really only return the generated relations
    # return generated_relations
    return generated_relations + true_relations