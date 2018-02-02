"""
This module will contain helper classes and functions for parsing
the data found in annotations.
"""
import bioc
import os, glob

from . import annotation
#import annotation


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


    def read_texts_and_xmls(self, num_docs=-1):
        """
        Reads corresponding text and annotation files
        that are located in self.datadir
        num_docs: number of docs to read in. If left as default, will read all docs.
        """
        annotated_docs = {} # {file_name: (text, [annotation_1, ..., annotation_n])}
        text_files = glob.glob(os.path.join(self.datadir, 'corpus', '*'))
        for i, file_path in enumerate(text_files):
            if i % 25 == 0:
                print("{}/{}".format(i, num_docs if num_docs != -1 else len(text_files)))
            if i == num_docs:
                break
            file_name = os.path.basename(file_path)
            text, annotations, relations = self.read_text_and_xml(file_name)
            annotated_docs[file_name] = annotation.AnnotatedDocument(file_name, text, annotations, relations)
        return annotated_docs

    def read_text_and_xml(self, file_name):
        """
        Reads a single text file and its corresponding xml.
        """
        return [self.read_text_file(file_name)] + self.read_bioc_xml(file_name)


    def read_text_file(self, file_name):
        """
        Reads the text in a file_name
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
        annos = {anno.id: anno for anno in passage.annotations}
        relations = {relat.id: relat for relat in passage.relations}
        return [annos, relations]
        #annos = [EntityAnnotatio]
        return {'annotations': annos,
               'relations': relations}
