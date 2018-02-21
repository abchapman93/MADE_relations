import os, pickle
import bioc
import re
import made_utils
import tokenizing
from PyRuSH.RuSH import RuSH

from nltk import pos_tag


def min_preprocess(text):
   """
   Preprocessing to try and prevent errors in pyRuSH
   TODO: Hopefully can figure out how to avoid this
   """
   text = re.sub('\*', '-', text)
   text = re.sub('[\[\]\(\)]', '-', text)
   text = re.sub('[\"]', '-', text)
   return text

class BaseAnnotation(object):
    def __init__(self):
        self.start_index = -1
        self.end_index = -1
        self.type = ''
        self.spanned_text = ''

    def __repr__(self):
        return '[{0}], {1}:{2}, type=[{3}]'.format(self.spanned_text, self.start_index, self.end_index, self.type)

    @property
    def text(self):
        return self.spanned_text

    @property
    def span(self):
        return (self.start_index, self.end_index)

    # adding this so that pyConText's HTML markup can work seamlessly
    def getSpan(self):
        return (self.start_index, self.end_index)

    def getCategory(self):
        # pyConText graph objects actually expect a list here
        return [self.type]

class EntityAnnotation(BaseAnnotation):
    """
    Takes a bioc Annotation object.
    Sets attributes to match that annotation.
    """

    def __init__(self, bioc_anno, file_name):
        super().__init__()
        self.file_name = file_name
        self.id = -1
        self.from_bioc(bioc_anno, file_name)

    def from_bioc(self, bioc_anno, file_name):

        assert len(bioc_anno.locations) == 1
        loc = bioc_anno.locations[0]

        self.bioc_anno = bioc_anno

        self.type = bioc_anno.infons['type']
        self.start_index = int(loc.offset)
        self.end_index = int(loc.offset) + int(loc.length)
        self.spanned_text = bioc_anno.text

        self.id = bioc_anno.id

        return


class RelationAnnotation(BaseAnnotation):
    """
    Takes a bioc.Relation object and two connected bioc.Annotation objects.
    """
    def __init__(self, bioc_rel, anno1, anno2, file_name, true_relation=True):
        super().__init__()
        self.file_name = file_name
        self.id = -1
        self.true_relation = true_relation
        self.annotation_1 = anno1
        self.annotation_2 = anno2
        if bioc_rel:
            self.id = bioc_rel.id
            self.type = bioc_rel.infons['type']
        else:
            self.id = str(anno1.id) + str(anno2.id)
            self.type = 'none'

    @classmethod
    def from_bioc_rel(cls, bioc_rel, anno1, anno2, file_name):
        # assert that the types are correct
        # these should be of bioc classes for now
        assert isinstance(bioc_rel, bioc.bioc_relation.BioCRelation)
        for anno in (anno1, anno2):
            assert isinstance(anno, EntityAnnotation)

        return cls(bioc_rel, anno1, anno2, file_name, true_relation=True)

    @classmethod
    def from_null_rel(cls, anno1, anno2, file_name):
        """
        Creates a new, fake relation.
        """
        return cls(None, anno1, anno2, file_name, true_relation=True)


        self.id = bioc_rel.id
        self.type = bioc_rel.infons['type']
        self.annotation_1 = anno1
        self.annotation_2 = anno2
        return self

        # If bioc_anno1 and bioc_anno2 are None and annotations is not empty,
        # find bioc_anno1 and bioc_anno2 in the list of annotations.


    def get_annotations(self):
        """
        Returns the two EntityAnnotation objects.
        """
        return [self.annotation_1, self.annotation_2]

    def get_span(self):
        spans = self.annotation_1.span + self.annotation_2.span
        return (min(spans), max(spans))

    def get_example_string(self, doc):
        first_span = self.annotation_1.span
        second_span = self.annotation_2.span
        sorted_entities = sorted((self.annotation_1, self.annotation_2), key=lambda x:x.span[0])
        entity1, entity2 = sorted_entities
        start, end = self.span
        string = ''
        string += doc.text[start -40 : start]
        string += '<{}>'.format(self.type.upper())
        string += '<{}>'.format(entity1.type.upper())
        string += entity1.text
        string += '</{}>'.format(entity1.type.upper())
        string += doc.text[entity1.span[1]:entity2.span[0]]
        string += '<{}>'.format(entity2.type.upper())
        string += entity2.text
        string += '</{}>'.format(entity2.type.upper())
        string += '</{}>\n\n'.format(self.type.upper())
        string += doc.text[end: end+40]
        string = re.sub('\n', ' ', string)
        string = re.sub('\t', ' ', string)
        string = re.sub('  ', ' ', string)
        return string

    @classmethod
    def create_relation(cls, bioc_anno1, bioc_anno2):
        return cls()


    @property
    def span(self):
        spans = self.annotation_1.span + self.annotation_2.span
        return (min(spans), max(spans))

    @property
    def spans(self):
        return (self.annotation_1.span, self.annotation_2.span)

    @property
    def entity_types(self):
        return (self.annotation_1.type, self.annotation_2.type)

    def __repr__(self):
        return "'{0}':'{1}', {2}:{3}, type={4}".format(self.annotation_1.text,
                                        self.annotation_2.text,
                                        self.entity_types[0],
                                        self.entity_types[1], self.type)



# this class encapsulates all data for a document which has been annotated_doc_map
# this includes the original text, its annotations and its tokenized self
class AnnotatedDocument(object):
    def __init__(self, file_name, text, annotations=[], relations=[]):
        self.file_name = file_name
        #self.text = text
        self.text = min_preprocess(text)
        self.bioc_annotations = annotations # a dictionary mapping annotation id's to bioc.Annotation objects
        self.bioc_relations = relations # a dictionary mapping relation id's to bioc.Relation objects

        self.relations = relations # A list containg RelationAnnotation objects
        #self.annotations = annotations
        self.annotations = [EntityAnnotation(a, file_name) for a in annotations]
        self._sentences = {} # Tokenized sentences, {offset: sentence}
        self._tokens = {} # Tokenized words, {offset: token}
        self._tag_seqs = {} # POS tag seentences
        self._tags = {} # POS tags
        # NOTE : This "positive_label" relates to positive/possible cases of pneumonia
        self.positive_label = -1
        self.tokenized_document = None

        self.tokenize_and_tag_document() # Tokenizes and tags
        if relations != []:
            self.connect_relation_pairs()




    def create_relation(self, relation, bioc_annotation_1, bioc_annotation_2, true_relation=True):
        """
        Takes two bioc annotation node objects and true_relation,
        a boolean that states whether this is a true relation or a
        generated negative sample.
        """
        # convert into EntityAnnotations
        annotation_1 = EntityAnnotation(bioc_annotation_1, self.file_name)
        annotation_2 = EntityAnnotation(bioc_annotation_2, self.file_name)
        relation_annotation = RelationAnnotation.from_bioc_rel(relation, annotation_1,
                                        annotation_2, self.file_name)
        return relation_annotation


    def tokenize_and_tag_document(self, doc_tokenizer=None, pos_tagger=None):
        """
        Tokenizes and tags a document.
        """
        # TODO: Make a more sophisticated POS tagging method
        if not pos_tagger:
            pass
        if pos_tagger:
            raise NotImplementedError("Custom taggers have not been added yet")

        if not doc_tokenizer:
            doc_tokenizer = tokenizing.DocumentTokenizer()
        tokenized_doc = doc_tokenizer.tokenize_doc(self.text)
        for tokenized_sentence in tokenized_doc:
            tokens, spans = zip(*tokenized_sentence)
            #POS tags
            tags = [tag for (token, tag) in pos_tag(tokens)]
            idx = 0
            sentence = []
            tag_seq = []
            offset = spans[0][0] # Beginning offset of the sentence


            # Save each token
            for token, span, tag in zip(tokens, spans, tags):
                self._tokens[span[0]] = token
                self._tags[span[0]] = tag
                sentence.append(token)
                tag_seq.append(tag)
            self._sentences[offset] = sentence
            self._tag_seqs[offset] = tag_seq


    def get_text_at_span(self, span):
        return self.get_tokens_or_tags_at_span(span, 'tokens')

    def get_tokens_at_span(self, span):
        return self.get_tokens_or_tags_at_span(span, 'tags')

    def get_tokens_or_tags_at_span(self, span, seq_type='tokens'):
        if seq_type == 'tokens':
            items = sorted(self._tokens.items(), key=lambda x:x[0])
        else:
            items = sorted(self._tags.items(), key=lambda x:x[0])

        length = len(items)
        # First, get to the initial offset
        offset = 0
        while offset < span[0]:
            offset, _= items[1] # Look at the next span in the list
            items.pop(0)

        # Now iterate through until we get to the end of the span
        tokens = []
        idx = 0
        for offset, token in items:
            if offset >= span[1]:
                break
            tokens.append(token)
        return tokens

    def get_tokens_before_or_after(self, offset, delta=-1, n=1, padding=True):
        return self.get_tokens_or_tags_before_or_after(offset, 'tokens',
                                                            delta=delta, n=n,
                                                            padding=padding)

    def get_tags_before_or_after(self, offset, delta=-1, n=1, padding=True):
        return self.get_tokens_or_tags_before_or_after(offset, 'tags',
                                                            delta=delta, n=n,
                                                            padding=padding)

    def get_tokens_or_tags_before_or_after(self, offset, seq='sentences', delta=-1, n=1, padding=True):
        """
        Returns a list of all tokens that occur before or after offset up to n.
        Delta should be either 1 or -1 and defines whether to go forwards or backwards.
        If it reaches the beginning of a sentence, returns <PHI>.
        If it reaches the end, returns <OMEGA>
        """
        if seq == 'tokens':
            items = self._tokens
        elif seq == 'tags':
            items = self._tags

        else:
            raise ValueError("Must be either 'tokens' or 'tags'")
        # TODO: Fix PHI and omega
        # Move all of these functions to the lexical features
        tokens = []
        offset += delta # Step backwards/forwards until we find a new token
        while len(tokens) < n:
            # If you've gone either before the first token or past the last
            if offset < 0 or offset > max(items.keys()):
                if padding and delta == 1:
                    tokens.append('OMEGA')
                elif padding and delta == -1:
                    tokens.append('PHI')
                break
            if offset in self._sentences: # this means it's the start of a new sentence
                if delta == -1: # If this is the beginning, we want to include this
                    tokens.append(items[offset])
                # If padding is true, include PHI or OMEGA
                if len(tokens) < n and padding and delta == -1:
                    tokens.append('PHI')
                elif len(tokens) < n and padding and delta == 1:
                    tokens.append('OMEGA')
                break
            elif offset in items: # This means we've found a new token
                tokens.append(items[offset])
                offset += delta
            else:
                offset += delta
        if delta == -1:
            return list(reversed(tokens))
        else:
            return tokens


    def in_same_sentence(self, span):
        """
        Checks whether a relation entity is within a single sentence.
        """
        if len(self.get_sentences_overlap_span(span)) == 1:
            return 1
        else:
            return 0
        #start, end = span
        #idx = start
        #while idx < end:
        #    if idx in self._sentences: # If the current idx is a key in the sentence spans, it marks a new sentence
        #        return 1
        #    idx += 1
        #return 0

    def get_sentences_overlap_span(self, span):
        """
        Iterates through the sentences and returns a list of sentences that OVERLAP
        with span.
        """
        to_return = []
        offset, end = span

        # first find the beginning of the current sentences
        while offset not in self._sentences:
            #sorted_offsets = list(sorted([v for v in self._sentences.keys()]))
            # Find the closest start of sentence to offset
            if offset <= 0:
                break
            offset -= 1

        # Now that we've broken out of that loop, we've found the beginning
        # of the sentence that contains offset
        to_return.append(self._sentences[offset])

        # Now go back to the first offset and find the next sentence
        offset = span[0] + 1
        while offset < end:
            if offset in self._sentences:
                # This means we've found a new sentence
                to_return.append(self._sentences[offset])
            else:
                pass
            offset += 1

            if offset > len(self.text): # We've already appended the final sentence
                break

        return to_return


    def get_token_at_offset(self, offset):
        if offset in self._tokens:
            return self._tokens[offset]
        else:
            raise ValueError("Something's not right.")


    def connect_relation_pairs(self):
        """
        For every relation in self.relations, finds the two
        annotations that it connects.
        """
        for relation_id, relation in self.bioc_relations.items():
            node1, node2 = relation.nodes
            # Use add_relation to add the two nodes
            bioc_annotation_1 = self.bioc_annotations[node1.refid]
            bioc_annotation_2 = self.bioc_annotations[node2.refid]
            relation_annotation = self.create_relation(relation,
                                    bioc_annotation_1,
                                    bioc_annotation_2, )
            annotation_1, annotation_2 = relation_annotation.get_annotations()
            # Now append annotations and relations
            self.annotations.extend([annotation_1, annotation_2])
            self.relations.append(relation_annotation)

    def get_sentences(self):
        """
        Returns the sentences as a list of lists of words
        """
        return [tokens for (offset, tokens) in sorted(self._sentences.items(), key=lambda x:x[0])]

    def get_tokens(self, spans=False):
        """
        Returns the tokens as a list of words
        """
        if not spans:
            return [token for (offset, token) in sorted(self._tokens.items(), key=lambda x:x[0])]
        else:
            return [(offset, token) for (offset, token) in sorted(self._tokens.items(), key=lambda x:x[0])]



    def get_tags(self, spans=False):
        """
        Returns the pos tags as a list of strings
        """
        if not spans:
            return [string for (offset, string) in sorted(self._tags.items(), key=lambda x:x[0])]
        else:
            return [(offset, string) for (offset, string) in sorted(self._tags.items(), key=lambda x:x[0])]


    def get_annotations(self):
        return self.annotations

    def get_relations(self):
        return self.relations

    def add_relations(self, relations):
        self.relations += relations

    def to_bioc_xml(self, outdir):
        outpath = os.path.join(outdir, self.file_name + '.bioc.xml')
        writer = bioc.BioCWriter()
        writer.collection = bioc.BioCCollection()
        collection = writer.collection
        document = bioc.BioCDocument()
        document.id = self.file_name

        passage = bioc.BioCPassage()
        passage.offset = '0'
        document.add_passage(passage)
        collection.add_document(document)

        # Add annotations that already have bioc annotations
        for anno in self.get_annotations():
            passage.add_annotation(anno.bioc_anno)

        for relat in self.get_relations():
            # Create new BioCRelation
            relation = bioc.bioc_relation.BioCRelation()
            relation.id = relat.id
            relation.put_infon('type', relat.type)

            # Reference that nodes that contain the annotations
            node1 = bioc.bioc_node.BioCNode()
            node1.role = 'annotation 1'
            node1.refid = relat.annotation_1.id
            relation.add_node(node1)

            node2 = bioc.bioc_node.BioCNode()
            node2.role = 'annotation 2'
            node2.refid = relat.annotation_2.id
            relation.add_node(node2)

            passage.add_relation(relation)

        writer.write(outpath)





    def __repr__(self):
        return "[{} annotations, {} relations]".format(len(self.annotations), len(self.relations))
        #return '[type=[{3} {1}:{2}]'.format(self.type, self.annotation_1.type, self.annotation_2.type,)

if __name__ == '__main__':
    outdir = '../data'
    assert os.path.exists(outdir)
    reader = made_utils.TextAndBioCParser()
    docs = reader.read_texts_and_xmls()
    ## Pickle the documents
    outpath = os.path.join(outdir, 'annotated_documents.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump(docs, f)
    print("Saved at {}".format(outpath))
    exit()
    doc = list(docs.values())[0]
    relat = doc.get_relations()[0]
    print(relat.get_example_string(doc))
    exit()
    print(doc.get_tags_before_or_after(0, -1, 6, padding=True))
    print(doc.get_tokens_before_or_after(10, -1, 6, padding=True))
    annos = doc.get_annotations()
    for anno in annos:
        print(anno)
        print(doc.get_text_at_span(anno.span))

    exit()
