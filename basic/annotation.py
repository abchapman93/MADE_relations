import bioc
import re
import made_utils
import tokenizing
from PyRuSH.RuSH import RuSH


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
        self.from_bioc(bioc_anno)

    def from_bioc(self, bioc_anno):

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
    def __init__(self, file_name, text, annotations={}, relations={}):
        self.file_name = file_name
        #self.text = text
        self.text = min_preprocess(text)
        self.bioc_annotations = annotations # a dictionary mapping annotation id's to bioc.Annotation objects
        self.bioc_relations = relations # a dictionary mapping relation id's to bioc.Relation objects

        self.relations = [] # A list containg RelationAnnotation objects
        self.annotations = []
        self._sentences = {} # Tokenized sentences, {offset: sentence}
        self._tokens = {} # Tokenized words, {offset: token}
        # NOTE : This "positive_label" relates to positive/possible cases of pneumonia
        self.positive_label = -1
        self.tokenized_document = None

        self.tokenize_document()
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


    def tokenize_document(self, doc_tokenizer=None):
        if not doc_tokenizer:
            doc_tokenizer = tokenizing.DocumentTokenizer()
        tokenized_doc = doc_tokenizer.tokenize_doc(self.text)
        for tokenized_sentence in tokenized_doc:
            tokens, spans = zip(*tokenized_sentence)
            idx = 0
            sentence = []
            offset = spans[0][0] # Beginning offset of the sentence
            for token, span in tokenized_sentence:
                self._tokens[span[0]] = token
                sentence.append(token)
            self._sentences[offset] = sentence


    def get_text_at_span(self, span):
        token_spans = sorted(self._tokens.items(), key=lambda x:x[0])
        length = len(token_spans)
        # First, get to the initial offset
        offset = 0
        while offset < span[0]:
            offset, _= token_spans[1] # Look at the next span in the list
            token_spans.pop(0)

        # Now iterate through until we get to the end of the span
        tokens = []
        idx = 0
        for offset, token in token_spans:
            if offset >= span[1]:
                break
            tokens.append(token)
        return tokens

    def get_tokens_before_or_after(self, offset, delta=-1, n=1):
        """
        Returns a list of all tokens that occur before or after offset up to n.
        Delta should be either 1 or -1 and defines whether to go forwards or backwards.
        If it reaches the beginning of a sentence, returns <PHI>.
        If it reaches the end, returns <OMEGA>
        """
        tokens = []
        if delta == -1:
            offset += delta # Step backwards/forwards until we find a new token
        while len(tokens) < n:
            if offset in self._sentences: # this means it's the start of a new sentence
                if delta == -1: # If this ist the beginning, we want to include this
                    tokens.append(self._tokens[offset])
                break
                # TODO: Decide what to do with sentence boundries
                #for diff in range(n - len(tokens)): # Add PHI as many times as necessary
                #    to_append = '<PHI>' if delta == -1 else '<OMEGA>'
                #    tokens.append(to_append)
            elif offset in self._tokens: # This means we've found a new token
                tokens.append(self._tokens[offset])
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
        if offset not in self._sentences:
            offsets = sorted([v for v in self._sentences.keys() if v < offset],
                            key=lambda v: offset-v)
            begin_offset = offsets[0]
        while offset not in self._sentences:
            if offset <= 0:
                break
            offset -= 1
        to_return.append(self._sentences[offset])
        # Now go back to the first offset and find the next sentence
        offset = span[0] + 1
        while offset < end:
            if offset in self._sentences:
                to_return.append(self._sentences[offset])
                break
            else:
                offset += 1

            if offset > len(self.text):
                break

        return to_return


    def wrong_get_sentences_in_span(self, span):
        """
        Iterates through the sentences and returns the lists of tokens up to span
        """
        sent_spans = sorted(self._sentences.items(), key=lambda x:x[0])
        # TODO: Use a faster search algorithm if you have to
        offset = 0
        while offset < span[0]:
            offset, _ = sent_spans[0]
            sent_spans.pop(0)

        # Now iterate through to the end of span
        sents = []
        idx = 0
        partial_sents = []
        for i in sent_spans:
        #for offset, sent in sent_spans:
            # If the next offset is larger, only append part of it
            offset = sent_spans[i][0]
            next_offset = sent_spans[i + 1][0]
            if next_offset >= span[1]:
                partial_sent = get_text_at_span()
                break
            sents.append(sent)



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

    def get_tokens(self):
        """
        Returns the tokens as a list of words
        """
        return [token for (offset, token) in sorted(self._tokens.items(), key=lambda x:x[0])]


    def get_annotations(self):
        return self.annotations

    def get_relations(self):
        return self.relations



    def __repr__(self):
        return "[{} annotations, {} relations]".format(len(self.annotations), len(self.relations))
        #return '[type=[{3} {1}:{2}]'.format(self.type, self.annotation_1.type, self.annotation_2.type,)

if __name__ == '__main__':
    reader = made_utils.TextAndBioCParser()
    docs = reader.read_texts_and_xmls(1)
    doc = list(docs.values())[0]
    annos = doc.get_annotations()
    for anno in annos:
        print(anno)
        print(doc.get_text_at_span(anno.span))

    exit()
