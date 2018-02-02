import bioc

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

    def __init__(self, bioc_anno):
        super().__init__()
        self.id = -1
        self.from_bioc(bioc_anno)

    def from_bioc(self, bioc_anno):

        assert len(bioc_anno.locations) == 1
        loc = bioc_anno.locations[0]

        self.bioc_anno = bioc_anno

        self.type = bioc_anno.infons['type']
        self.start_index = loc.offset
        self.end_index = loc.offset + loc.length
        self.spanned_text = bioc_anno.text

        self.id = bioc_anno.id

        return


class RelationAnnotation(BaseAnnotation):
    """
    Takes a bioc.Relation object and two connected bioc.Annotation objects.

    """
    def __init__(self, bioc_rel, anno1, anno2):
        super().__init__()
        self.id = -1
        self.from_bioc(bioc_rel, anno1, anno2)

    def from_bioc(self, bioc_rel, anno1, anno2):
        # assert that the types are correct
        # these should be of bioc classes for now
        assert isinstance(bioc_rel, bioc.bioc_relation.BioCRelation)
        for anno in (anno1, anno2):
            assert isinstance(anno, EntityAnnotation)

        self.id = bioc_rel.id
        self.type = bioc_rel.infons['type']
        self.annotation_1 = anno1
        self.annotation_2 = anno2

        # If bioc_anno1 and bioc_anno2 are None and annotations is not empty,
        # find bioc_anno1 and bioc_anno2 in the list of annotations.

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
    def __init__(self, file_name, text, annotations, relations):
        self.file_name = file_name
        self.text = text
        self.bioc_annotations = annotations # a dictionary mapping annotation id's to bioc.Annotation objects
        self.bioc_relations = relations # a dictionary mapping relation id's to bioc.Relation objects

        self.relations = [] # A list containg RelationAnnotation objects
        self.annotations = []
        self.sentences = [] # Tokenized sentences
        # NOTE : This "positive_label" relates to positive/possible cases of pneumonia
        self.positive_label = -1
        self.tokenized_document = None

        self.connect_relation_pairs()

    def connect_relation_pairs(self):
        """
        For every relation in self.relations, finds the two
        annotations that it connects.
        """
        for relation_id, relation in self.bioc_relations.items():
            node1, node2 = relation.nodes
            bioc_annotation_1 = self.bioc_annotations[node1.refid]
            bioc_annotation_2 = self.bioc_annotations[node2.refid]

            # convert into EntityAnnotations
            annotation_1 = EntityAnnotation(bioc_annotation_1)
            annotation_2 = EntityAnnotation(bioc_annotation_2)

            relation_annotation = RelationAnnotation(relation, annotation_1, annotation_2)
            self.relations.append(relation_annotation)
            self.annotations.extend([annotation_1, annotation_2])
        return



    def __repr__(self):
        return "[{} annotations, {} relations]".format(len(self.annotations), len(self.relations))
        #return '[type=[{3} {1}:{2}]'.format(self.type, self.annotation_1.type, self.annotation_2.type,)
