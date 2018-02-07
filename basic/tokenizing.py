from PyRuSH.RuSH import RuSH
from nltk.tokenize.treebank import TreebankWordTokenizer
from spacy.tokenizer import Tokenizer


class DocumentTokenizer(object):
    """
    Used to split a document into sentences and tokens.
    Returns a list of lists TODO
    """
    def __init__(self, rush=None, word_tokenizer=None):
        if not rush:
            rush = RuSH('rush_rules.tsv')
        if not word_tokenizer:
            word_tokenizer = TreebankWordTokenizer()

        self.rush = rush
        self.word_tokenizer = word_tokenizer

    def tokenize_doc(self, doc):
        """
        Takes raw string. Returns a list of lists where each list is the
        sentence, and each sentence contains two-tuples of tokens and spans.
        """
        tokenized_sents_and_spans = []
        sentence_spans = self.rush.segToSentenceSpans(doc)
        for sent_span in sentence_spans:
            sentence = doc[sent_span.begin: sent_span.end]
            tokenized_sents_and_spans.append(self.tokenize_sent(sentence, sent_span.begin))
        return tokenized_sents_and_spans

    def tokenize_sent(self, sentence, offset):
        tokens = self.word_tokenizer.tokenize(sentence)
        spans = self.word_tokenizer.span_tokenize(sentence)
        tokens_and_spans = []
        for token, span in zip(tokens, spans):
            start, end = span
            true_start = start + offset
            true_end = end + offset
            tokens_and_spans.append((token, (true_start, true_end)))
        return tokens_and_spans


if __name__ == '__main__':
    rush = RuSH('rush_rules.tsv')
    input_str = "The             patient was admitted on 03/26/08\n and was started on IV antibiotics elevation" +\
             ", was also counseled to minimizing the cigarette smoking. The patient had edema\n\n" +\
             "\n of his bilateral lower extremities. The hospital consult was also obtained to " +\
             "address edema issue question was related to his liver hepatitis C. Hospital consult" +\
             " was obtained. This included an ultrasound of his abdomen, which showed just mild " +\
             "cirrhosis. "

    word_tokenizer = TreebankWordTokenizer()
    doc_tokenizer = DocumentTokenizer(rush, word_tokenizer)
    doc_tokenizer.tokenize_doc(input_str)
    exit()

    sentences = rush.segToSentenceSpans(input_str)

    #nlp = spacy.load('en_core_web_sm')
    for sentence in sentences[:1]:
        print('Sentence({0}-{1}):\t>{2}<'.format(sentence.begin, sentence.end,
                                        input_str[sentence.begin:sentence.end]))

        text = input_str[sentence.begin: sentence.end]
        print(tokenizer.tokenize(text))
        print(tokenizer.span_tokenize(text))
        spans = tokenizer.span_tokenize(text)
        tokens = tokenizer.tokenize(text)
        for span, token in zip(spans, tokens):
            print(span, token)
            assert(text[span[0]:span[1]] == token)
