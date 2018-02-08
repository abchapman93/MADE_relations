from PyRuSH.RuSH import RuSH
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize import PunktSentenceTokenizer


class DocumentTokenizer(object):
    """
    Used to split a document into sentences and tokens.
    Returns a list of lists TODO
    """
    def __init__(self, sent_tokenizer=None, word_tokenizer=None):
        if not sent_tokenizer:
            #self.sent_tokenizer = ClinicalRushSentenceTokenizer('rush_rules.tsv')
            self.sent_tokenizer = DefaultSentenceTokenizer()
        if not word_tokenizer:
            self.word_tokenizer = TreebankWordTokenizer()

        #self.rush = rush
        #self.word_tokenizer = word_tokenizer

    def tokenize_doc(self, doc):
        """
        Takes raw string. Returns a list of lists where each list is the
        sentence, and each sentence contains two-tuples of tokens and spans.
        """
        tokenized_sents_and_spans = []
        try:
            # sentence_span is a list of tuples of spans
            sentence_spans = self.sent_tokenizer.tokenize_sents(doc)
        except Exception as e:
            raise e
            return []
            #raise e
        for start, end in sentence_spans:
            sentence = doc[start: end]
            tokenized_sents_and_spans.append(self.tokenize_sent(sentence, start))
        return tokenized_sents_and_spans

    def tokenize_sent(self, sentence, offset):
        try:
            tokens = self.word_tokenizer.tokenize(sentence)
        except Exception as e:
            print("Word tokenizing failed")
            print(sentence)
            raise e
        try:
            spans = self.word_tokenizer.span_tokenize(sentence)
        except Exception as e:
            print("Span tokenizing failed")
            print(sentence)
            raise e
        tokens_and_spans = []
        for token, span in zip(tokens, spans):
            start, end = span
            true_start = start + offset
            true_end = end + offset
            tokens_and_spans.append((token, (true_start, true_end)))
        return tokens_and_spans


class ClinicalRushSentenceTokenizer(object):
    def __init__(self, rules='./rush_rules.tsv'):
        self.rules = rules
        self.rush = RuSH(self.rules)

    def tokenize_sents(self, text):
        try:
            sent_spans = self.rush.segToSentenceSpans(text)
        except Exception as e:
            # Let's try to track down where this is happening in the text
            for i in range(int(len(text)/10)):
                start = i * 10
                end = start + 10
                try:
                    self.rush.segToSentenceSpans(text[start:end])
                except Exception as e:
                    with open('failed_snippet.txt', 'a') as f:
                        f.write(text[start:end] + '\n')
                    print("Failed at {}".format(start))
                    raise e
        sent_spans = [(s.begin, s.end) for s in sent_spans]
        return sent_spans


class DefaultSentenceTokenizer(object):
    def __init__(self):
        self.tokenizer = PunktSentenceTokenizer()

    def tokenize_sents(self, text):
        """
        Returns spans
        """
        return self.tokenizer.span_tokenize(text)


if __name__ == '__main__':
    rush = RuSH('rush_rules.tsv')
    input_str = "The             patient was admitted on 03/26/08\n and was started on IV antibiotics elevation" +\
             ", was also counseled to minimizing the cigarette smoking. The patient had edema\n\n" +\
             "\n of his bilateral lower extremities. The hospital consult was also obtained to " +\
             "address edema issue question was related to his liver hepatitis C. Hospital consult" +\
             " was obtained. This included an ultrasound of his abdomen, which showed just mild " +\
             "cirrhosis. "
    #input_str = open('failed.txt').read()[:-9001]

    sent_tokenizer = ClinicalRushSentenceTokenizer('rush_rules.tsv')
    sent_tokenizer = DefaultSentenceTokenizer()
    print(sent_tokenizer.tokenize_sents(input_str))
    #print(sent_tokenizer.span_tokenize(input_str))
    exit()
    print(sent_tokenizer.tokenize_sents(input_str))



    word_tokenizer = TreebankWordTokenizer()
    doc_tokenizer = DocumentTokenizer(rush, word_tokenizer)
    print(doc_tokenizer.tokenize_doc(input_str))
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
