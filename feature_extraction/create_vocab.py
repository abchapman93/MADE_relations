"""
This script generates a vocabulary out of all texts in the corpus.
Creates up to trigrams, cushioned by PHI and OMEGA.
"""
import glob, os, sys
import pickle
from collections import defaultdict
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk import ngrams

sys.path.append('..')
sys.path.append(os.path.join('..', 'basic'))
from basic import annotation
from basic import tokenizing
from basic.annotation import min_preprocess as preprocess


DATADIR = os.path.join(os.path.expanduser('~'), 'Box Sync', 'NLP_Challenge', 'MADE-1.0')
OUTDIR = os.path.join('..', 'data')

def main():
    vocab = defaultdict(int)
    word_tokenizer = TreebankWordTokenizer()
    sent_tokenizer = tokenizing.DefaultSentenceTokenizer()
    corpus = glob.glob(os.path.join(DATADIR, 'corpus', '*'))
    for text_file in corpus:
        text = preprocess(open(text_file).read().lower())
        doc = annotation.AnnotatedDocument(text_file, text)
        for sent in doc.get_sentences():
            # Place PHI and OMEGA on either side of the sentence
            sent_list = list(sent)
            for i in range(2):
                sent_list.insert(0, '<PHI>')
                sent_list.append('<OMEGA>')

            # Create up to trigrams
            for n in (1, 2, 3):
                grams = ngrams(sent_list, n)
                for gram in grams:
                    vocab[' '.join(gram)] += 1


    outpath = os.path.join(OUTDIR, 'vocab.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump(vocab, f)
    print("Saved {} terms at {}".format(len(vocab), outpath))
    print(list(vocab.items())[:20])



if __name__ == '__main__':
    main()
