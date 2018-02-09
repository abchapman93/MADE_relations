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
import made_utils


DATADIR = os.path.join(os.path.expanduser('~'), 'Box Sync', 'NLP_Challenge', 'MADE-1.0')
OUTDIR = os.path.join('..', 'data')

def main():
    vocab = defaultdict(int)
    # TODO: Should we remove stop words?
    word_tokenizer = TreebankWordTokenizer()
    sent_tokenizer = tokenizing.DefaultSentenceTokenizer()

    docs = made_utils.read_made_data()
    #for text_file in corpus:
        #doc = annotation.AnnotatedDocument(text_file, text)
    for rpt_id, doc in docs.items():
        tokens = doc.get_tokens()
        # NOTE: lowercasing tokens for the vocab, will include capitalization in a separate feature
        tokens_list = [t.lower() for t in tokens]

        # Create up to trigrams
        for n in (1, 2, 3):
            grams = ngrams(tokens_list, n)
            for gram in grams:
                # Sort the ngram so that it doesn't matter the order it occurs in
                vocab[' '.join(sorted(gram))] += 1


    outpath = os.path.join(OUTDIR, 'vocab.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump(vocab, f)
    print("Saved {} terms at {}".format(len(vocab), outpath))
    print(list(vocab.items())[:20])



if __name__ == '__main__':
    main()
