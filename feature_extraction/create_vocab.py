"""
This script generates a vocabulary out of all texts in the corpus.
"""
import glob, os, sys
import pickle
from collections import defaultdict
from nltk.tokenize.treebank import TreebankWordTokenizer

sys.path.append('..')
sys.path.append(os.path.join('..', 'basic'))
from basic import annotation
from basic.annotation import min_preprocess as preprocess


DATADIR = os.path.join(os.path.expanduser('~'), 'Box Sync', 'NLP_Challenge', 'MADE-1.0')
OUTDIR = os.path.join('..', 'data')

def main():
    vocab = defaultdict(int)
    tokenizer = TreebankWordTokenizer()
    corpus = glob.glob(os.path.join(DATADIR, 'corpus', '*'))
    for text_file in corpus:
        text = preprocess(open(text_file).read().lower())
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            vocab[token] += 1

    outpath = os.path.join(OUTDIR, 'vocab.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump(vocab, f)
    print("Saved {} terms at {}".format(len(vocab), outpath))
    print(list(vocab.items())[:20])



if __name__ == '__main__':
    main()
