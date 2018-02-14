"""
This script generates a vocabulary out of all texts in the corpus.
Creates up to trigrams, cushioned by PHI and OMEGA.
"""
import glob, os, sys
import re
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

from lexical_features import normalize_grams


#DATADIR = os.path.join(os.path.expanduser('~'), 'Box Sync', 'NLP_Challenge', 'MADE-1.0')
DATADIR = os.path.join('..', 'data')
INPATH = os.path.join(DATADIR, 'training_documents_and_relations.pkl')

def main():
    vocab = defaultdict(int)
    vocab_tags = defaultdict(int)
    # TODO: Should we remove stop words?
    word_tokenizer = TreebankWordTokenizer()
    sent_tokenizer = tokenizing.DefaultSentenceTokenizer()

    #docs = made_utils.read_made_data()
    with open(INPATH, 'rb') as f:
        docs, _ = pickle.load(f)

    print("Loaded {} documents".format(len(docs)))
    #for text_file in corpus:
        #doc = annotation.AnnotatedDocument(text_file, text)
    for doc in docs:
        tokens = doc.get_tokens()
        tags = doc.get_tags()
        # NOTE: lowercasing tokens for the vocab, will include capitalization in a separate feature
        tokens_list = [t.lower() for t in tokens]
        tags = [t.lower() for t in tags]

        # Create up to trigrams
        for n in (1, 2, 3):
            grams = ngrams(tokens_list, n)
            tag_grams = ngrams(tags, n)
            for gram in grams:
                # Sort the ngram so that it doesn't matter the order it occurs in
                # Transform it into a string
                gram_string = ' '.join(sorted(gram))
                # Normalize
                gram_string = normalize_grams(gram_string)
                vocab[gram_string] += 1
            for tag_gram in tag_grams:
                vocab_tags[' '.join(sorted(tag_gram))] += 1


    outpath = os.path.join(DATADIR, 'vocab.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump((vocab,vocab_tags), f)
    print("Saved {} terms and {} tags at {}".format(len(vocab), len(vocab_tags), outpath))
    print(list(vocab.items())[:20])



if __name__ == '__main__':
    main()
