




def select_k_best(X, y, labels_type, vectorizer, k=1000):
    ch2 = SelectKBest(chi2, k=k)
    ch2.fit(X, y)

    features = vectorizer.get_feature_names()
    top_ranked = []
    for i, score  in enumerate(ch2.scores_):
        if score > 0:
            top_ranked.append((i, score))
    top_ranked = list(sorted(top_ranked, key=lambda x:x[1], reverse=True))
    #top_ranked_idx = list(map(list, zip(*top_ranked)))[0]
    with open('{}_lexical_feature_scores.txt'.format(labels_type), 'w'):
        pass
    f = open('{}_lexical_feature_scores.txt'.format(labels_type), 'a')
    for idx, score in top_ranked:
        feature_name = features[idx]
        line = "{}\t{}\n".format(feature_name, score)
        f.write(line)
    f.close()

    # Let's save ch2
    outpath = os.path.join(MODELDIR, '{}_chi2.pkl'.format(labels_type))
    with open(outpath, 'wb') as f:
        pickle.dump(ch2, f)
    print("Saved feature selector at {}".format(outpath))

def main():

    with open(os.path.join(DATADIR, 'data_lexical.pkl'), 'rb') as f:
        X, y_non_bin = pickle.load(f)

    with open(os.path.join(MODELDIR, 'lex_dict_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)


    #num_data = 10
    #X = X[:num_data, :]
    #y_non_bin = y_non_bin[:num_data]

    print("X: {}".format(X.shape))
    print("y: {}".format(len(y_non_bin)))

    # Transform y
    y_bin = [int(0) if label == 'none' else int(1)  for label in y_non_bin]
    y = y_non_bin
    y = np.array(y)

    for labels_type, labels in (('binary', y_bin), ('full', y)):
        select_k_best(X, labels, labels_type, vectorizer, k=1000)









if __name__ == '__main__':
    main()
