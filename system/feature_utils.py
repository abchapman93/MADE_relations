def save_example_feature_dict(feature_dict, doc, relat):
    f = open('example_feat_dict.txt', 'a')
    start, end = relat.span
    f.write(doc.text[start - 24:start])
    f.write('\n')
    f.write(' <ENTITIES:> ')
    f.write(doc.text[start: end])
    f.write(' </ENTITIES: ')
    f.write('\n')
    f.write(doc.text[end: end+24])
    f.write('\n\n')
    f.write('RELATION: {}'.format(str(relat)))
    f.write('\n\n')
    f.write('FEATURES:')
    for feature_name, feature_values in feature_dict.items():
        f.write('-- {}\n\t'.format(feature_name))
        if isinstance(feature_values, str):
            f.write(' {}'.format(feature_values))
        else:
            try:
                f.write('\t=>'.join(feature_values))
            except:
                f.write(' {}'.format(feature_values))
        f.write('\n')
    f.write('\n\n')
    f.write('-'*99)
    f.write('\n\n')
    f.close()
