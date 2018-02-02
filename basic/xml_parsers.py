"""
This module will contain helper classes and functions for parsing
the data found in annotations.
"""
import bioc

class BiocParser(object):
    def __init__(self):
        pass

    def parse_file(self, file_name):
        with open(file_name) as parser:
            #collection_info = parser.get_collection_info()
            print(dir(parser))


if __name__ == '__main__':
    path = "C:\\Users\\u0752374\\Box Sync\\NLP_Challenge\\MADE-1.0\\annotations\\1_9.bioc.xml"
    parser = BiocParser()
    print(parser.parse_file(path))
