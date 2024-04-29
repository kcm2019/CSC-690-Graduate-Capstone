#import data_describe as dd
from data_describe.text.text_preprocessing import *
from tqdm import tqdm
import nltk

def read_file_with_progress(file_path):
    file_string = ''
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Reading lines"):
            file_string += line
    return file_string

# Usage
file_path = './adelphidata.txt'  # replace with your file path
file_string = read_file_with_progress(file_path)

tokens = to_list(tokenize(file_string))
print(tokens[0])

lower = to_list(to_lower(tokens))
no_punct = to_list(remove_punct(lower))
no_single_char_and_space = to_list(remove_single_char_and_spaces(no_punct))
no_stopwords = to_list(remove_stopwords(no_single_char_and_space))
lem_docs = to_list(lemmatize(no_stopwords))
stem_docs = to_list(stem(no_stopwords))
clean_text_docs = to_list(bag_of_words_to_docs(no_stopwords))
print(clean_text_docs[0])




