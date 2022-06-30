import pandas as pd
import numpy as np
import os
import re
import operator
import nltk
import json
import pickle
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from num2words import num2words
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')

algo = pd.read_json('sample.json')

algo['WordTokenize'] = algo['Description'] + algo['Title']

# Preprocessing Data
cnt = 0
for i in algo.WordTokenize:
    algo['WordTokenize'][cnt] = i.lower()
    cnt = cnt+1

algo.WordTokenize = algo.WordTokenize.replace(
    to_replace='lines:(.*\n)', value='', regex=True)
algo.WordTokenize = algo.WordTokenize.replace(
    to_replace='[.!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~]', value=' ', regex=True)  # remove punctuation except
algo.WordTokenize = algo.WordTokenize.replace(
    to_replace='-', value=' ', regex=True)
algo.WordTokenize = algo.WordTokenize.replace(
    to_replace='\s+', value=' ', regex=True)  # remove new line
algo.WordTokenize = algo.WordTokenize.replace(
    to_replace='  ', value='', regex=True)  # remove double white space
algo.WordTokenize = algo.WordTokenize.apply(
    lambda x: x.strip())  # Ltrim and Rtrim of whitespace

# print(algo['Description'])

# Word Tokenization
cnt = 0
for i in algo.WordTokenize:
    algo['WordTokenize'][cnt] = word_tokenize(i)
    cnt = cnt+1

# print(algo['WordTokenize'])


def wordLemmatizer(data):
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    file_clean_k = pd.DataFrame()
    for index, entry in enumerate(data):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if len(word) > 1 and word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                Final_words.append(word_Final)
            # The final processed set of words for each iteration will be stored in 'text_final'
                file_clean_k.loc[index, 'Keyword_final'] = str(Final_words)
                file_clean_k.loc[index, 'Keyword_final'] = str(Final_words)
                file_clean_k = file_clean_k.replace(
                    to_replace="\[.", value='', regex=True)
                file_clean_k = file_clean_k.replace(
                    to_replace="'", value='', regex=True)
                file_clean_k = file_clean_k.replace(
                    to_replace=" ", value='', regex=True)
                file_clean_k = file_clean_k.replace(
                    to_replace='\]', value='', regex=True)
    return file_clean_k


algo['Clean_Keyword'] = wordLemmatizer(algo['WordTokenize'])

# TF-IDF ALGORITHM

# Create Vocabulary
vocabulary = set()
for doc in algo.Clean_Keyword:
    vocabulary.update(doc.split(','))
vocabulary = list(vocabulary)
# Intializating the tfIdf model
tfidf = TfidfVectorizer(vocabulary=vocabulary)
# Fit the TfIdf model
tfidf.fit(algo.Clean_Keyword)
# Transform the TfIdf model
tfidf_tran = tfidf.transform(algo.Clean_Keyword)

with open('tfid.pkl', 'wb') as handle:
    pickle.dump(tfidf, handle)

with open('tfid_tran.pkl', 'wb') as handle:
    pickle.dump(tfidf_tran, handle)

with open("vocabulary.txt", "w") as file:
    file.write(str(vocabulary))
