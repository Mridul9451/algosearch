import pandas as pd
import numpy as np
import re
import nltk
import pickle
import sys
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')

tfidf_tran = pickle.load(open('tfid_tran.pkl', 'rb'))
tfidf = pickle.load(open('tfid.pkl', 'rb'))

algo = pd.read_json('sample.json')
with open("vocabulary.txt", "r") as file:
    vocabulary = eval(file.readline())


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


def gen_vector_T(tokens):
    Q = np.zeros((len(vocabulary)))
    x = tfidf.transform(tokens)
    # print(tokens[0].split(','))
    for token in tokens[0].split(','):
        # print(token)
        try:
            ind = vocabulary.index(token)
            Q[ind] = x[0, tfidf.vocabulary_[token]]
        except:
            pass
    return Q


def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim


def cosine_similarity_T(k, query):
    preprocessed_query = preprocessed_query = re.sub("\W+", " ", query).strip()
    tokens = word_tokenize(str(preprocessed_query))
    q_df = pd.DataFrame(columns=['q_clean'])
    q_df.loc[0, 'q_clean'] = tokens
    q_df['q_clean'] = wordLemmatizer(q_df.q_clean)
    d_cosines = []

    query_vector = gen_vector_T(q_df['q_clean'])
    for d in tfidf_tran.A:
        d_cosines.append(cosine_sim(query_vector, d))

    out = np.array(d_cosines).argsort()[-k:][::-1]
    # print("")
    d_cosines.sort()
    a = pd.DataFrame()
    for i, index in enumerate(out):
        a.loc[i, 'index'] = str(index)
        a.loc[i, 'URL'] = algo['URL'][index]
        a.loc[i, 'Title'] = algo['Title'][index]
        a.loc[i, 'Description'] = algo['Description'][index]
    for j, simScore in enumerate(d_cosines[-k:][::-1]):
        a.loc[j, 'Score'] = simScore
    return a


x = cosine_similarity_T(10, sys.argv[1])
#x = cosine_similarity_T(5, 'median')
output = x.to_json()
print(output)
sys.stdout.flush()
# print(x)
