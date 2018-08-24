import itertools
import random
import json
import pprint
import pickle
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# nltk.download('punkt')

import warnings
warnings.filterwarnings("ignore")


with open('/Users/christopher/Desktop/projects/Metis_Projects/Project_4/data/train-v2.0.json') as f:
    data = json.load(f)


# dictionary of wiki articles and their respective content presented as one long string
page = {}

# dictionary of questions pertaining to specific wiki page
qa = {}


def extract_content(data):
    for i in range(len(data["data"])):
        page[str(i)] = dict([("title", None), ("content", [])])
        qa[str(i)] = dict([("questions", []), ("answers",[])])


    for i in range(len(data["data"])):
        page[str(i)]["title"] = data["data"][i]["title"]
        for j in data["data"][i]["paragraphs"]:
            page[str(i)]["content"].append(j["context"])
            for k in range(len(j["qas"])):
                qa[str(i)]["questions"].append(j["qas"][k]["question"])

    # save unconcatenated paragraphs for each wiki page
    pickle_out = open('unconcat_paragraphs.pickle','wb')
    pickle.dump(page, pickle_out)
    pickle_out.close()

    # concatenate all paragraphs for a wiki page into 1 long string
    for i in range(len(data["data"])):
        page[str(i)]["content"] = ' '.join(page[str(i)]["content"])

    # save content & questions
    json.dump(page, open('wiki_pages.json','w'))
    json.dump(qa, open('wiki_questions.json','w'))



extract_content(data)

# create corpus of all wiki pages content
corpus_titles = []
corpus = []
for i in range(len(page)):
    corpus_titles.append(page[str(i)]["title"])
    corpus.append(page[str(i)]["content"])

# sanity check
number_of_docs = len(page)
corpus_length = len(corpus)
corpus_titles_length = len(corpus_titles)
#print('number of docs:', number_of_docs, '\n', 'corpus_length: ', corpus_length, '\n', 'corpus_title_length:', corpus_titles_length)



def top_wiki_indices(arr, n):

    ordered_cs = {}
    # select the inputed question's cosine similarity array
    cs_q = arr[-1]

    # delete cosine similarity value of 1 (the question itself)
    cs_q = np.delete(cs_q,-1)


    # get indices of top n wikis based on highest cosine similarity score
    wiki_indices = np.argpartition(cs_q, -n)[-n:]
    wiki_indices_list = list(wiki_indices)

    wiki_values_list = list(cs_q[wiki_indices])

    # create dictionary with index of cs value as keys, and actual cs value for values
    index_cs_values = dict(zip(wiki_indices_list, wiki_values_list))

    # created ordered dictionary (highest to lowest cs values)
    for key,value in sorted(index_cs_values.items(), key=lambda item:(item[1],item[0]), reverse=True):
        ordered_cs[key]=value

    return list(ordered_cs.keys())



def best_wiki(own_question=False):

    X_corpus = corpus
    X_corpus_titles = corpus_titles
    #print(len(X_corpus_titles), len(X_corpus))

    # choose a question & add it to the corpus
    if own_question==True:
        own_question = input('Ask a quesiton: ')
        X_corpus.append(own_question)

    else:
        wiki_num = random.randint(0,len(qa))
        chosen_q = random.choice(qa[str(wiki_num)]["questions"])
        X_corpus.append(chosen_q)

        print(chosen_q)

    tfidf = TfidfVectorizer(strip_accents='ascii', stop_words='english')
    X_tfidf = tfidf.fit_transform(X_corpus).toarray()

    # convert array of documents & tfidf values to dataframe
    df = pd.DataFrame(X_tfidf, columns=tfidf.get_feature_names())

    # compute cosine similarity
    cs_array = cosine_similarity(df)

    # Find which wiki page corresponds highest to question asked by finding
    # index of highest value in the question's cosine similarity array.
    # Index in cs_array is mapped 1 to 1 with the index of article titles in X_corpus_titles
    best_wiki = X_corpus_titles[np.argmax(np.delete(cs_array[-1],-1))]

    # remove appended question to maintain clean corpus
    del X_corpus[-1]

    print('Most Relevant Wiki Article: ', best_wiki)

    # get indices of top n wiki articles
    top_wikis = top_wiki_indices(cs_array, 5)

    # find top n wiki articles
    top_wikis_list = []

    for index in top_wikis:
        top_wikis_list.append(X_corpus_titles[index])

    top_wiki_index = top_wikis[0]

    # load unconcatenated paragraphs dictionary
    pickle_in = open('unconcat_paragraphs.pickle','rb')
    paragraphs = pickle.load(pickle_in)

    # get first few paragraphs of top wiki article
    few_paragraphs = []

    for num in range(3):
        few_paragraphs.append(paragraphs[str(top_wiki_index)]["content"][num])

    # best_wiki
    return top_wikis_list, '\n', '\n', '\n', few_paragraphs



def best_wiki_mini(question):

    X_corpus = corpus
    X_corpus_titles = corpus_titles

    X_corpus.append(question)

    tfidf = TfidfVectorizer(strip_accents='ascii', stop_words='english')
    X_tfidf = tfidf.fit_transform(X_corpus).toarray()

    # convert array of documents & tfidf values to dataframe
    df = pd.DataFrame(X_tfidf, columns=tfidf.get_feature_names())

    # compute cosine similarity
    cs_array = cosine_similarity(df)

    # Find which wiki page corresponds highest to question asked by finding
    # index of highest value in the question's cosine similarity array.
    # Index in cs_array is mapped 1 to 1 with the index of article titles in X_corpus_titles
    best_wiki = X_corpus_titles[np.argmax(np.delete(cs_array[-1],-1))]

    # remove appended question to maintain clean corpus
    del X_corpus[-1]

    top_wikis = top_wiki_indices(cs_array, 5)

    # find top n wiki articles
    top_wikis_list = []

    for index in top_wikis:
        top_wikis_list.append(X_corpus_titles[index])


    return top_wikis_list



def get_score(num=False):

    correct_list = []

    if num != False:
        num = num

    else:
        num = len(qa)


    for i in range(num):
        for j in qa[str(i)]["questions"]:

            # run best_wiki for each j, question
            top_wikis_list = best_wiki_mini(j)

            # check if a top 3 articles is correct one
            # append 1 if right, 0 if wrong
            if page[str(i)]["title"] in top_wikis_list:
                correct_list.append(1)
            else:
                correct_list.append(0)

    correct_percent = sum(correct_list) / len(correct_list)

    return correct_percent



if __name__ == "__main__":
    pprint.pprint(best_wiki(own_question=True))
