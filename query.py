# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 13:02:04 2017

@author: admin
"""
import time
import codecs
import jieba
import re
import sqlite3
import pandas as pd
from collections import defaultdict
from gensim import corpora, models, similarities
#from pprint import pprint  # pretty-printer
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
TEMP_FOLDER = "d:/temp"

# def get_sql(query_string):
#     if query_string != None:
#         word_list = list(jieba.cut(query_string))
#         sql = "SELECT * FROM zhengfu_test where content like "
#         for i in range(len(word_list)):
#             if i == 0:  #是word_list里第一个词
#                 sql = sql + "'%" + word_list[i] + "%'"
#             else:
#                 sql = sql + "or content like '%" + word_list[i] + "%'"
#         print(sql)
#         return sql
#     else:
#         print("请输入要搜索的关键字！")
#         return None

def getData():
    cx = sqlite3.connect("d:/test2.db")

    sqlcmd = "select id, question from questiontable"
    try:
        data = pd.read_sql(sqlcmd,cx)
        return data
    except Exception as e:
        print e
        return None
    finally:
        cx.close()


def stoplist():
    path = 'stopwords.txt'
    reader = codecs.open(path, 'r', encoding='utf-8')
    stopwords = reader.read()
    reader.close()
    return stopwords

def filewordProcess(content):
    wordlist = []
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'\d', ' ', content)
    content = re.sub(r'\n', ' ', content)
    content = re.sub(r'\t', ' ', content)
    content = re.sub(r'[a-zA-Z]', ' ', content)
    for seg in jieba.cut(content,cut_all=False):
        if seg not in stoplist():
            if seg != ' ':
                wordlist.append(seg)
    file_content = ' '.join(wordlist)
    #print(file_content)
    return file_content


def cleaning(data):
    # remove common words and tokenize
    #stopwords = stoplist()
    questions = []
    id_list = []
    for i in range(len(data)):
        wordProcessed = filewordProcess(data.iloc[i, 1]) #标题列
        questions.append(wordProcessed)
        id_list.append(data.iloc[i, 0])
    print(questions)
    texts = [[word for word in question.split()] for question in questions]
    print(texts)
    texts = remove_word_once(texts)
    return texts, id_list


def remove_word_once(texts):
    # remove words that appear only once

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    clean_texts = [[token for token in text if frequency[token] > 1] for text in texts]
    return clean_texts
    

def localization(questions):
    dictionary = corpora.Dictionary(questions)
    dictionary.save(os.path.join(TEMP_FOLDER, 'deerwester.dict'))  # store the dictionary, for future reference
    print(dictionary)
    corpus = [dictionary.doc2bow(question) for question in questions]
    corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'deerwester.mm'), corpus)  # store to disk, for later use
    tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary)  # initialize an LSI transformation
    lsi.save(os.path.join(TEMP_FOLDER, 'model.lsi'))  # same for tfidf, lda, ...
    corpus_lsi = lsi[corpus_tfidf]
    index = similarities.MatrixSimilarity(corpus_lsi)  # transform corpus to LSI space and index it
    index.save(os.path.join(TEMP_FOLDER, 'deerwester.index'))

    #pprint(texts)

# def calculate_sims(texts):
#
#     dictionary, corpus = text_to_dict(texts)
#     # lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
#     tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model
#     corpus_tfidf = tfidf[corpus]
#     lsi = models.LsiModel(corpus_tfidf, id2word=dictionary)  # initialize an LSI transformation
#     corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
#     lsi.save(os.path.join(TEMP_FOLDER, 'model.lsi'))  # same for tfidf, lda, ...
#     # lsi = models.LsiModel.load(os.path.join(TEMP_FOLDER, 'model.lsi'))
#     cut_query = ' '.join(list(jieba.cut(query_string)))
#     vec_bow = dictionary.doc2bow(cut_query.split())
#     vec_lsi = lsi[vec_bow]  # convert the query to LSI space
#     # print(vec_lsi)
#     index = similarities.MatrixSimilarity(corpus_lsi)  # transform corpus to LSI space and index it
#     index.save(os.path.join(TEMP_FOLDER, 'deerwester.index'))
#     sims = index[vec_lsi]  # perform a similarity query against the corpus
#     # print(list(enumerate(sims))) # print (document_number, document_similarity) 2-tuples
#     sims = sorted(enumerate(sims), key=lambda item: -item[1])
#     return sims


def csims(querystring):
    dictionary = corpora.Dictionary.load('d:/temp/deerwester.dict')
    lsi = models.LsiModel.load(os.path.join(TEMP_FOLDER, 'model.lsi'))
    index = similarities.MatrixSimilarity.load('d:/temp/deerwester.index')
    #cut_query = ' '.join(list(jieba.cut(querystring)))
    cut = list(jieba.cut(querystring))
    cut_query = [word for word in cut if word not in stoplist()]
    print cut_query
    #cut_query = [[word for word in question.split()] for question in questions]
    vec_bow = dictionary.doc2bow(cut_query)
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    sims = index[vec_lsi]  # perform a similarity query against the corpus
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return sims


if __name__=='__main__':

    
    query_string = "口红有效果吗"

    try:
        time0 = time.time()

        #print data
        time1 =time.time()
        print "耗时：%s" % (time1 - time0)
        #data = data.drop_duplicates('questions')

        if not os.listdir(TEMP_FOLDER):
            data = getData()
            texts, id_list = cleaning(data)
            localization(texts)

        sims = csims(query_string)
        time1 = time.time()
        print "耗时：%s" % (time1-time0)
        for item in sims[:3]:
            print data.iloc[item[0], 0], data.iloc[item[0], 1], item[1]  #(id, title, 相似度)
    except Exception as e:
        print e
