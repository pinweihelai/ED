# -*- coding: utf-8 -*-

import codecs
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
import os
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import  RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd
import pickle
import jieba
import re
import string
import time
import numpy as np
import pymysql
import datetime
from time import time

delEStr = string.punctuation + ' ' + string.digits
filename_list = []
category_list = []
words_list = []
heading_list = []
#top_feature_list = []
all_words = {}
rootpath = 'D:\文本分类\标签资讯'
category = os.listdir(rootpath)
n_features = 10000
n_topics = 20
n_top_words = 20

#======读取编码为utf8的本文文件=====
def readtext(path):
    text = codecs.open(path, 'r', encoding='utf-8')
    content = text.read()
    text.close()
    return content
stopwords = readtext('D:\文本分类\chinese_stopword.txt')

#======分词========

def filewordProcess(content):
    wordlist = []
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'\d', ' ', content)
    content = re.sub(r'\n', ' ', content)
    content = re.sub(r'\t', ' ', content)
    for seg in jieba.cut(content,cut_all=False):
        if seg not in stopwords:
            if seg != ' ':
                wordlist.append(seg)
    file_content = ' '.join(wordlist)
    return file_content


def MaxMinNormalization(x):
    x_1 =x.ix[:,:10000]
    x_v = x_1.values.astype(float)
    Min = x_v.min(axis =1)
    Cha = x_v.ptp(axis =1)
    for i in range(len(x_v)):
        for j in range(len(x_v[i])):
            x_v[i][j] = (x_v[i][j]-Min[i])/Cha[i]
    return  x_v

def write_to_txt(path,contents):
    f = codecs.open(path,'w','utf8')
    f.write(contents)
    f.close()

def print_top_words(model, feature_names, n_top_words):
    top_feature_list = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        top_feature_idx = topic.argsort()[:-n_top_words - 1:-1]
        #print(top_feature_idx)
        top_feature_list.extend(top_feature_idx)
        print(top_feature_list)
    return top_feature_list


# 创建词向量矩阵，创建tfidf值矩阵
def tfidf_mat(words_list,filename_list,category_list):
    tf_Vectorizer = CountVectorizer(stop_words='english',max_features=n_features)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
    tf = tf_Vectorizer.fit_transform(words_list)
    tfidf = tfidf_vectorizer.fit_transform(words_list)
    
    feature_names = tf_Vectorizer.get_feature_names()  # 特征名
#    tf_df = pd.DataFrame(tf.toarray())  # 全词库 词频 向量矩阵
#    tfidf_df = pd.DataFrame(tfidf.toarray())  # tfidf值矩阵
    #print(tfidf_df)
    # print freWordVector_df
    # tf-idf 筛选
    tfidf_sx_featuresindex = tfidf_df.sum(axis=0).sort_values(ascending=False)[:10000].index
    freWord_tfsx_df = tfidf_df.ix[:, tfidf_sx_featuresindex]  # tfidf法筛选后的idf矩阵
    df_columns = pd.Series(feature_names)[tfidf_sx_featuresindex]
    #tfidf_df_1 =freWord_tfsx_df
    min_max_scaler = preprocessing.MinMaxScaler()  #X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))X_scaled = X_std * (max - min) + min
    tfidf_df_1 = min_max_scaler.fit_transform(freWord_tfsx_df)
    tfidf_df_1 = pd.DataFrame(tfidf_df_1)
    tfidf_df_1.columns = df_columns
    #tfidf_df_1['label'] = tfidf_df['label']
    le = preprocessing.LabelEncoder()  # 评分到0到n_class-1
    tfidf_df_1['label'] = le.fit_transform(category_list)
    #print(tfidf_df_1['label'])#类别和数字的映射
    tfidf_df_1.index = filename_list
#    print(tfidf_df_1)
    return tfidf,tf, feature_names


# 卡方检验
def Chi_square_test(tfidf_df_1):
    ch2 = SelectKBest(chi2, k=10000)
    nolabel_feature = [x for x in tfidf_df_1.columns if x not in ['label']]
    ch2_sx_np = ch2.fit_transform(tfidf_df_1[nolabel_feature],tfidf_df_1['label'])
    label_np = np.array(tfidf_df_1['label'])
    X = ch2_sx_np
    y = label_np
    skf = StratifiedKFold(y,n_folds=5)
    y_pre = y.copy()
    for train_index,test_index in skf:
        X_train,X_test = X[train_index],X[test_index]
        y_train,y_test = y[train_index],y[test_index]
        clf1 = RandomForestClassifier(n_estimators=30,n_jobs=2,).fit(X_train, y_train)
        y_pre[test_index] = clf1.predict(X_test)
    clf = RandomForestClassifier(n_estimators=30,n_jobs=2).fit(X,y)
    joblib.dump(clf, "D:\文本分类\\train_model.m")
    print('准确率为 %.6f' % (np.mean(y_pre == y)))
    return y,y_pre,nolabel_feature


if __name__ in '__main__':
    for categoryName in category:
        if (categoryName == '.DS_Store'): continue
        categoryPath = os.path.join(rootpath, categoryName)  # 这个类别的路径
        filesList = os.listdir(categoryPath)  # 这个类别内所有文件列表
        # 循环对每个文件分词
        for filename in filesList:
            if (filename == '.DS_Store'): continue
            starttime = time.clock();
            contents = pd.read_table(categoryPath + '/' + filename, header=None, encoding='utf-8')
            context = contents[1]
            heading = contents[0]
            heading = heading.dropna()
            context = context.dropna()
            for i in range(len(context)):
                wordProcessed = filewordProcess(context[i])
                # 内容,标题分词成列表
                # filenameWordProcessed = fileWordProcess(filename) # 文件名分词，单独做特征
                # words_list.append((wordProcessed,categoryName,filename)) # 训练集格式：[(当前文件内词列表，类别，文件名)]
                words_list.append(wordProcessed)
                filename_list.append(filename)
                category_list.append(categoryName)
                endtime = time.clock();
                print('类别:%s >>>>文件:%s >>>>导入用时: %.3f' % (categoryName, filename, endtime - starttime))

    #tfidf,tf, feature_names = tfidf_mat(words_list, filename_list, category_list);
    
    tf_vectorizer = CountVectorizer(stop_words='english',max_features=n_features)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
    tf = tf_vectorizer.fit_transform(words_list)
    tfidf = tfidf_vectorizer.fit_transform(words_list)
    
    # Fit the NMF model
    print("Fitting the NMF model with tf-idf features, "
          "n_samples=%d and n_features=%d..."
          % (tf.shape[0], n_features))
    t0 = time()
    nmf = NMF(n_components=n_topics, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))
    
    print("\nTopics in NMF model:")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    feature_list = print_top_words(nmf, tfidf_feature_names, n_top_words)
    #feature_list = sum(feature, [])
    tf_df = pd.DataFrame(tf.toarray())  # 全词库 词频 向量矩阵
    tfidf_df = pd.DataFrame(tfidf.toarray())  # tfidf值矩阵
    tfidf_fs_df = tfidf_df.ix[:, feature_list]
    df_columns = pd.Series(tfidf_feature_names)[feature_list]
    tfidf_fs_df.columns = df_columns
    le = preprocessing.LabelEncoder()  # 评分到0到n_class-1
    tfidf_fs_df['label'] = le.fit_transform(category_list)
    tfidf_fs_df.to_csv('d:/文本分类/nmf_data.csv', index = False)
    
    print("Fitting LDA models with tf features, "
          "n_samples=%d and n_features=%d..."
          % (tf.shape[0], n_features))
    
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    t0 = time()
    lda.fit(tf)
    print("done in %0.3fs." % (time() - t0))
    
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    feature_list = print_top_words(lda, tf_feature_names, n_top_words)
    #feature_list = sum(feature, [])
    tf_fs_df = tf_df.ix[:, feature_list]
    
    df_columns = pd.Series(tf_feature_names)[feature_list]
    tf_fs_df.columns = df_columns
    le = preprocessing.LabelEncoder()  # 评分到0到n_class-1
    tf_fs_df['label'] = le.fit_transform(category_list)
    tf_fs_df.to_csv('d:/文本分类/lda_data.csv', index = False)
   # y, y_pre, nolabel_feature = Chi_square_test(tfidf_df_1)

    # =======================列表，数据，本文本地化==============
#==============================================================================
#     write_to_txt('D:\文本分类\word_list_.txt', str(words_list))
#     write_to_txt('D:\文本分类\\filename_list_.txt', str(filename_list))
#     write_to_txt('D:\文本分类\category_list_.txt', str(category_list))
#     write_to_txt('D:\文本分类\\features_list_.txt', str(nolabel_feature))
# 
#     f1 = open('D:\pickle\wordlist.txt', 'wb')
#     pickle.dump(words_list, f1)
#     f1.close()
#     f2 = open('D:\pickle\\filenamelist.txt', 'wb')
#     pickle.dump(filename_list, f2)
#     f2.close()
#     f3 = open('D:\pickle\categorylist.txt', 'wb')
#     pickle.dump(category_list, f3)
#     f3.close()
#     f4 = open('D:\pickle\\feature.txt', 'wb')
#     pickle.dump(nolabel_feature, f4)
#     f4.close()
#==============================================================================
