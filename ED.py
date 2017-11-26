# -*- coding: utf-8 -*-
import difflib
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from numpy import linalg
import time
import pickle
import re
from pinyin_test import get_pinyin

'''更新数据库时必须更新temp.pkl见localization.py'''
load_start_time = time.time()
f = file("d:/share/code/bin/temp.pkl", "rb")
tf_sen_array = pickle.load(f)
feature_names = pickle.load(f)
sen_id_list = pickle.load(f)
#*********************************
tf_word_array = pickle.load(f)
feature_names_word = pickle.load(f)
word_id_list = pickle.load(f)
f.close()
load_end_time = time.time()
print "加载本地化时间：", (load_end_time - load_start_time)

def difflib_leven(str1, str2):
    flag = 0
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    print len(s.get_opcodes())
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        print('{:7} a[{}: {}] --> b[{}: {}] {} --> {}'.format(tag, i1, i2, j1, j2, str1[i1: i2], str2[j1: j2]))
        if tag == 'replace':
            flag = 1
            if i1 != 0 and max(i2-i1, j2-j1) <= 2:
                leven_cost += 1
            else:
                leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            if j1 == 0: #在首字母位置插入，加2
                leven_cost += 2
            else:
                if (j2 - j1) <= 2:
                    leven_cost += (1 + flag)
                else:
                    leven_cost += (j2 - j1)
        elif tag == 'delete':
            if (i2-i1) > 2: #只有当删掉两个字母时才累加
                leven_cost += (i2-i1)
            leven_cost += flag #如果delete前面有replace, 则累加1
    return leven_cost


def cos(vector1,vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a,b in zip(vector1,vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB==0.0:
        return None
    else:
        return dot_product / ((normA*normB)**0.5)

def csim(A, B):
    num = float(A * B.T)  # 若为行向量则 A * B.T
    denom = linalg.norm(A) * linalg.norm(B)
    cos = num / denom  # 余弦值
    sim = 0.5 + 0.5 * cos  # 归一化
    return sim

def process(asr_result):
    asr_result = unicode(asr_result, 'utf-8')
    asr_result = re.sub(u'[，。？：；！/、]', '', asr_result)
    asr_result = re.sub(u'那个', '', asr_result)  # 去停用词
    return asr_result

def similarity(asr_result):
    #while 1:
    #asr_result = input("Enter your input: ")
    #time0 = time.time()
    '''对于识别结果只有两个字的走WORD_CORRECT表，一个字的直接返回'''
    if len(asr_result) < 2 or len(set(asr_result)) < 2:
        print "test"
        return 0.0, u"无匹配".encode("utf-8")

    if len(asr_result) == 2:
        features = feature_names_word
        tf_array = tf_word_array
        id_list = word_id_list
        table = 'WORD_CORRECT'
        flag = 0
    elif len(asr_result) > 2:
        features = feature_names
        tf_array = tf_sen_array
        id_list = sen_id_list
        table = 'REQ_ANS_DATA_ZH'
        flag = 1

    # asr_result = re.sub(u'呢', 'ne', asr_result)#防止呢的拼音被轉換成‘ni’
    # asr_result = re.sub(u'还', 'hai', asr_result)#防止還的拼音被轉換成‘huan’
    # asr_result = re.sub(u'嗯', 'en', asr_result)  # 因爲嗯的拼音沒有

    asr_result_pinyin = ' '.join(get_pinyin(asr_result))
    print asr_result_pinyin
    word_list = []
    # 將語音識別出來的結果與數據庫中所有的記錄一起轉換詞頻向量
    word_list.append(asr_result_pinyin)
    #print(word_list)r

    try:
        if flag == 0:
            tf_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))

        else:
            tf_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 3))

        tf_vectorizer.fit(word_list)
    except ValueError:
        return 0.0, u"无匹配".encode("utf-8")
    else:
    #tf_df = pd.DataFrame(tf.toarray())
        print tf_vectorizer.get_feature_names()
        feature_list = tf_vectorizer.get_feature_names()

        # singleWord_list = []
        # twoGram_list = []
        # for item in feature_names:
        #     if ' ' in item:
        #         twoGram_list.append(item)
        #     else:
        #         singleWord_list.append(item)
        vector_start_time = time.time()
        vector = np.zeros(len(tf_array[0]))
        for word in feature_list:
            if word in features:
                vector[features.index(word)] += 1
            else:
                # if ' ' in word:
                #     continue
                if flag == 0 and ' ' not in word:
                    continue
                for i in range(len(features)):
                    # if ' ' in feature_names[i]:
                    #     continue
                    '''优化计算编辑距离效率'''
                    if ' ' in word:
                        if ' ' not in features[i]:
                            continue
                    else:
                        if ' ' in features[i]:
                            continue

                    leven_cost = difflib_leven(word, features[i])
                    if leven_cost <= 1:
                        vector[i] += 1


                        # if ' ' in word:
                        #     vector[i] += 50
                        # else:
                        #     vector[i] += 1

        print vector
        vector_end_time = time.time()
        print "构造向量时间：", (vector_end_time - vector_start_time)
        if vector.any() == 0:
            distance = 0.0
            changeText = u"无匹配"
            return distance, changeText.encode("utf-8")

        print len(tf_array[0])
        cos_list = []  # 保存計算出來的所有相似度
        cos_start_time = time.time()

        for i in range(len(tf_array)):
            csim_value = csim(np.matrix(tf_array[i]), np.matrix(vector))
            # print(cos(tf_array[i], tf_array[-1]))
            cos_list.append(csim_value)
        # time1 = time.time()
        # print "運行時間：",time1-time0
        cos_end_time = time.time()
        print "计算余弦相似度时间：", (cos_end_time - cos_start_time)
        #print cos_list
        max_index = np.argmax(cos_list)  # 獲得餘弦距離最大值的索引
        ID = id_list[max_index]  # 獲得餘弦距離最大值對應數據庫中的ID
        print "相似度為：", cos_list[max_index]  # 返回餘弦距離最大值的索引
        db_start_time = time.time()
        sql = "select name from "+table+" where id = '%s'" % ID
        cx = sqlite3.connect("d:/share/code/bin/correct.db")
        cu = cx.cursor()
        cu.execute(sql)
        changeText = cu.fetchone()[0]
        cu.close()
        cx.close()
        db_end_time = time.time()
        print "查询数据库时间：", (db_end_time - db_start_time)
        print "替換為：", changeText
        return cos_list[max_index], changeText.encode("utf-8")

if __name__ == "__main__":


    text = "我愿不愿不愿去见她"
    print text
    time0 = time.time()
    text = process(text)
    distance, changeText = similarity(text)
    time1 = time.time()

    print "运行时间为：", (time1-time0)
    print changeText