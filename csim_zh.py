# -*- coding: utf-8 -*
'''
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import codecs
import numpy as np
import time

#分詞版本
import jieba
jieba.add_word("好的")
punctuations = '，。？：；！/'
def make_wordlist(text):
    wordlist = []
    for item in jieba.cut(text):
        if item != ' ' and item not in punctuations:
            wordlist.append(item)
    wordlist = ' '.join(wordlist)
    return wordlist


# contents = pd.read_table("d:/house.txt", header=None, encoding='utf-8')
# print(len(contents))
wordlists = []

for i in range(len(contents)):
    content = make_wordlist(contents[0][i])
    wordlists.append(content)

print(wordlists)
'''
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import time
import re
from xpinyin import Pinyin
p = Pinyin()
#計算餘弦距離
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
def process(asr_result):
    asr_result = unicode(asr_result, 'utf-8')
    asr_result = re.sub(u'呢', 'ne', asr_result)#防止呢的拼音被轉換成‘ni’
    asr_result = re.sub(u'还', 'hai', asr_result)#防止還的拼音被轉換成‘huan’
    asr_result = re.sub(u'嗯', 'en', asr_result)  # 因爲嗯的拼音沒有
    asr_result = re.sub(u'[，。？：；！/]', '', asr_result)
    return asr_result

def similarity(asr_result):
    #while 1:
    #asr_result = input("Enter your input: ")
    #time0 = time.time()

    asr_result_pinyin = p.get_pinyin(asr_result, ' ')
    print asr_result_pinyin

    cx = sqlite3.connect("d:/aibeta.db")
    cu=cx.cursor()
    cu.execute("select id,condition from REQ_ANS_DATA_ZH")
    word_list = []
    id_list = []
    for item in cu.fetchall():
        #print(item[0])
        id_list.append(item[0])#保存數據庫中的ID
        word_list.append(item[1])

    # 將語音識別出來的結果與數據庫中所有的記錄一起轉換詞頻向量
    word_list.append(asr_result_pinyin)
    #print(word_list)r
    tf_vectorizer = CountVectorizer(stop_words=None)
    tf = tf_vectorizer.fit_transform(word_list)
    #tf_df = pd.DataFrame(tf.toarray())
    tf_array = tf.toarray()
    #print(tf_df)
    #print(tf_df.ix[0])
    print tf_array[-1]
    if tf_array[-1].any() == 0:
        distance = 0.0
        changeText = None
        return distance, changeText
    print "test"
    print len(tf_array[0])
    cos_list = []#保存計算出來的所有相似度
    for i in range(len(tf_array)-1):
        csim = cos(tf_array[i], tf_array[-1])
        #print(cos(tf_array[i], tf_array[-1]))
        cos_list.append(csim)
    #time1 = time.time()
    #print "運行時間：",time1-time0
    print cos_list
    max_index = np.argmax(cos_list)#獲得餘弦距離最大值的索引
    ID = id_list[max_index]#獲得餘弦距離最大值對應數據庫中的ID
    print "相似度為：",cos_list[max_index] #返回餘弦距離最大值的索引

    sql = "select name from REQ_ANS_DATA_ZH where id = '%s'"%ID

    cu.execute(sql)
    changeText = cu.fetchone()[0]
    cu.close()
    cx.close()
    print "替換為：", changeText
    return cos_list[max_index], changeText.encode("utf-8")

if __name__ == "__main__":
    text = "龚玥"
    text = process(text)
    distance, changeText = similarity(text)
    print distance, changeText
