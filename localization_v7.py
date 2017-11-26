# -*- coding: utf-8 -*-
import sqlite3
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import re
from pinyin_test import get_pinyin
stop_words = [u'我问你', u'我说', u'你告诉我', u'告诉我', u'我知道了', u'我知道',u'你那边',u'在那边', u'那边的', u'那边', u'我是说',
              u'我是问', u'我就问你', u'我就是问你', u'你这个',u'这个', u'你是', u'好的', u'这样', u'好吧']

def process(asr_result):
    '''去除标点符号、停用词'''
    #asr_result = unicode(asr_result, 'utf-8')
    asr_result = re.sub(u'[，。：；！/、]', '', asr_result)
    asr_result = fiter_stops(asr_result)
    asr_result = re.sub(u'[嗯啊哦的咯]', '', asr_result)  # 去停用词
    asr_result = re.sub(u'loft', u'lao fu te', asr_result)
    return asr_result

def fiter_stops(asr_result):
    for word in stop_words:
        if word in asr_result:
            print 'filter:', word
            asr_result = re.sub(word, '', asr_result)
            print 'after filter:', asr_result
    return asr_result


'''构造句向量矩阵REQ_ANS_DATA_ZH表'''
cx = sqlite3.connect("d:/share/ai2.0/bin/correct/correct.db")
cu = cx.cursor()
cu.execute("select id,name from REQ_ANS_DATA_ZH")
word_list = []
id_list = []
for item in cu.fetchall():
    # print(item[0])

    word = process(item[1])
    word_pinyin = ' '.join(get_pinyin(word))
    if word_pinyin == '':
        continue
    word_list.append(word_pinyin)
    id_list.append(item[0])  # 保存數據庫中的ID

tf_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))
tf = tf_vectorizer.fit_transform(word_list)
tf_array = tf.toarray()
print len(tf_array[0])
feature_names = tf_vectorizer.get_feature_names()

'''构造词向量矩阵word_correct表'''
cu.execute("select id, pingyin from word_correct")
word_id_list = []
word_list = []
for item in cu.fetchall():
    word_id_list.append(item[0])
    word_list.append(item[1])
tf_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
tf_word = tf_vectorizer.fit_transform(word_list)
tf_word_array = tf_word.toarray()
feature_names_word = tf_vectorizer.get_feature_names()
print feature_names_word

'''本地化'''
f = file('d:/share/ai2.0/bin/correct/temp_v7.pkl', 'wb')
pickle.dump(tf_array, f, True)
pickle.dump(feature_names, f, True)
pickle.dump(id_list, f, True)
pickle.dump(tf_word_array, f, True)
pickle.dump(feature_names_word, f, True)
pickle.dump(word_id_list, f, True)

f.close()
cu.close()
cx.close()
