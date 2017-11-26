# -*- coding: utf-8 -*-
import sqlite3
import pickle
from sklearn.feature_extraction.text import CountVectorizer

'''构造句向量矩阵REQ_ANS_DATA_ZH表'''
cx = sqlite3.connect("d:/share/ai2.0/bin/correct/correct.db")
cu = cx.cursor()
cu.execute("select id,condition from REQ_ANS_DATA_ZH")
word_list = []
id_list = []
for item in cu.fetchall():
    # print(item[0])
    id_list.append(item[0])  # 保存數據庫中的ID
    word_list.append(item[1])
tf_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
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
tf_word = tf_vectorizer.fit_transform(word_list)
tf_word_array = tf_word.toarray()
feature_names_word = tf_vectorizer.get_feature_names()
print feature_names_word

'''本地化'''
f = file('d:/share/ai2.0/bin/correct/temp.pkl', 'wb')
pickle.dump(tf_array, f, True)
pickle.dump(feature_names, f, True)
pickle.dump(id_list, f, True)
pickle.dump(tf_word_array, f, True)
pickle.dump(feature_names_word, f, True)
pickle.dump(word_id_list, f, True)


f.close()
cu.close()
cx.close()
