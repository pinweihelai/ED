import sqlite3
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
import time

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

time0 = time.time()
asr_result = "what is the price"

cx = sqlite3.connect("d:/aibeta.db")
cu=cx.cursor()
cu.execute("select condition from REQ_ANS_DATA")
word_list = []
for item in cu.fetchall():
    print(item[0])
    word_list.append(item[0])

    #tf = tf_vectorizer.fit_transform(["phone tv bag book desk"])
    #print(tf)
word_list.append(asr_result)
print(word_list)
tf_vectorizer = CountVectorizer(stop_words=None)
tf = tf_vectorizer.fit_transform(word_list)
tf_df = pd.DataFrame(tf.toarray())
tf_array = tf.toarray()
#print(tf_df)
#print(tf_df.ix[0])
print(tf_array[0])
print(len(tf_array[0]))
cos_list = []
for i in range(len(tf_df)-1):
    csim = cos(tf_array[i], tf_array[-1])
    #print(cos(tf_array[i], tf_array[-1]))
    cos_list.append(csim)
time1 = time.time()
print("運行時間：",time1-time0)
print(cos_list)
print(np.argmax(cos_list)) #返回餘弦距離最大值的索引