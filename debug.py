
from sklearn.feature_extraction.text import CountVectorizer

word_list =["o hua jia chi na bian 260000 a duo shao qian","hua jia chi a shi duo shao qian yi fang de chi"]

tf_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))

tf = tf_vectorizer.fit_transform(word_list)
# tf_df = pd.DataFrame(tf.toarray())
tf_array = tf.toarray()
feature_names = tf_vectorizer.get_feature_names()
# print(tf_df)
# print(tf_df.ix[0])
print(tf_array[-1])
print(feature_names)