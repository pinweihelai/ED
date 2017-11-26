# -*- coding: utf-8 -*-
# 作者：MebiuW
# 微博：@MebiuW
# python 版本：2.7
# 时间 2016/9/10

from pyltp import Segmentor
from pyltp import Postagger

from pyltp import NamedEntityRecognizer

segmentor1 = Segmentor()  # 初始化实例
segmentor1.load('/mnt/hgfs/share/ai2.0/bin/correct/ltp_data/cws.model')  # 加载模型

postagger = Postagger()  # 初始化实例
postagger.load('/mnt/hgfs/share/ai2.0/bin/correct/ltp_data/pos.model')  # 加载模型

recognizer = NamedEntityRecognizer()  # 初始化实例
recognizer.load('/mnt/hgfs/share/ai2.0/bin/correct/ltp_data/ner.model')  # 加载模型
# 分词
def segmentor(sentence):


    words = segmentor1.segment(sentence)  # 分词
    # 默认可以这样输出
    print '\t'.join(words)
    # 可以转换成List 输出
    words_list = list(words)
    segmentor1.release()  # 释放模型
    return words_list


def posttagger(words):

    postags = postagger.postag(words)  # 词性标注
    # for word, tag in zip(words, postags):
    #     print word + '/' + tag
    postagger.release()  # 释放模型
    return postags


# 命名实体识别
def ner(words, postags):

    netags = recognizer.recognize(words, postags)  # 命名实体识别
    for word, ntag in zip(words, netags):
        print word + '/' + ntag
    recognizer.release()  # 释放模型
    return netags




if __name__ == '__main__':
    # 测试分句子
    print('******************测试将会顺序执行：**********************')

    # 测试分词
    words = segmentor('你说的是什么东西')
    print('###############以上为分词测试###############')
    # 测试标注
    tags = posttagger(words)
    print('###############以上为词性标注测试###############')
    # 命名实体识别
    netags = ner(words, tags)
    print('###############以上为命名实体识别测试###############')
