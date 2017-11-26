import codecs
import jieba
def readtext(path):
    text = codecs.open(path, 'r', encoding='utf-8')
    content = text.read()
    text.close()
    return content
house = readtext('d:/house.txt')
wordlist = []
for seg in jieba.cut(house):
    print(seg)


