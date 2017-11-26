import sqlite3
import codecs
import re
from xpinyin import Pinyin
p = Pinyin()
cx = sqlite3.connect("d:/aibeta.db")
cu=cx.cursor()
correct_reader = codecs.open("d:/house_correct.txt", 'r', encoding='utf-8')
error_reader = codecs.open("d:/house_error.txt", 'r', encoding='utf-8')
i = 0
while 1:
    correct = correct_reader.readline().strip()
    error = error_reader.readline().strip()

    correct = re.sub(r'[，。？：；！/]', '', correct)
    correct = re.sub(r'\d', '', correct)
    correct= re.sub(r'还', 'hai', correct)
    correct = re.sub(r'呢', 'ne', correct)

    error = re.sub(r'[，。？：；！/]', '', error)
    error = re.sub(r'\d', '', error)

    if not correct:
        break
    #print(line)
    correct_pinyin = p.get_pinyin(correct,' ')
    correct_pinyin = re.sub(r'嗯', 'en', correct_pinyin)

    sql = "insert into CORRECT_ERROR_PY(id,error,correct,pinyin) values(%d,'%s','%s','%s')"%(i,error, correct, correct_pinyin)

    cu.execute(sql)
    i=i+1

cx.commit()
cu.close()
cx.close()
correct_reader.close()
error_reader.close()