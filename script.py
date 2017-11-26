# -*- coding: utf-8 -*-
'''
將文本文件中的内容轉化為拼音插入數據庫REQ_ANS_DATA_ZH
'''
import sqlite3
import codecs
import re
from pinyin_test import get_pinyin
cx = sqlite3.connect("d:/share/ai2.0/bin/correct/correct.db")
cu=cx.cursor()
reader = codecs.open("d:/house9.txt", 'r', encoding='utf-8')
i = 1700
while 1:
    line = reader.readline().strip()
    if not line:
        break
    line = re.sub(u'[，。？：；！/、]', '', line)
    # line_p = re.sub(r'\d', '', line)
    # line_p = re.sub(r'还', 'hai', line_p)
    # line_p = re.sub(r'呢', 'ne', line_p)
    print(line)

    line_pinyin = ' '.join(get_pinyin(line))


    sql = "insert into REQ_ANS_DATA_ZH(id,name,condition) values(%d,'%s','%s')"%(i,line, line_pinyin)

    cu.execute(sql)
    i=i+1

cx.commit()
cu.close()
cx.close()