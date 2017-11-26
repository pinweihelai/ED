# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import os
import re

from pinyin import PinYin

pinyin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'pinyin.txt')
pin = PinYin(pinyin_path)
re_zh = re.compile('([\u4E00-\u9FA5]+)')

def get_pinyin(sentence):
    ret = []
    for s in re_zh.split(sentence):
        s = s.strip()
        if not s:
            continue
        if re_zh.match(s):
            ret += pin.get(s)
        else:
            for word in s.split():
                word = word.strip()
                if word:
                    ret.append(word)
    return ret

if __name__ == "__main__":
    print ' '.join(get_pinyin("唇膏"))