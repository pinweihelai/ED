# -*- coding: utf-8 -*-
import difflib
def difflib_leven(str1, str2):
    flag = 0
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    print len(s.get_opcodes())
    print (s.get_opcodes()[0])
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        print('{:7} a[{}: {}] --> b[{}: {}] {} --> {}'.format(tag, i1, i2, j1, j2, str1[i1: i2], str2[j1: j2]))
        if tag == 'replace':
            flag = 1
            if i1 != 0 and max(i2-i1, j2-j1) <= 2:
                leven_cost += 1
            else:
                leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            if j1 == 0: #在首字母位置插入，加2
                leven_cost += 2
            else:
                if (j2 - j1) <= 2:
                    leven_cost += (1 + flag)
                else:
                    leven_cost += (j2 - j1)
        elif tag == 'delete':
            if (i2-i1) > 2: #只有当删掉两个字母时才累加
                leven_cost += (i2-i1)
            leven_cost += flag #如果delete前面有replace, 则累加1
    return leven_cost

def difflib_leven(str1, str2):

    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    #print len(s.get_opcodes())
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        #print('{:7} a[{}: {}] --> b[{}: {}] {} --> {}'.format(tag, i1, i2, j1, j2, str1[i1: i2], str2[j1: j2]))
        if tag == 'replace':
            leven_cost += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            leven_cost += (j2 - j1)
        elif tag == 'delete':
            leven_cost += (i2-i1)
    return leven_cost

str1 = "li"
str2 = "wei"
print difflib_leven(str1, str2)
