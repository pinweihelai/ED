import sqlite3


cx = sqlite3.connect("d:/aibeta.db")
cu = cx.cursor()
cu.execute("select id,condition from REQ_ANS_DATA_ZH")
dic = {}
recordsList = cu.fetchall()
for record in recordsList:
    print(record)
    dic[record[0]] = 0.0
cu.close()
cx.close()
print(dic.keys()[0])