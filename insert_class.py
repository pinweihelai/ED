import sqlite3


cx = sqlite3.connect("d:/aibeta.db")
cu = cx.cursor()
i=1
while i<=274:

    sql = "update unique_answer set class = %d where id = %d"%(i,i)
    print(sql)
    cu.execute(sql)
    i = i+1
cx.commit()
cu.close()
cx.close()
