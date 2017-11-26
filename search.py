# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
import sys,os

from whoosh import qparser
from whoosh.qparser import QueryParser

sys.path.append("../")
from whoosh.index import create_in,open_dir, exists_in
from whoosh.fields import *
from whoosh.query import *
from jieba.analyse import ChineseAnalyzer
import sqlite3
import time

db_path = "d:/test2.db"
index_dir = "tmp"

def getAnalyzer():
    analyzer = ChineseAnalyzer()
    return analyzer

def getConn(db_path):
    cx = sqlite3.connect(db_path)
    cu = cx.cursor()
    return cx, cu

def getSchema():
    schema = Schema(id=NUMERIC(stored=True), industry=TEXT(stored=True),
                goods=TEXT(stored=True), sex=TEXT(stored=True),
                type=TEXT(stored=True), question=TEXT(stored=True, analyzer=getAnalyzer()),
                answer=TEXT(stored=True))
    return schema

def my_docs(sql):
    cx, cu = getConn(db_path)
    cu.execute(sql)
    mydocs = cu.fetchall()
    cu.close()
    cx.close()
    return mydocs

def add_doc(writer, sql, single = False, id = None):
    if single:
        sql = "select * from questiontable where id = %d"% id
    else:
        sql = "select * from questiontable"
    for item in my_docs(sql):
        writer.add_document(
            id=item[0],
            industry=item[1],
            goods=item[2],
            sex=item[3],
            type=item[4],
            question=item[5],
            answer=item[6]
        )

def createIndex(index_dir):
    '''创建索引，只需调用一次'''
    #先删除索引再创建
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    ix = create_in(index_dir, getSchema())  # for create new index
    writer = ix.writer()
    add_doc(writer)
    writer.commit()
    writer.close()
    ix.close()


'''待定'''
# def index_my_docs(dirname, clean=False):
#   if clean:
#     createIndex(dirname)
#   else:
#     incremental_index(dirname)


# def incremental_index(dirname):
#     ix = open_dir(dirname)
#
#     # The set of all paths in the index
#     indexed_ids = set()
#     # The set of all paths we need to re-index
#     to_index = set()
#
#     with ix.searcher() as searcher:
#       writer = ix.writer()
#
#       # Loop over the stored fields in the index
#       for fields in searcher.all_stored_fields():
#         indexed_id = fields['id']
#         indexed_ids.add(indexed_id)
#
#         if not os.path.exists(indexed_path):
#           # This file was deleted since it was indexed
#           writer.delete_by_term('path', indexed_path)
#
#         else:
#           # Check if this file was changed since it
#           # was indexed
#           indexed_time = fields['time']
#           mtime = os.path.getmtime(indexed_path)
#           if mtime > indexed_time:
#             # The file has changed, delete it and add it to the list of
#             # files to reindex
#             writer.delete_by_term('path', indexed_path)
#             to_index.add(indexed_path)
#
#       # Loop over the files in the filesystem
#       # Assume we have a function that gathers the filenames of the
#       # documents to be indexed
#       for path in my_docs():
#         if path in to_index or path not in indexed_paths:
#           # This is either a file that's changed, or a new file
#           # that wasn't indexed before. So index it!
#           add_doc(writer, path)
#
#       writer.commit()

def del_doc(id):
    ix = open_dir(index_dir)
    writer = ix.writer()
    writer.delete_by_term('id', id)
    writer.commit()

def update_doc(id, answer):
    '''
    :param id: 要修改的文档id
    :param answer: 修改后的内容
    :return:
    '''
    ix = open_dir(index_dir)
    writer = ix.writer()
    writer.update_document(id = id, answer = answer)
    writer.commit()


def search(goods, querystring):
    '''
    :param goods: 产品类别
    :param querystring: 问题
    :return: 答案
    '''
    ix = open_dir("tmp")  # for read only
    with ix.searcher() as searcher:
    #     q = getQuery(goods, querystring)
    #
    #     # corrected = searcher.correct_query(q, querystring)
    #     # if corrected.query != q:
    #     #     print("Did you mean:", corrected.string.encode('utf-8'))
    #     results = searcher.search(q, limit=1)
    #
    #     if results != None:
    #         for hit in results:
    #             print hit['answer']
    #             return hit['answer']
    #     else:
    #         print "无检索结果"
    #         return None

        #***********************************
        og = qparser.OrGroup.factory(0.9)
        parser = QueryParser("question", schema=ix.schema, group=og)
        q = parser.parse(querystring)
        for i in q.all_terms():
            print i[1]
        and_list = []
        and_list.append(Term("goods", goods))
        and_list.append(q)
        results = searcher.search(And(and_list), limit=1)
        if results != None:
            for hit in results:
                print hit['answer']
                return hit['answer']




def getQuery(goods, querystring):
    '''对querystring分词，模糊查询，拼接SQL'''
    or_list = []
    #querystring = unicode(querystring, 'utf-8')
    analyzer = getAnalyzer()
    for t in analyzer(querystring):
        or_list.append(Term("question", t.text))
    print or_list
    query = Or(or_list)
    and_list = []
    and_list.append(Term("goods", goods))
    and_list.append(query)
    return And(and_list)

def spellCorrect(querystring):
    ix = open_dir("tmp")
    with ix.searcher() as searcher:
        corrector = searcher.corrector("question")
        for i in corrector.suggest(querystring, limit=1):
            print i


if __name__ == "__main__":

    # if not exists_in(index_dir):
    #     createIndex(index_dir)
    # time0 = time.time()
    # search("rouge","口洪")
    # time1 = time.time()
    # print "time:", (time1-time0)
    search("rouge","口红效果")