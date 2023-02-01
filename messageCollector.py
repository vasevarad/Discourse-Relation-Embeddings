import MySQLdb
import getpass
import sys

def dbConnect(db, host='127.0.0.1',charset='utf8mb4', use_unicode=True):
    dbConn=None
    try:
        dbConn=MySQLdb.connect(host=host,
                                user=getpass.getuser(),
                                db=db,
                                charset = charset,
                                use_unicode = use_unicode
                               )
    except:
        print("DB Connection Failed: %s"%(db))
        sys.exit()
    dbCursor=dbConn.cursor()
    dictCursor=dbConn.cursor(MySQLdb.cursor.DictCursor)
    return dbConn, dbCursor, dictCursor
                                
                                


def excecute():
    raise Exception("not implemented")

def createCollectionTable():
    raise Exception("not implemented")


