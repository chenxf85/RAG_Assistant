import sqlite3




def init_db(db_path:str):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    #TEXT 是一种SQL数据类型，用来存储文本数据（即字符串

    c.execute('''
        CREATE TABLE IF NOT EXISTS files (
            collection TEXT, 
            filename TEXT  )
    ''')

    conn.commit()
    conn.close()

def add_filelist(collection, new_file_names:dict[str,int], db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    #
    c.executemany(
        "INSERT INTO files (collection, filename) VALUES (?, ?)",
        [(collection, name) for name in new_file_names]
    )
    #对于已经存在的文件，只需要修改length长度：在原来基础上+length

    conn.commit()
    conn.close()

def get_filelist(collection, db_path):
    conn = sqlite3.connect(db_path)

    c = conn.cursor()
    c.execute("SELECT filename FROM files WHERE collection=?", (collection,))

    files = [row[0] for row in c.fetchall()]

    conn.close()
    return files
def delete_files(collection, file_names, db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 用 IN (...) 删除多个文件名
    placeholders = ",".join("?" for _ in file_names)  # e.g. "?, ?, ?"
    sql = f"DELETE FROM files WHERE collection=? AND filename IN ({placeholders})" #删除表files中满足collection和filename在placeholders中的文件

    c.execute(sql, (collection, *file_names))  # 解包 file_names
    conn.commit()
    conn.close()

def delete_collection(collection, db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DELETE FROM files WHERE collection=?", (collection,))
    conn.commit()
    conn.close()

