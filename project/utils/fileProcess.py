
import os
import sqlite3
from langchain_community.document_loaders import PyMuPDFLoader,UnstructuredWordDocumentLoader, UnstructuredFileLoader,UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_unstructured  import UnstructuredLoader
def file_loader(file, loaders):
    # 把文件转为loader，通过load返回的是数据库可以处理的Document对象
    # if isinstance(file, tempfile._TemporaryFileWrapper):
    #     file = file.name
    if not os.path.isfile(file):
        [file_loader(os.path.join(file, f), loaders) for f in os.listdir(file)]
        return

    file_type = file.split('.')[-1]
    file_name = file.split('\\')[-1]

    # if file_name in self.files[embeddings]:
    #     return
    if file_type == 'pdf':
        loaders.append((file_name, PyMuPDFLoader(file)))
    elif file_type == 'md':
        loaders.append((file_name, UnstructuredMarkdownLoader(file)))
    elif file_type == 'txt':
        loaders.append((file_name, UnstructuredFileLoader(file)))
    elif file_type == 'docx' or file_type=="doc":
        loaders.append((file_name, UnstructuredWordDocumentLoader(file)))

    else:
        raise ValueError(f"file type {file_type} not support ")


def get_docs(vectordb,files: list[str], path):
    """
    从数据库中获取文件的向量
    :param files: list[str] 文件名列表
    :param path: str 数据库路径
    :return: list[Document] 向量列表
    """
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    key = "source"
    if not files:
        return "请选择文件"
    docs= dict.fromkeys(files,[])

    # 1. 获取所有匹配的 metadata id
    for file in files:
        query = f"""
        SELECT id FROM embedding_metadata
        WHERE key=? AND string_value=?
        """
        cursor.execute(query, [key,file])
        ids= [row[0] for row in cursor.fetchall()] # list[int]
        if not ids:
            return "未查到符合要求的文件"
        placeholder = ",".join(["?"] * len(ids))
        query = f"SELECT string_value FROM embedding_metadata WHERE key=? AND id IN ({placeholder})"
        cursor.execute(query, ["chroma:document"]+ids)
        page_contents= [row[0] for row in cursor.fetchall()]  # list[int]

        docs[file]:list[Document]= [Document(page_content=page_content, metadata={"source":file,"page":i+1})
                                    for i,page_content in enumerate(page_contents)]

    conn.close()
    return docs
