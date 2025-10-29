import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from embedding.call_embedding import get_embedding

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from utils.fileProcess import file_loader
from langchain_core.documents import Document

from collections import defaultdict
from globals import EMBEDDING_MODEL_DICT,DEFAULT_DB_PATH, DEFAULT_PERSIST_PATH, SPARK_MODEL_DICT
# 首先实现基本配置

from globals import models_cache
from database.gen_files_list import init_db, add_filelist, get_filelist, delete_files, delete_collection

from langchain_chroma import Chroma

import sqlite3

print(os.getcwd())

import base64
import re
def is_valid_collection_name(name: str) -> bool:
    """
    判断字符串是否是合法的 Chroma collection 名称。
    """
    if not isinstance(name, str) or len(name) == 0 or len(name) > 63:
        return False
    # 允许的字符：字母、数字、下划线、短横线、点
    pattern = re.compile(r'^[A-Za-z0-9_.-]+$')
    return bool(pattern.fullmatch(name))

class KnowledgeDB(object):
    # 将create_db和get_vectordb合并为一个类
    # 数据库类，该类用于管理知识库，包括获取文件目录下的所有文件，加载文件；
    # 进行文本切割，创建持久化知识库。
    # 可视化一个知识库列表
    # 可以选择加载已有知识库，或者重新创建新的知识库。
    # 为不同的嵌入模型维护不同的数据库（不同的embedding，相同的文本也会有不同的输出和可能不同的输出维度）
    # 用户可以在后端持久化数据库，并且可以加载用户自身已有数据库，所以无需加载数据库到内存（变量），游客无法在后端保存数据库，只能使用当前页面缓存的数据库。
    # 为每一个用户维持一个数据库储存当前向量化的信息，这是一个跟随用户会话状态而 变化的变量，所以需要state化；并且由于用户名字和数据库的embedding类型不一样，需要维护多个变量，所以把这些维护成一个类，而state的变量必须是可深拷贝的对象
    def __init__(self, username: str = "guest", embedding_key: str = None, embedding_base: str = None,
                 spark_app_id: str = None, spark_api_secret: str = None, wenxin_secret: str = None):

        #   self.vectordbs:dict[str,list[str]] = {}        #加载不同embedding_type(str)数据库，按collection_name加载Chroma对象的名称（embedding名）；
        # 因为gr.State()创建的会话变量，必须是可以deep_copy，Chroma对象无法深拷贝，这里改为储存Chroma 对象的collection_name

        self.username = username  #
        self.DB_PATH = DEFAULT_DB_PATH  # 在app交互界面由用户选择本地文件路径
        self.persist_directory = os.path.join(DEFAULT_PERSIST_PATH,
                                              self.username)  # +"/"+user #如果是用户访问为每个用在后端数据库目录：包括不同embedding的数据库
        self.files: dict[str, list[str]] = {}  # 按照不同collection下的文件分类，这里只储存文件名，用于前端展示文件列表
        self.embedding_key = embedding_key
        self.embedding_base = embedding_base
        self.spark_app_id = spark_app_id
        self.spark_api_secret = spark_api_secret

        self.wenxin_secret = wenxin_secret
        self.file_list_path = os.path.join(self.persist_directory,
                                           "file_list.sqlite3")  # 储存所有embedding_type下不同embedding的文件列表；
        print(self.file_list_path)
        # 我们希望修改DB这个state变量的成员变量的值，同样需要 用回调函数的方法，在login中创建
        if not os.path.exists(self.persist_directory):
            # os.mkdir 和makedirs区别：
            os.mkdir(self.persist_directory)
        if not os.path.exists(self.file_list_path):
            init_db(self.file_list_path)
        self.documents: dict[str, dict[str, list[Document]]] = defaultdict(lambda: defaultdict(list))


    def reset(self, username: str = "guest", embedding_key: str = None, embedding_base: str = None,
              spark_app_id: str = None, spark_api_secret: str = None, wenxin_secret: str = None):

        #   self.vectordbs:dict[str,list[str]] = {}        #加载不同embedding_type(str)数据库，按collection_name加载Chroma对象的名称（embedding名）；
        # 因为gr.State()创建的会话变量，必须是可以deep_copy，Chroma对象无法深拷贝，这里改为储存Chroma 对象的collection_name

        self.username = username  #
        self.DB_PATH = DEFAULT_DB_PATH  # 在app交互界面由用户选择本地文件路径
        self.persist_directory = os.path.join(DEFAULT_PERSIST_PATH,
                                              self.username)  # +"/"+user #如果是用户访问为每个用在后端数据库目录：包括不同embedding的数据库
        #   self.files: dict[str, list[str]] = {}  # 按照不同collection下的文件分类，这里只储存文件名，用于前端展示文件列表
        self.embedding_key = embedding_key
        self.embedding_base = embedding_base
        self.spark_app_id = spark_app_id
        self.spark_api_secret = spark_api_secret
        self.wenxin_secret = wenxin_secret
        self.file_list_path = os.path.join(self.persist_directory,
                                           "file_list.sqlite3")  # 储存所有embedding_type下不同embedding的文件列表；
        if not os.path.exists(self.persist_directory):
            # os.mkdir 和makedirs区别：
            os.mkdir(self.persist_directory)
        if not os.path.exists(self.file_list_path):
            init_db(self.file_list_path)
        self.documents: dict[str, dict[str, list[Document]]] = defaultdict(lambda: defaultdict(list))
    def get_dbs_file(self):

        # 加载数据库应该也使用指定key，因为后续将数据库用户RAG需要用到向量查询，需要embedding_key，优雅的加入:在创建chroma对象时，输入key作为collection_metadata
        # 加载时直接调用，如果需要更改则更新chroma的embedding即可
        # 获取已存在的数据库类型，根据当前目录下的子文件夹确定
        # 列出目录名，而不是文件名
        _, dir_list, _ = next(os.walk(self.persist_directory))

        print(dir_list)

        if dir_list != []:
            for embedding_type in dir_list:  # 遍历数据库
                path = os.path.join(self.persist_directory, embedding_type)
                sql_path = os.path.join(path, "chroma.sqlite3")

                collections_data: dict[str, dict[str, str]] = self.get_collections(
                    sql_path)  # collection_name:{embedding_key:   ,....}

                for collection_name, collection_metadata in collections_data.items():  # collection_name我们设置为embedding的名称，遍历数据库下的collections

                    embedding=collection_metadata.get("embedding",collection_name) #本地模型还原embedding名称

                    self.files[embedding] = get_filelist(embedding, self.file_list_path)

        self.init_doc_lists("OPENAI", "text-embedding-ada-002")

    def get_docs(self, embedding, rag_files=None):
        if not rag_files:
            doc_list = list(self.documents[embedding].values())
            final_docs = []
            for docs in doc_list:
                final_docs.extend(docs)
            return final_docs
        else:
            docs = []
            for file in rag_files:
                if file in self.documents[embedding]:
                    docs.extend(self.documents[embedding][file])
            return docs

    def init_doc_lists(self, embedding_type, embeddings,embedding_dir=None):
        print(embedding_type, embeddings)

        if not self.documents.get(embeddings, False):
           # 初始化
            vectordb = self.load_knowledge_db(embedding_type, embeddings,embedding_dir=embedding_dir)
            get_result = vectordb.get()

            for id_, content, metadata in zip(get_result["ids"], get_result["documents"],
                                              get_result["metadatas"]):
                self.documents[embeddings][metadata["source"]].append(
                    Document(page_content=content, metadata=metadata, id=id_))

    def get_collections(self, path) -> dict[str, dict[str, str]]:

        # 连接数据库
        conn = sqlite3.connect(path)
        cursor = conn.cursor()  #

        # 获取所有 collection 的名字和metadata
        cursor.execute("SELECT id, name FROM collections")
        collections = cursor.fetchall()  #
        collections_data = {}
        for id, name in collections:
            # collection_name是collection这张表变量的名字，不是collection_data
            cursor.execute(f"SELECT key,str_value FROM collection_metadata WHERE collection_id='{id}'")
            collection_metadata: list[tuple[str]] = cursor.fetchall()
            collections_data[name] = {key: str_value for key, str_value in collection_metadata}

        return collections_data

    def get_files(self, dir_path):
        file_list = []
        for filepath, dirname, filenames in os.walk(dir_path):
            for filename in filenames:
                file_list.append(os.path.join(filepath, filename))

        return file_list


    def create_db_info(self, files=DEFAULT_DB_PATH, embedding_type="OPENAI", embeddings="text-embedding-ada-002",
                       embedding_dir=None):

        if not files:
            return "添加失败，请上传文件"

        new_files, exist_files = self.add_files(files, embedding_type, embeddings, embedding_dir)
        print(self.files)
        if not exist_files:
            return "添加成功！已在数据库初始化文件"
        else:
            return f"添加成功！其中：文件{exist_files}已经存在，已和上一次记录合并"

    def add_files(self, files=DEFAULT_DB_PATH, embedding_type: str = "OPENAI", embeddings: str = None, 
                 embedding_model_dir=None):
        """
        该函数用于加载文件，切分文档，生成文档的嵌入向量，创建向量数据库。

        参数:
        file: 存放文件的路径。
        embeddings: 用于生产 Embedding 的模型

        返回:
        vectordb: 创建的数据库。

        """

        split_docs, embedding, collection_metadata, new_file_names, exist_file_names \
            = self.document_process(files, embedding_type, embeddings, embedding_model_dir)


        # for doc in split_docs:
        #     self.documents[embeddings][doc.metadata["source"]].append(doc)  # 按照文件名分类存储文档

        #  file_names=new_file_names+exist_file_names

        vectordb = self.load_knowledge_db(embedding_type, embeddings, collection_metadata,embedding_model_dir)
        vectordb.add_documents(
            documents=split_docs,
            #     ids=final_id, #这个id是embedding的id，不是doc的id，在数据库里是可以存储的。可以通过.get方法返回id；为了实现按照文件名删除，这里我们设置了id为文件名，也可以通过doc.metedata设置=embedding_metatda;
            # 允许我们将数据库保存到磁盘上，文件夹不存在会自动创建
            # 储存key，用户加载数据库时默认使用该key调用embedding，如果需要更改key，可以在加载后再修改
        )
        #self.vectordbs[embedding_type][collection_name] = vectordb  # 每次添加完文件后，collection_metadata会更新为此次调用的key

        self.files[embeddings].extend(new_file_names)  # 加载切分前的文档id,更新文件列表
        # 将文件名按照不同collection_name保存到文件中，文件名是[loader[0] for loader in loaders]

        add_filelist(embeddings, new_file_names, self.file_list_path)
        return new_file_names, exist_file_names

        # vectordb.persist() ，最新langchain删除该用法

    def document_process(self, files=DEFAULT_DB_PATH, embedding_type: str = "OPENAI", embeddings: str = None,
                          embedding_dir=None):
        if type(files) != list:
            files = [files]

        if embeddings not in self.files:
            self.files[embeddings] = []

        loaders = []

        for file in files:
            file_loader(file, loaders)
            # loader[1]是文件loader,loader[0]是文件名

        # print(len(loaders))


        final_docs = []
        new_file_names = []
        exist_file_names = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
        for loader in loaders:
            docs_id = []
            if loader is not None:
                subdocs: list[Document] = loader[1].load()
                file_name = loader[0]
                #   print(len(subdocs))
                for idx, doc in enumerate(subdocs):
                    doc.metadata["source"] = file_name  # 可选：保留 metadata，便于 where 查询
                    #   unique_id = f"{file_name}-{idx}"
                    #     doc.metadata["doc_id"] = unique_id  #嵌入id不能够相同，doc.id没有作用，embedding后不保存，应该在from_document处指定
                    docs_id.append(doc)
                if file_name not in self.files[embeddings]:
                    new_file_names.append(file_name)
                else:
                    exist_file_names.append(file_name)  # 记录新文件的名称和切分后的文档数量
                split_docs = text_splitter.split_documents(docs_id)
                self.documents[embeddings][file_name].extend(split_docs)
                final_docs.extend(split_docs)

        # 当key等值为None时，获取的是env文件的key，这里需要返回，用于储存在collection的metadata中
        embedding, embedding_key, embedding_base, spark_app_id, spark_api_secret, wenxin_secret = \
            get_embedding(embedding_type, embedding=embeddings, embedding_key=self.embedding_key,
                          spark_app_id=self.spark_app_id, embedding_base=self.embedding_base,
                          spark_api_secret=self.spark_api_secret, wenxin_secret=self.wenxin_secret,

                          embedding_dir=embedding_dir,
                          )

        # 加载数据库：文本向量化用的是text-embedding，生成的是句向量（以文本为单位）；而llm的embedding层是词嵌入，生成的是词向量。
        # 不同的embedding得到的文本向量维度和数据不同，应该维护不同的数据库；
        # 这里我们为不同type维持不同数据库，同一type不同模型维护不同collection
        # If a persist_directory is specified, the collection will be persisted there.
        # Otherwise, the data will be ephemeral in-memory.

        collection_metadata = {"embedding":embeddings,
            "embedding_key": embedding_key, "embedding_base": embedding_base,
                               "spark_app_id": spark_app_id, "spark_api_secret": spark_api_secret,
                               "wenxin_secret": wenxin_secret,
                               "hnsw:space": "cosine"}
        print(collection_metadata)
        return final_docs, embedding, collection_metadata, new_file_names, exist_file_names

    def del_file(self, files, embedding_type: str = "OPENAI", embeddings: str = None,embedding_dir=None
                 ):

        # sql_path=os.path.join(persist_directory, "chroma.sqlite3")
        # collection_metadata=self.get_collections(sql_path)
        # todo
        if files == []:
            return "删除失败，请选择文件删除"

        print(f"需要删除文件{files}")
        vectordb = self.load_knowledge_db(embedding_type, embeddings,embedding_dir=embedding_dir)
        if files == self.files[embeddings]:
            # vectordb.delete_collection() 清空数据集并不会清空embedding，只会删除collection表的内容，metadata仍存在
            print("全删")
            vectordb.delete(where={"source": {"$in": files}})  # in表示字段值在files内都符合删除条件
            delete_collection(embeddings, self.file_list_path)
            self.files[embeddings] = []
            # 清除字典某个key
            self.documents[embeddings] = defaultdict(list)  # 清空字典
            print(self.files)
            return "删除成功,数据库已清空"
        else:
            print("删某个")
            vectordb.delete(where={"source": {"$in": files}})  # in表示字段值在files内都符合删除条件
            for file in files:
                self.files[embeddings].remove(file)
                # 清除字典某个key
                self.documents[embeddings].pop(file)
            delete_files(embeddings, files, self.file_list_path)
            print(self.files)

            return f"删除成功,文件{files}被成功删除"

    def update_file(self, files, new_files, embedding_type: str = "OPENAI", embeddings: str = None,
                    embedding_model_dir=None):
        # 更新数据库文件:如果checkboxgroup选中,则用new_files更新files;
        # 否则new_files替换同名文件
        # 一：可以删除原文件再添加;二：由于替换的document的嵌入数量不一定和原来的相等，而chroma.update_documents()更新文件要求二者匹配，所以选择方法一

        if not new_files:
            return "请选择文件更新"

        file_names = [file.split("\\")[-1] for file in new_files]  # 添加文件用路径，删除文件用文件名，指定metadata["source"]
        print(f"需要更新的文件名{file_names}")
        if not files:  # 直接替换同名文件或者新增文件
            # 如果是在更新文件中需要删除同名文件，拿这时候的files可能是文件路径，因此需要提取文件名

            #  检验files的所有元素在不在self.files中
            exist_file_names = []  # 替换原文件
            new_file_names = []  # 相当于新增文件
            for file in file_names:
                if file in self.files[embeddings]:
                    exist_file_names.append(file)
                else:
                    new_file_names.append(file)
            print(f"更新重复的文件名{exist_file_names}")
            # 剔除不存在的文件

            self.del_file(exist_file_names, embedding_type, embeddings,embedding_model_dir)  # 只会删除同名文件，

            self.add_files(new_files, embedding_type, embeddings, embedding_model_dir)  # 删除后再添加，此时肯定不存在同名文件，exist_file_names要提前判断
            print(self.files)

            if not exist_file_names:
                # 等价于添加操作
                return f"由于上传的文件在数据库中不存在，已将文件添加到数据库"
            else:
                if not new_file_names:
                    return f"文件更新成功"
                else:
                    return f"文件更新成功，存在新增文件：{new_file_names}"

        else:

            if len(files) != len(file_names):
                return "更新失败！文件数量不匹配，请重新选择文件"
            # 检验file_name每个元素是否已存在self.files[embedding]
            for file in file_names:
                if file in self.files[embeddings]:
                    return "更新失败！存在同名文件，更新同名文件不应选中文件"

            self.del_file(files, embedding_type, embeddings,embedding_model_dir)
            self.add_files(new_files, embedding_type, embeddings,
                        embedding_model_dir)
            print(self.files)
            return "更新成功！！"

            # 同名文件
            # 检验新文件和旧文件数量是否匹配

    def load_knowledge_db(self, embedding_type, embeddings: str = None, colletion_metadatas=None,  embedding_dir=None):
        """
        该函数用于加载向量数据库。

        参数:
        path: 要加载的向量数据库路径。
        embeddings: 向量数据库使用的 embedding 模型。

        返回:
        vectordb: 加载的数据库。
        """
        path = os.path.join(self.persist_directory, embedding_type)
        embedding, *_ = get_embedding(embedding_type, embedding=embeddings, embedding_key=self.embedding_key,
                                      embedding_base=self.embedding_base,
                                      spark_app_id=self.spark_app_id,
                                      spark_api_secret=self.spark_api_secret,
                                      wenxin_secret=self.wenxin_secret,

                                      embedding_dir=embedding_dir,

                                      )
        collection_name = base64.urlsafe_b64encode(embeddings.encode()).decode().rstrip("=") if not \
        is_valid_collection_name(embeddings) else embeddings
        if not colletion_metadatas:
            vectordb = Chroma(
                persist_directory=path,
                collection_name=collection_name,#collection_name不允许有：
                embedding_function=embedding,

            )
        else:
            vectordb = Chroma(
                persist_directory=path,
                collection_name=collection_name,
                embedding_function=embedding,
                collection_metadata=colletion_metadatas
            )
        return vectordb

    def get_vectordb(self, embedding_type="OPENAI", embedding="text-embedding-ada-002",embedding_dir=None):
        """
        返回向量数据库对象
        输入参数：
        question：
        llm:
        vectordb:向量数据库(必要参数),一个对象
        template：提示模版（可选参数）可以自己设计一个提示模版，也有默认使用的
        embedding：可以使用zhipuai等embedding，不输入该参数则默认使用 openai embedding，注意此时api_key不要输错
        """
        if not os.path.exists(os.path.join(self.persist_directory, embedding_type)):
            raise ValueError(f"{embedding_type}目录不存在,请先创建知识库")
        else:
            vectordb = self.load_knowledge_db(embedding_type, embedding,embedding_dir=embedding_dir)
        return vectordb
