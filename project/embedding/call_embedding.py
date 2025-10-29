
from globals import normalize_path
from langchain_community.embeddings import SparkLLMTextEmbeddings
from langchain_openai import OpenAIEmbeddings

from llm.call_llm import parse_llm_api_key
from llm.call_llm import parse_llm_api_base
from globals import EMBEDDING_MODEL_DICT, models_cache, default_model_dir
from langchain_community.embeddings import BaichuanTextEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import subprocess, os
from embedding.HuggingFaceEmbeddings2 import HuggingFaceEmbeddings2
from utils.checkPort import getValidPort
import time
from globals import OLLAMA_PORT_MAP
from langchain_ollama import OllamaEmbeddings
from llm.model_to_llm import download_ollama_model,download_huggingface_model

def get_embedding(embedding_type:str=None,embedding: str=None, embedding_key: str=None,embedding_base:str=None,
                  spark_app_id:str=None, spark_api_secret:str=None,
                  wenxin_secret:str=None,  embedding_dir:str=None):
    if embedding_type == "HuggingFaceEmbedding":
    # 使用HuggingFace本地embedding模型
        embedding=get_huggingface_embedding(embedding, embedding_dir)
        return embedding, \
        embedding_key, embedding_base, spark_app_id, spark_api_secret, wenxin_secret

    elif embedding_type == "OllamaEmbedding":
# 使用Ollama本地embedding模型
        embedding=get_ollama_embedding(embedding, embedding_dir)

        return embedding, \
            embedding_key, embedding_base, spark_app_id, spark_api_secret, wenxin_secret

    if not embedding_key: #""和None都可以

        if embedding_type == "SPARK":
            spark_app_id, spark_api_secret, embedding_key = parse_llm_api_key(embedding_type, embedding)
            wenxin_secret=""
        elif embedding_type=="WENXIN":
            embedding_key,wenxin_secret = parse_llm_api_key(embedding_type,embedding)

            spark_app_id = ""
            spark_api_secret = ""
        else:
            embedding_key = parse_llm_api_key(embedding_type, embedding)
            spark_app_id = ""
            spark_api_secret = ""
            wenxin_secret = ""


    if not embedding_base  :
        embedding_base = parse_llm_api_base(embedding_type,embedding)


    # 因为SPARK和OPENAI只支持 一种免费的嵌入模型调用，所以这里单独写，而不传入模型名称；
    #因为建立数据库时，在元数据保存了key的信息，便于下次调用；所以这里返回key是为了create_db的add_Files，获取key信息。
    if embedding_type == "SPARK":
        return SparkLLMTextEmbeddings(spark_app_id=spark_app_id, spark_api_secret=spark_api_secret,
                                      spark_api_key=embedding_key,base_url=embedding_base),\
            embedding_key,embedding_base,spark_app_id,spark_api_secret,wenxin_secret
    elif embedding_type =="OPENAI":
        return OpenAIEmbeddings(model=embedding,openai_api_key=embedding_key,
                                openai_api_base=embedding_base),\
            embedding_key,embedding_base,spark_app_id,spark_api_secret, wenxin_secret
    elif embedding_type =="BAICHUAN":
      #只支持默认的url
        return BaichuanTextEmbeddings(
            baichuan_api_key=embedding_key,
            model_name=embedding,
        ),\
            embedding_key,embedding_base,spark_app_id,spark_api_secret,wenxin_secret
    elif embedding_type =="QWEN":
        #只支持默认url
        return  DashScopeEmbeddings(
            model=embedding, dashscope_api_key=embedding_key
        ),\
            embedding_key,embedding_base,spark_app_id,spark_api_secret,wenxin_secret

    else:
        return OpenAIEmbeddings(
            model=embedding,
            openai_api_key=embedding_key,
            openai_api_base=embedding_base),\
            embedding_key,embedding_base,spark_app_id,spark_api_secret,wenxin_secret








def download_embedding(
    embedding_type: str = "OllamaEmbedding",
    embedding: str = None,
    embedding_dir: str = None,

):

    """
    通用 Embedding 模型下载函数
    支持 ModelScope、HuggingFace、Ollama

    Args:
        embedding_type: ["ModelScope", "HuggingFace", "OllamaEmbedding"]
        embedding: 模型名称，例如 "mxbai-embed-large" 或 "iic/nlp_gte_sentence-embedding-base-zh"
        embedding_dir: 模型下载保存路径
        port: Ollama 服务端口（仅 Ollama 模型用）

    Returns:
        str: 本地模型路径
    """
    from modelscope import snapshot_download
    if embedding is None:
        raise ValueError("embedding 参数不能为空")
    if embedding_type == "HuggingFaceEmbedding":
        return download_huggingface_model(embedding,embedding_dir)

    # ------------------- Ollama -------------------
    elif embedding_type=="OllamaEmbedding":
        return download_ollama_model(embedding,embedding_dir)


def get_huggingface_embedding(model_name: str, model_dir: str = None):
    """
    获取HuggingFace本地embedding模型
    
    Args:
        model_name: 模型名称
        model_dir: 自定义模型目录
        is_download: 是否自动下载模型
        
    Returns:
        HuggingFaceEmbeddings: embedding模型实例
    """
    try:
        # 检查模型是否已缓存
        model_dir = normalize_path(model_dir)
        if model_name in models_cache["HuggingFaceEmbedding"].get(model_dir,{}):
            return models_cache["HuggingFaceEmbedding"][model_dir][model_name][0]

        # 创建embedding模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path=os.path.join(model_dir,model_name)

        #使用重新封装的HuggingFaceEmbeddings2，是因为Chroma数据库初始化必须传入embeddingFunction，
        # 直接用HuggingFaceEmbeddings会立刻加载模型，占用大量显存
        #这里使用重新封装的HuggingFaceEmbeddings2，只有在真正使用时模型才会加载
        embedding_model = HuggingFaceEmbeddings2(
            model_name=model_path,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}  # 标准化embedding向量
        )

        return embedding_model
        
    except Exception as e:
        raise ValueError(f"加载HuggingFace embedding模型失败: {str(e)}")



def get_ollama_embedding(model: str,model_dir

                       ):
    """
    启动指定目录下的 Ollama 服务并返回 Embedding 模型实例

    Args:
        model: Ollama embedding 模型名称 (例如 "nomic-embed-text")
        model_dir: 模型存放路径
        port: 起始端口号 (默认 11510)
        base_port: 用于自动分配端口的起始值
        max_port: 自动端口分配的最大范围

    Returns:
        OllamaEmbeddings 实例
    """

    env = os.environ.copy()
    model_dir=normalize_path(model_dir)
    if model in models_cache["OllamaEmbedding"].get(model_dir, {}):
        return models_cache["OllamaEmbedding"][model_dir ][model][0]
    else:
        # --- Step 1. 检查该目录是否已有记录的端口 ---
        if model_dir in OLLAMA_PORT_MAP:
            port = OLLAMA_PORT_MAP[model_dir]

        else:
            # --- Step 2. 自动分配空闲端口 ---
            port=getValidPort(port=11500,max_port=11600)
            # --- Step 3. 启动新的 Ollama 服务 ---
            print(f"🚀 启动新的 Ollama 服务（模型目录: {model_dir}, 端口: {port}）")
            env["OLLAMA_MODELS"] = model_dir
            env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
            OLLAMA_PORT_MAP[model_dir] = port
            subprocess.Popen(["ollama", "serve"], env=env,
                             stdout=subprocess.DEVNULL,  # 屏蔽标准输出
                             stderr=subprocess.DEVNULL  # 屏蔽错误输出
                             )
            time.sleep(2)  # 等待服务启动

        base_url = f"127.0.0.1:{port}"
        os.environ["OLLAMA_HOST"] = base_url  # 优先级比输入参数base_url更高
        embedding_model = OllamaEmbeddings(
            model=model,
            base_url=base_url
        )
            # 启动一个后台进程并保持运行


        return embedding_model



