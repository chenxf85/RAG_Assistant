


from openai import OpenAI
import traceback,logging
from dotenv import load_dotenv, find_dotenv

import os
from collections import deque

# 是否要用LLM chain
from globals import SPARK_MODEL_DICT, LLM_MODEL_MAXTOKENS_DICT


def parse_llm_api_key(model_type:str, model:str=None,env_file:dict=None):
    """
    通过 model 和 env_file 的来解析平台参数
    model参数只有spark才用到，因为spark不同模型的key不一样。
    """


    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ
      #  print(f"{env_file}__{env_file is None}__解析后")


    if model_type not in  LLM_MODEL_MAXTOKENS_DICT :
        raise ValueError(f"model{model_type} not support!!!")
    else:
        if model_type =="SPARK" :
            if model=="text-embedding":
                return (env_file["SPARKEMBEDDING_APP_ID"], env_file["SPARKEMBEDDING_API_SECRET"], env_file["SPARKEMBEDDING_API_KEY"])
            else:
                return env_file[model+"_API_KEY"]
        elif model_type =="WENXIN" and model not in  LLM_MODEL_MAXTOKENS_DICT["WENXIN"].keys() : # 向量模型
                return env_file["WENXIN_EMBEDDING_API_KEY"],env_file["WENXIN_SECRET_KEY"]
        else:
            return env_file[model_type + "_API_KEY"]


def parse_llm_api_base(model_type:str, model,env_file=None)->str:
    """
    通过 model 和 env_file 的来解析平台参数
    """
    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ
       # print(env_file)
    if model_type not in  LLM_MODEL_MAXTOKENS_DICT :
        raise ValueError(f"model{model_type} not support!!!")
    else:
        if model_type=="SPARK" :
            if model=="SPARK-X1":
                return env_file["SPARK_BASE_URL"]+"v2/"
            else:
                return env_file["SPARK_BASE_URL"]+"v1/"
        elif model_type=="OPENAI":
            raw_urls = env_file.get("OPENAI_BASE_URL", "")
            urls = [url.strip() for url in raw_urls.split(",") if url.strip()]
            if len(urls) == 2:  # 读取默认env文件
                if model in OPENAI_MODEL_DICT["GPT_FREE"]:
                    return urls[0]
                elif model in OPENAI_MODEL_DICT["NEW_API"]:
                    return urls[1]
            else:  # 读取用户自定义的env
                return env_file["OPENAI_BASE_URL"]

        else:
            return env_file[model_type+"_BASE_URL"]







