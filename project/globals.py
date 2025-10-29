from collections import OrderedDict
import torch
# from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self
import gradio as gr


mixed_reason_models=["qwen3-max","qwen-plus","qwen-long","qwen-flash","qwen3-32b","qwen3-235b-a22b",
                     "glm-4.5","glm-4.5-air","glm-4.5-x","glm-4.5-airx","glm-4.5-flash"] #可以切换到深度推理模式的模型列表
only_reason_models=["qwq-plus","qwq-32b","glm-z1-air","glm-z1-airx","glm-z1-flash","deepseek-reasoner","SPARK-X1"] #只能用于深度推理模式的模型列表
ChatAgents={} #全局变量，为每个用户储存智能体

llama_processes={}
timeout=3600  #10min不活跃的agent将被释放,模拟断开连接后后端保留的agent的记录的最长时间
update_time=120 #2min 更新一次agent的活跃时间  模拟agent的活跃时间，
check_time=300  #5min检查一次agent的活跃时间，释放超时agent

#update_time<check_time<timeout
chat_config=["使用知识库","联网搜索","深度思考"] #智能体模式基于react框架，集成函数调用和模型推理，将RAG和联网搜索作为工具调用的一部分
#rag_mode=["使用知识库","有记忆","联网搜索"]
abstract_dict={"短文本精读":"stuff",
               "长文本略读":"map_reduce",
               "长文本精读":"refine"

}

EMBEDDING_MODEL_DICT = {
'OPENAI':["text-embedding-ada-002","text-embedding-3-small","text-embedding-3-large"],
'ZHIPUAI':["embedding-2","embedding-3"],

"QWEN":["text-embedding-v3","text-embedding-v2","text-embedding-v1"],
"BAICHUAN":["Baichuan-Text-Embedding"],
"HuggingFaceEmbedding":[],
#"default":["all-MiniLM-L6-v2"] #chroma不加嵌入默认的一个本地的轻量级语言模型
"OllamaEmbedding":[]
}

#文心嵌入模型抛弃使用，似乎QianfanEmbeddingEndpoint已经不能使用？ 传入了key也无效，但是官方使用方式可以调用。报错
# ：no enough credential found, any one of (access_key, secret_key), (ak, sk), access_token must be provided

#spark可以生成嵌入，但是在chroma生成数据库时报错：Expected each embedding in the embeddings to be a list
DEFAULT_DB_PATH = "../data_base/knowledge_db"
DEFAULT_PERSIST_PATH = "database/vector_data_base"

SPARK_MODEL_DICT={"SPARK-LITE":"lite",
                  "SPARK-PRO":"generalv3",    #3.1
                  "SPARK-PRO-128K":"pro-128k",
                  "SPARK-MAX":"generalv3.5",
                  "SPARK-MAX-32K":"max-32k",
                  "SPARK4.0-ULTRA":"4.0Ultra",
                  "SPARK-X1":"x1"}

# OPENAI_MODEL_DICT={"GPT_FREE":["gpt-3.5-turbo","gpt-4o-mini","gpt-4.1-mini","gpt-4.1-nano"]+
#                               EMBEDDING_MODEL_DICT["OPENAI"],
#                    "NEW_API":["gpt-4","gpt-4-32k","gpt-4-turbo","gpt-4o","o1-preview","o1","o3-mini-high"]
#                    }

LLM_MODEL_DICT = {
    "OPENAI": ["gpt-3.5-turbo", "gpt-4o-mini",
               "gpt-4.1-mini","gpt-4.1-nano","gpt-5-mini","gpt-5-nano",#共计一天200次免费调用
               "gpt-4o","gpt-4.1","gpt-5" #共计一天五次免费调用，无需科学上网
               ],
    "WENXIN": [" ernie-tiny-8k","ernie-lite-8k","ernie-speed-128k",
             "ernie-4.0-8k-latest","ernie-4.0-turbo-128k","ernie-4.5-turbo-128k",
                "ernie-x1-32k","ernie-x1-turbo-32k"
              ], #2，2，2，2，4,2统一为2k；后两者是完全免费的
    "SPARK": ["SPARK-LITE","SPARK-PRO", "SPARK-PRO-128K","SPARK-MAX","SPARK-MAX-32K","SPARK4.0-ULTRA","SPARK-X1"], #4，8，8，8  统一这里默认最大为4K
    "ZHIPUAI": ["glm-4-flash-250414","glm-4-air-250414","glm-4-long","glm-4-plus","glm-zero-preview","glm-z1-air","glm-z1-airx",
                "glm-z1-flash","glm-4.5", "glm-4.5-air", "glm-4.5-x", "glm-4.5-airx", "glm-4.5-flash"], # 4k
    "KIMI":["moonshot-v1-8k","moonshot-v1-32k","moonshot-v1-128k",
            "kimi-k2-0905-preview","kimi-k2-turbo-preview","kimi-thinking-preview"], #随上下文长度变化：保证输入+输出等于上下文长度 考虑到最小版本上下文8K，这里采取较常见输出token长度的4k
    "DEEPSEEK":["deepseek-chat","deepseek-reasoner"],   #8k
    "QWEN":["qwen3-max","qwen-plus","qwen-long","qwen-flash","qwen3-omini-flash","qwq-plus","qwq-32b","qwen3-32b","qwen3-235b-a22b"], #8k, omini(2k)
    "BAICHUAN":["Baichuan-M2","Baichuan4-Turbo","Baichuan4-Air","Baichuan4","Baichuan3-Turbo","Baichuan3-Turbo-128k",
                "Baichuan2-Turbo","Baichuan2-53B"] ,#2k


    "GEMINI":["gemini-2.0-flash-lite","gemini-2.0-flash","gemini-2.5-flash-lite","gemini-2.5-flash","gemini-2.5-pro",],#4k
  #  "CLAUDE":["claude-3-5-haiku","claude-3-5-sonnet","claude-3-opus-20240229"],#4k
    "GROK":["grok-3-mini-beta","grok-3-mini-fast-beta","grok-3-beta","grok-3-fast-beta","grok-4-0709"
            ,"grok-4-fast-reasoning","grok-4-fast-non-reasoning","grok-code-fast-1"],#128k
    "HuggingFace":[],
    "Ollama":[]#本地模型列表读取当前缓存目录下的文件
}


LLM_MODEL_MAXTOKENS_DICT={
    "OPENAI":{"gpt-3.5-turbo":(4,16),"gpt-4o-mini":(16,128000), "gpt-4.1-mini":(32,1047576), "gpt-4.1-nano":(32,1047576),"gpt-5-mini":(128000,400000),"gpt-5-nano":(128000,400000),
              "gpt-4o":(16,128000),"gpt-4.1":(32,1047576),"gpt-5":(128000,400000),},
    "WENXIN":{"ernie-4.0-8k-latest":(2,6),"ernie-4.0-turbo-128k":(32,124),
              "ernie-lite-8k":(2,6),"ernie-speed-128k":(4,124),"ernie-tiny-8k":(2,6),
              "ernie-x1-32k":(16,24),"ernie-x1-turbo-32k":(16,24),"ernie-4.5-turbo-128k":(12,123)},
    "SPARK":{"SPARK-LITE":(4,8),"SPARK-PRO":(8,8),"SPARK-PRO-128K":(4,128),"SPARK-MAX":(8,8),"SPARK-MAX-32K":(8,32),"SPARK4.0-ULTRA":(8,8),"SPARK-X1":(32,32)},#4,8,4,8,8,8,32K
    "ZHIPUAI":{"glm-4-flash-250414":(16,128),"glm-4-air-250414":(16,128),"glm-4-long":(4,1024),"glm-4-plus":(4,128),"glm-zero-preview":(4,128),"glm-z1-air":(32,128),"glm-z1-airx":(30,32),
               "glm-z1-flash":(32,128),"glm-4.5":(96,128), "glm-4.5-air":(96,128), "glm-4.5-x":(96,128), "glm-4.5-airx":(96,128), "glm-4.5-flash":(96,128)},
    "KIMI":{"moonshot-v1-8k":(8,8),"moonshot-v1-32k":(32,32),"moonshot-v1-128k":(128,128),
            "kimi-k2-0905-preview":(256,256),"kimi-k2-turbo-preview":(256,256),"kimi-thinking-preview":(256,256)}, #kimi最大长度是根据上下文和输入推断的,这里给的是最大上下文的，实际肯定远小于这个值，所以调整需要注意
    "DEEPSEEK":{"deepseek-chat":(8,64),"deepseek-reasoner":(8,64)}  ,
    "QWEN":{"qwen3-max":(64,256),"qwen-plus":(32,1000000),"qwen-flash":(32,1000000),"qwen-long":(8,1000000),
            "qwq-plus":(8,128),"qwq-32b":(8,96),"qwen-omini-flash":(16,64),
            "qwen3-32b":(16,126),"qwen3-235b-a22b":(32,126)},
    "BAICHUAN":{"Baichuan-M2":(2,32),"Baichuan4-Turbo":(2,32),"Baichuan4-Air":(2,32),"Baichuan4":(2,32),"Baichuan3-Turbo":(2,32),
                "Baichuan3-Turbo-128k":(2,128),"Baichuan2-Turbo":(2,32)},
    "GEMINI":{"gemini-2.0-flash":(8,1024),"gemini-2.0-flash-lite":(8,1024),"gemini-2.5-pro":(64,1024),"gemini-2.5-flash":(64,1024),"gemini-2.5-flash-lite":(64,1024)},
#GEMINI-1.5-PRO 上下文2M，其余1M
#"CLAUDE":{"claude-3-5-haiku":(8,200),"claude-3-5-sonnet":(8,200),"claude-3-opus-20240229":(4,200)},
"GROK":{"grok-3-mini-beta":(128,128),"grok-3-mini-fast-beta":(128,128),"grok-3-beta":(128,128),"grok-3-fast-beta":(128,128),
        "grok-4-fast-reasoning":(2000000,2000000),"grok-4-fast-non-reasoning":
            (2000000,2000000),"grok-code-fast-1":(256000,256000),"grok-4-0709":(256000,256000)},#200k上下文
"HuggingFace":{},
    "Ollama":{},
    "llama_cpp":{} #本地模型列表读取当前缓存目录下的文件
}



def get_token_dict(dicts: dict[str, dict[str, tuple]]):
    for company, value in dicts.items():
        for model, max_tokens in value.items():

            output_tokens=max_tokens[0] *1024 if max_tokens[0] <100000 else max_tokens[0]
            context_window=max_tokens[1] * 1024 if max_tokens[1] < 100000 else max_tokens[1]
            dicts[company][model] =(output_tokens, context_window)


import subprocess

import gc,os,sys
import gradio as gr
def normalize_path(path: str) -> str:
    """将路径标准化，使同一路径的不同写法统一"""
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))

def unload_models(request,model_type,model_name,model_dir):
    def unload_ollama_model(model_name: str,model):
        import os
        env = os.environ.copy()
        env["OLLAMA_HOST"] = model.base_url
        subprocess.run(["ollama", "stop", model_name], env=env)
        print(f"[−] 模型 {model_name} 已停止")

    def cleanup_objects(values):
        for obj in values:
            try:
                # 处理模型对象
                if hasattr(obj, 'cpu'):
                    obj.cpu()
                # 如果有清理方法
                if hasattr(obj, 'delete_model'):
                    obj.delete_model()
                elif hasattr(obj, 'free_memory'):
                    obj.free_memory()
                if model_type == "HuggingFaceEmbedding":
                    del obj.model
                print(sys.getrefcount(obj))
                del obj

            except Exception as e:
                print(f"清理对象时出错: {e}")

    values = models_cache[model_type][model_dir].pop(model_name)
    if model_type == "Ollama" or model_type == "OllamaEmbedding":
        model=values[0]
        unload_ollama_model(model_name,model)
    elif model_type =="llama_cpp":
        model = values[0]
        model.client.close()
    else:
        cleanup_objects(values)
    chat_chains=ChatAgents[request.session_hash].chat_qa_chains
    # del Chat_QA_chain_self.ranker_model
    # Chat_QA_chain_self.ranker_model=None

    for key, value in list(chat_chains.items()):
        if key[0]==model_name:
          #  print(f"[−] 卸载聊天链 {key[0]}")
            value=chat_chains.pop(key)
          #  print(sys.getrefcount(value.llm))
          #   del value.summarizer.llm
          #   del value.summarizer
          # #  print(sys.getrefcount(value.llm))
          #   del value.llm
            del value


    del model_name, values
    gc.collect()

    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def empty_gpu_memory(request:gr.Request):


    allocated = torch.cuda.memory_allocated("cuda:0") / 1024 ** 2
 #   total_memory = get_gpu_total_memory()
    for model_type in models_cache.keys():
        for model_dir in models_cache[model_type].keys():
            for model_name, _ in list(models_cache[model_type][model_dir].items()):
                print(f"[−] 卸载模型 {model_name}")
                unload_models(request,model_type,model_name,model_dir)
    print("显存清空")
    return gr.update(value= "显存清空")



default_model_dir={"HuggingFace":"model/llm/HuggingFace","Ollama":"model/llm/Ollama","llama_cpp":"model/llm/llama_cpp",
                   "HuggingFaceEmbedding":"model/embedding/HuggingFace","OllamaEmbedding":"model/embedding/Ollama"}
for key,value in default_model_dir.items():
    default_model_dir[key]=normalize_path(value)

OLLAMA_PORT_MAP = {}  # 全局记录端口与目录映射
models_cache:dict[str,dict[str,dict]]={"HuggingFace":{default_model_dir["HuggingFace"]:{}},
              "Ollama":{default_model_dir["Ollama"]:{}},
                "llama_cpp":{default_model_dir["llama_cpp"]:{}},
                "HuggingFaceEmbedding":{default_model_dir["HuggingFaceEmbedding"]:{}},
                "OllamaEmbedding":{default_model_dir["OllamaEmbedding"]:{}}}
                                         #模型缓存，HuggingFace和Ollama的模型列表，llama_cpp本地模型列表，以及embedding模型缓存
def start_default_ollama(model_dir):
    import os
    from utils.checkPort import getValidPort
    try:
        env=os.environ.copy()
        port=getValidPort()
        env["OLLAMA_MODELS"]=model_dir
        env["OLLAMA_HOST"]=f"127.0.0.1:{port}"
        subprocess.Popen(["ollama", "serve"],env=env,
                         stdout=subprocess.DEVNULL,  # 屏蔽标准输出
                         stderr=subprocess.DEVNULL  # 屏蔽错误输出
                         )
        OLLAMA_PORT_MAP[model_dir] =port
        print(f"ollama服务启动，端口号是{port},模型路径为{model_dir}")
    except Exception as e:
        print(f"ollama服务启动失败：错误信息是{e}")



