# 导入必要的库
import sys
import os                # 用于操作系统相关的操作，例如读取环境变量

workdir= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取当前文件的绝对路径
os.chdir(workdir)
sys.path.append(workdir)  #project文件夹
print(f"当前工作目录: {os.getcwd()}")  # 打印当前工作目录，确认是否正确

import gradio as gr
from dotenv import load_dotenv, find_dotenv



from serve.run_rag_assistant import run_rag_assistant ,Rag_assistant_block
from user.MyBlocks import MyBlock,login_block,register_block  

from globals import EMBEDDING_MODEL_DICT,SPARK_MODEL_DICT,DEFAULT_PERSIST_PATH,DEFAULT_DB_PATH,\
    LLM_MODEL_MAXTOKENS_DICT,ChatAgents,empty_gpu_memory,models_cache,default_model_dir

from prompt.prompt import default_template
from globals import get_token_dict,start_default_ollama
 

from threading import Thread,Lock

get_token_dict(LLM_MODEL_MAXTOKENS_DICT)
start_default_ollama(default_model_dir["Ollama"])
start_default_ollama(default_model_dir["OllamaEmbedding"])

_ = load_dotenv(find_dotenv())


chat_agents_lock = Lock() #加锁防止线程对于ChatAgents访问发生冲突

def on_close(request:gr.Request): #gradio回调函数调用自动传入参数request，可以获取当前会话的唯一id
    if request:
        id= request.session_hash
        empty_gpu_memory(request)
        ChatAgents.pop(id,None)
        print(f"会话关闭，已经释放{request.session_hash}的资源")

if __name__ == "__main__":


    theme=gr.themes.Base()
    theme.set(body_background_fill="#F0FFF0",background_fill_primary=" #fdf6e3",
           #   input_background_fill="#ADD8E6"
              )
    with gr.Blocks(title="Rag_assistant",theme=theme) as Rag_assistant:
        state=gr.State({"username":"","logged_in":False,"session_id":None})

        login_block=login_block()
        register_block=register_block()
        app_block=Rag_assistant_block(state)
        run_rag_assistant(state,login_block,register_block,app_block)
     #   Thread(target=clean_inactive_agents, daemon=True).start()
    Rag_assistant.unload(on_close)
    Rag_assistant.launch(#share=True,
                         pwa=True)


