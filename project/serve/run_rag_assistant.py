
import os

from langchain_core.tools import BaseTool, BaseToolkit
from llm.model_to_llm import model_to_llm
from embedding.call_embedding import get_embedding,download_embedding
from globals import normalize_path

#sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from user.log_in import login, logout, guest_login, loginToRegister
from user.sign_up import register
from user.MyBlocks import MyBlock
from database.create_db import KnowledgeDB
from globals import EMBEDDING_MODEL_DICT, SPARK_MODEL_DICT, DEFAULT_PERSIST_PATH, DEFAULT_DB_PATH, \
    LLM_MODEL_MAXTOKENS_DICT, ChatAgents
from prompt.prompt import default_template
from Agent.tools import  tools,register_tool
from globals import update_time, chat_config,default_model_dir
import gradio as gr
import traceback
import logging

from llm.model_to_llm import download_llm
from model.model_list import get_model_list
import copy
import subprocess,socket,time
from langchain_core.tools import BaseTool,Tool,StructuredTool
# 使用execjs 经常会遇到编码错误
# 解决方法
from user.MyBlocks import theme_block
import subprocess
from functools import partial
from globals import mixed_reason_models,models_cache
from globals import empty_gpu_memory,llama_processes
import types
# subprocess.Popen = partial(subprocess.Popen, encoding="utf-8")


# 需要写在 import execjs 之前
INIT_LLM = "gpt-3.5-turbo"

INIT_EMBEDDING_MODEL = "openai"

USER_PATH = "../figures/user.png"
AI_PATH = "../figures/AI.png"
LOGO_PATH = "../figures/logo.png"


# 定义一个函数，用于更新 LLM 类型下拉框的内容。

button_size = {"small": 10, "medium": 50, "large": 100}
local_llm_type=["HuggingFace","Ollama","llama_cpp"]
quantization_choices={"HuggingFace":["q4","q8","fp16"],"llama_cpp":[]}
#ollama默认拉取量化版本，不提供量化
def is_quant_match(f, quant):
    import re
    f = f.lower()
    if not f.endswith(".gguf"):
        return False
    parts = re.split(r"[-_.]", f)
    return any(p.startswith(quant) for p in parts)



def update_llm_dropdown(selected_type,model_dir,llm_model_maxtokens_dict):
    #  print(f"the selected type is {selected_type}")
    model_dir= default_model_dir.get(selected_type,None)

    get_model_list(model_dir,selected_type,llm_model_maxtokens_dict)
    models = list(llm_model_maxtokens_dict.get(selected_type, {}).keys())
    print(f"列表{models}")
    if selected_type == "QWEN":
        return gr.update(choices=models, value=models[0] if models else None), gr.update(visible=False), gr.update(
            visible=True) ,gr.update(),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False) # 将LLM下拉选项更新为：
    elif selected_type == "BAICHUAN":
        return gr.update(choices=models, value=models[0] if models else None), gr.update(visible=True), gr.update(
            visible=True) ,gr.update(),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False) # 将LLM下拉选项更新为：
    elif selected_type in local_llm_type:


        return (gr.update(choices=models, value=models[0] if models else None), gr.update(visible=False), gr.update(
            visible=False), gr.update(value=model_dir),
                gr.update(visible=True),gr.update(visible=True),gr.update(visible=True))   #将LLM下拉选项更新为：
    else:
        return gr.update(choices=models, value=models[0] if models else None), gr.update(visible=False), gr.update(
            visible=False),gr.update(),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False)      # 将LLM下拉选项更新为：

def update_embedding_dropdown(selected_type, state,embedding_dir):


    if selected_type == "SPARK" and state["username"] != "guest":
        embeddings = EMBEDDING_MODEL_DICT.get(selected_type, [])
        return (gr.update(choices=embeddings, value=embeddings[0] if embeddings else None), gr.update(
            visible=True), gr.update(visible=True), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False),gr.update(visible=False)),gr.update(visble=False)
    elif selected_type == "WENXIN" and state["username"] != "guest":
        embeddings = EMBEDDING_MODEL_DICT.get(selected_type, [])
        return (gr.update(choices=embeddings, value=embeddings[0] if embeddings else None), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                gr.update(visible=False),gr.update(visible=False)),gr.update(visible=False)
    elif selected_type in ["HuggingFaceEmbedding", "OllamaEmbedding"]:
        embedding_dir =  default_model_dir.get(selected_type,None)
        embeddings= get_model_list(embedding_dir, selected_type)
        # 本地embedding模型，显示本地模型配置
        return gr.update(choices=embeddings, value=embeddings[0] if embeddings else None), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                gr.update(visible=True,value=False),gr.update(visible=True),gr.update(visible=False,value=embedding_dir)
    else:
        embeddings = EMBEDDING_MODEL_DICT.get(selected_type, [])
        return gr.update(choices=embeddings, value=embeddings[0] if embeddings else None), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),gr.update(visible=False),\
            gr.update(visible=False)


def update_file(embedding_type,embedding,embedding_dir, knowledgeDB: KnowledgeDB):
    if embedding in knowledgeDB.files:
        print(knowledgeDB.files[embedding])
        knowledgeDB.init_doc_lists(embedding_type,embedding,embedding_dir)
        return gr.update(choices=knowledgeDB.files[embedding]), gr.update(),gr.update(value=knowledgeDB)
    else:
        return gr.update(choices=[]), gr.update(),gr.update()
    # 返回一个更新后的下拉框对象，包含所选类型的模型列表和默认值（第一个模型）。



def update_llm_config(llm_type,llm,llm_maxtokens,mode,chat_mode,model_dir):
    """
    该函数用于根据 LLM 类型和模型名称更新最大 token 数。

    参数:
    llm_type: LLM 类型。
    llm_model: LLM 模型名称。

    返回:
    max_tokens: 更新后的最大 token 数。
    """
    # 根据 LLM 类型和模型名称，获取对应的最大 token 数。
    print(f"模型：{llm}")
    if not llm:
        return gr.update(),gr.update(),gr.update()
    max_tokens = llm_maxtokens[llm_type][llm][0]  #如果模型列表里没有模型，输入为None，这里指定一个值，防止报错

    if llm_type=="llama_cpp" :
        model_path = os.path.join(model_dir, llm) if llm else model_dir
        quantization_choices["llama_cpp"]=[] #每次更新前清空
        for quant in ["q4","q6","q8"]:
            candidates = [f for f in os.listdir(model_path) if is_quant_match(f, quant)]
            if candidates:
                quantization_choices["llama_cpp"].append(quant)
    update=gr.update(choices=quantization_choices[llm_type],value=quantization_choices[llm_type][0] if quantization_choices[llm_type] else None,visible=True )\
        if llm_type in ["HuggingFace","llama_cpp"] else gr.update(visible=False)
    if mode=="智能体模式":
        if llm in mixed_reason_models or "qwen3".lower() in llm.lower() :
            return gr.update(maximum=max_tokens, value=max_tokens / 2),gr.update(choices=[chat_config[-1]],visible=True),update
        else:
            if "深度思考" in chat_mode:
                chat_mode.remove("深度思考")
            return gr.update(maximum=max_tokens, value=max_tokens/2),gr.update(visible=False,value=chat_mode),update #不支持深度思考
    else:
        if llm in mixed_reason_models or "qwen3".lower() in llm.lower()  :
            return gr.update(maximum=max_tokens, value=max_tokens / 2),gr.update(choices=chat_config),update
        else:
            if "深度思考" in chat_mode:
                chat_mode.remove("深度思考")
            return gr.update(maximum=max_tokens, value=max_tokens/2),gr.update(choices=chat_config[:-1],value=chat_mode),update
            #不支持深度思考


def create_db_info(files=DEFAULT_DB_PATH, embedding_type="OPENAI", embeddings="text-embedding-ada-002",
                   DB: KnowledgeDB = None, embedding_key: str = None, embedding_base: str = None,
                   spark_app_id: str = None, spark_api_secret: str = None, wenxin_secret: str = None,
                   local_embedding_model: str = None, embedding_dir: str = None,
                   ):
    try:
        embeddings=local_embedding_model if local_embedding_model  else embeddings
        embedding_dir=normalize_path(embedding_dir)
        if embedding_type in ["OllamaEmbedding","HuggingFaceEmbedding"] and embeddings not in models_cache[embedding_type].get(embedding_dir,{}):
            return f"模型{embeddings}未加载",gr.update(),gr.update()

        DB.reset(embedding_key=embedding_key, embedding_base=embedding_base,
                 spark_app_id=spark_app_id, spark_api_secret=spark_api_secret, wenxin_secret=wenxin_secret)
        # 传递本地embedding相关参数
        return DB.create_db_info(files, embedding_type, embeddings, embedding_dir
                               ), \
               gr.update(choices=DB.files.get(embeddings, []), value=None), DB


    except Exception as e:
        error_info = traceback.format_exc()  # 将完整的错误输出为字符串
        logging.basicConfig(filename="log.txt",
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                            level=logging.ERROR)  # 设置日志级别，只有调用该函数才会将大于等于level的日志信息保存到项目根目录下的log文件中;不调用，则输出到控制台
        logging.error(f"chat_process发生错误：\n{error_info}")  # 错误输出到控制台，存在于项目根目录
        print(error_info)
        return f"文件上传知识库失败：{str(e)}", gr.update(choices=DB.files.get(embeddings, [])), DB


def delete_db(db_file, embedding_type, embedding, DB,local_embedding,embedding_dir):
    embedding = local_embedding if local_embedding else embedding
    embedding_dir=normalize_path(embedding_dir)
    if embedding_type in ["OllamaEmbedding", "HuggingFaceEmbedding"] and embedding not in models_cache[embedding_type].get(embedding_dir,{}):
        return f"模型{embedding}未加载", gr.update(), gr.update()
    return DB.del_file(db_file, embedding_type, embedding,embedding_dir), gr.update(choices=DB.files.get(embedding, []),
                                                                      value=None), DB


def update_db(files, new_files, embedding_type, embedding, DB,local_embedding,embedding_dir):
    embedding = local_embedding if local_embedding else embedding
    embedding_dir = normalize_path(embedding_dir)
    if embedding_type in ["OllamaEmbedding", "HuggingFaceEmbedding"] and embedding not in models_cache[embedding_type].get(embedding_dir,{}):
        return f"模型{embedding}未加载", gr.update(), gr.update()
    return DB.update_file(files, new_files, embedding_type, embedding,embedding_dir), gr.update(choices=DB.files.get(embedding, []),
                                                                                  value=None), DB


def search_db(file, embedding,local_embedding,DB):
    embedding=local_embedding if local_embedding else embedding
    search_file = [aim_file for aim_file in DB.files.get(embedding,[]) if aim_file.startswith(file)]  # 前缀匹配搜索
    if not search_file:
        return "检索失败！", gr.update(value=None)
    else:
        return "检索成功！", gr.update(value=search_file)


def update_embedding_key(embedding_key, DB):
    DB.embedding_key = embedding_key
    return DB


def update_spark_app_id(spark_app_id, DB):
    DB.spark_app_id = spark_app_id
    return DB


def update_spark_api_secret(spark_api_secret, DB):
    DB.spark_api_secret = spark_api_secret
    return DB


def update_wenxin_secret(wenxin_secret, DB):
    DB.wenxin_secret = wenxin_secret
    return DB


def choose_all(DB, embedding):
    return gr.update(value=DB.files[embedding])


def cancel_all():
    return gr.update(value=[])


# def update_template(template_type, template, state):
#     ChatAgents[state["session_id"]].prompts[template_type] = template

def update_template(template, state):
    ChatAgents[state["session_id"]].prompts = template


# def show_template(template_type, state):
#     return gr.update(value=ChatAgents[state["session_id"]].prompts[template_type])

def update_mode(request:gr.Request,mode,llm,tool_choose):
    tool_list=ChatAgents[request.session_hash].tools
    tool_choose=[] if tool_choose is None else tool_choose
    if mode=="聊天模式":
        tool_list.pop("RagTool",None) #聊天模式是否RAG用户指定 ，不作为工具
        if tool_choose and  "RagTool" in tool_choose:
            tool_choose.remove("RagTool")
        if llm in mixed_reason_models or llm.startswith("qwen3"):
            return (gr.update(visible=True,choices=chat_config),gr.update(visible=False,value=None),gr.update(visible=True),
                    gr.update(choices=tool_list,visible=True,value=tool_choose),gr.update(visible=False))
        else:
            return gr.update(visible=True, choices=chat_config[:-1]), gr.update(visible=False, value=None), gr.update(
                visible=True), gr.update(choices=tool_list, visible=True,value=tool_choose), gr.update(visible=False)
    elif mode=="智能体模式":
        tool_list["RagTool"]=tool_list.get("RagTool",None)

        if llm in mixed_reason_models or llm.startswith("qwen3"):
            return (gr.update(visible=True,choices=[chat_config[-1]],value=None),gr.update(visible=False,value=None),
                    gr.update(visible=True),gr.update(choices=tool_list,visible=True),gr.update(visible=True))
        else:
            return gr.update(visible=False,value=None), gr.update(visible=False,
                                                            value=None), gr.update(
                visible=True), gr.update(choices=tool_list, visible=True), gr.update(visible=True)
    elif mode=="指定模式":
        return gr.update(visible=False,value=None),gr.update(visible=True,value="文本摘要"),gr.update(visible=False),gr.update(visible=False),gr.update(visible=False)
    return None



def update_chat_mode(chat_mode: list[str],mode:str):

    if "使用知识库" in chat_mode and mode=="聊天模式" :
        return gr.update(visible=True)
    else:
        return  gr.update(
                visible=False)


def update_tool_list(code,choosed_tools,state):
    if not code:
        return gr.update()
    local_vars = {}
    session_id=state["session_id"]
    user_tools=ChatAgents[session_id].tools
    try:
        # 安全执行用户输入的函数代码
       # exec(textwrap.dedent(code) , {}, local_vars)
        exec(code, {"BaseTool":BaseTool,"BaseToolkit":BaseToolkit,"os":os,
                    "Tool":Tool,"StructuredTool":StructuredTool}, local_vars)
        #如果是自定义函数作为工具,导入包需要在函数内部
        tool_names=[] #读取的用户自定义工具

        for func_tool in local_vars.values():
            #定义工具类出错,可能是缺失API参数,此时改为使用实例定义
            if isinstance(func_tool, type):
                # 支持自定义 BaseTool 或 BaseToolkit 类，或导入 LangChain 自带工具
                flag = issubclass(func_tool, BaseTool) or issubclass(func_tool, BaseToolkit)
                if flag:
                    try:
                        tool = func_tool()  # 尝试实例化
                    except Exception as e:
                        # ❌ 如果实例化失败（如缺 api_key），仅跳过，不注册、不报错
                        print(f"[跳过未初始化的工具类] {func_tool.__name__}: {e}")
                        continue

                    # ✅ 成功实例化才注册
                    name = getattr(tool, "name", tool.__class__.__name__)
                    if name not in user_tools:
                        register_tool(tool, user_tools=user_tools)
                        tool_names.append(name)
                    else:
                        del tool
                        gr.Warning(f"注册失败，存在同名工具：{name}！")

            # ✅ 如果 func_tool 本身就是实例（带好 api_key）
            elif isinstance(func_tool, BaseTool) or isinstance(func_tool, BaseToolkit):
                name = getattr(func_tool, "name", func_tool.__class__.__name__)
                if name not in user_tools:
                    register_tool(func_tool, user_tools=user_tools)
                    tool_names.append(name)
                else:
                    gr.Warning(f"注册失败，存在同名工具：{name}！")

            elif isinstance(func_tool,types.FunctionType) or isinstance(func_tool,types.MethodType):
                ##支持python函数构造工具，但是需要有doc对函数作用介绍
                    if not func_tool.__name__ in user_tools:
                        if not func_tool.__doc__:
                            gr.Warning("注册失败，没有为工具添加描述；请在函数中，用文本字符串描述工具作用")
                        else:
                            register_tool(func_tool,name=func_tool.__name__,description=func_tool.__doc__,
                              user_tools=user_tools)
                            tool_names.append(func_tool.__name__)
                    else:
                        gr.Warning(f"注册失败：存在同名工具！")


        if not tool_names:
            gr.Error("❌ 没有检测到Tool定义，请检查格式。")
            return gr.update()
        else:
            gr.Info(f"成功定义{len(tool_names)}个工具或工具包,分别为：{', '.join(tool_names)}")
            if choosed_tools is None:
               choosed_tools=[]
            merged = list(dict.fromkeys( tool_names+choosed_tools)) #将选中的工具和刚定义的工具合并，作为选中的工具值；
            #user_tools是工具列表

            return gr.update(choices=user_tools.keys(),value=merged)


    except Exception as e:
        gr.Warning(f"❌ 注册失败：{e}")


# def update_rag_config(chat_mode):
#     if "使用知识库" in chat_mode:
#         return gr.update(visible=True)
#     else:
#         return gr.update(visible=False)

def check_legal_quantized(model_type,quantization_config):
    if model_type=="llama_cpp" and not quantization_config:#必须量化
        return False
    else:
        return True

def update_model_list(model_dir,model_type,llm_model_maxtokens_dict):
    if not model_dir:
        return gr.update()
    try:
        if  os.path.exists(model_dir):
            default_model_dir[model_type]=model_dir
            llm_model_maxtokens_dict=get_model_list(model_dir,model_type,llm_model_maxtokens_dict)
            model_list=list(llm_model_maxtokens_dict.get(model_type,[]).keys())
            model_dir=normalize_path(model_dir)
            if model_dir not in models_cache[model_type]:
                models_cache[model_type][model_dir]={}
            return gr.update(choices=model_list,value=model_list[0] if model_list else None)
        else:
            gr.Warning(f"模型路径不存在，请检查！")
            return gr.update()
    except Exception as e:
        gr.Error(f"❌ 当前路径下读取模型列表失败，错误是：{e}")

def update_embedding_list(embedding_dir,embedding_type):
    if not embedding_dir:
        return gr.update()
    try:
        if os.path.exists(embedding_dir):
            default_model_dir[embedding_type] = embedding_dir
            model_list=get_model_list(embedding_dir,embedding_type)
            embedding_dir = normalize_path(embedding_dir)
            if embedding_dir not in models_cache[embedding_type]:
                models_cache[embedding_type][embedding_dir]={}
            return gr.update(choices=model_list,value=model_list[0] if model_list else None)
        else:
            gr.Warning(f"模型路径不存在，请检查！")
            return gr.update()
    except Exception as e:
        gr.Error(f"❌ 当前路径下读取模型列表失败，错误是：{e}")
        return gr.update()
def chat_qa_chain_answer(question_input, mode,chatbot, model_type,model, model_dir,llm_model_maxtokens_dict,quantization_config,
                               embedding_type, embedding,embedding_dir,
                               temperature, top_k_llm, history_len, max_tokens, top_p,
                               top_k_query, score_threshold, fecth_k, lambda_mult, search_type, chat_mode, special_mode,
                               is_with_history,
                               llm_key, llm_base, DB, upload_files, rag_files, rag_config,web_config,  websearch_max_results, tools,state,agent_output_mode,
                               reranker_config):
    context_window=llm_model_maxtokens_dict[model_type][model][1]
    agent = ChatAgents[state["session_id"]]
    if not check_legal_quantized(model_type,quantization_config):
        yield "guff格式必须指定量化模型",agent.chat_history
        return
    model_name = model + "_" + quantization_config if model_type in ["HuggingFace", "llama_cpp"] \
        else model
    model_dir=normalize_path(model_dir)
    if model_type in ["HuggingFace","llama_cpp","Ollama"]  \
            and model_name not in  models_cache[model_type].get(model_dir,{}):

        yield f"本地模型{model}未加载!",agent.chat_history
        return



    chat_mode = {"is_rag": "使用知识库" in chat_mode,
                 "is_web_search": "联网搜索" in chat_mode,
                 "is_reasoning":"深度思考" in chat_mode
                 }
    embedding_dir = normalize_path(embedding_dir)
    if chat_mode["is_rag"] and embedding_type in ["HuggingFaceEmbedding","OllamaEmbedding"] and embedding not in  models_cache[embedding_type].get(embedding_dir,{}):
        yield f"本地嵌入模型{embedding}未加载",agent.chat_history
        return

    search_type: str = "similarity_score_threshold" if search_type == "余弦相似度" else "mmr"
    is_abstract=  (special_mode=="文本摘要" and mode=="指定模式")
    if is_abstract:
        if not upload_files and not rag_files:
            yield "错误！生成文本摘要模式下未指定文件！", agent.chat_history
            return
        elif question_input:
            yield "错误！生成文本摘要模式下不支持用户输入问题！", agent.chat_history
            return


    if mode=="聊天模式" or mode=="指定模式":
        output_stream = agent.chat_qa_chain_self_answer(question_input, chatbot, model_type, model, model_dir, context_window,quantization_config,
                                                        embedding_type, embedding,embedding_dir,
                                                        temperature, top_k_llm, history_len, max_tokens, top_p,
                                                        top_k_query, score_threshold, fecth_k, lambda_mult, search_type,
                                                        chat_mode, is_abstract, is_with_history,
                                                        llm_key, llm_base, DB, upload_files, rag_files, rag_config,web_config,websearch_max_results,tools,
                                                        reranker_config,)

    elif mode=="智能体模式":
        output_stream = agent.agent_answer(question_input, chatbot,
                                           model_type=model_type,
                                           model=model,
                                           model_dir=model_dir,
                                           embedding_dir=embedding_dir,# 只在本地模型时起作用 ?
                                           quantization_config=quantization_config,  # 只在本地模型时起作用 ?
                                           temperature=temperature,
                                           top_k_llm=top_k_llm,
                                           top_p=top_p,
                                           max_tokens=max_tokens,
                                           api_key=llm_key,
                                           api_base=llm_base,
                                           tools=tools,
                                           is_with_history=is_with_history,
                                           embedding_type=embedding_type,
                                           embedding=embedding,
                                           DB=DB,
                                           context_window=context_window,
                                           search_type=search_type,
                                           top_k_query=top_k_query,
                                           score_threshold=score_threshold,
                                           fetch_k=fecth_k,
                                           lambda_mult=lambda_mult,
                                           rag_files=rag_files,
                                           upload_files=upload_files,
                                           agent_output_mode=agent_output_mode,
                                           chat_mode=chat_mode,
                                           history_len=history_len

                                           )
    for info, chat_history in output_stream:
        yield info, chat_history

def load_llm(model_type,model_dir,model,local_llm,quantization,llm_max_tokens_dict):
    model=local_llm if local_llm else model
    model_dir=model_dir if model_dir else default_model_dir[model_type]
    model_dir=normalize_path(model_dir)
  #  model_path=os.path.join(model_dir,model)
    model_name=model+"_"+quantization if model_type in ["HuggingFace","llama_cpp" ] else model
    if model_name in models_cache[model_type].get(model_dir,{}):
        return f"模型{model}加载成功",gr.update()
    else:
        try:
            flag=True
            model_list=list(get_model_list(model_dir,model_type,llm_max_tokens_dict)[model_type].keys())
            if model not in model_list:
                flag,info=download_llm(model_type,model,model_dir)
                if not flag:
                    return gr.update(value=info), gr.update()
                model_list = list(get_model_list(model_dir, model_type, llm_max_tokens_dict)[model_type].keys())
            model_to_llm(model_type,model,model_dir,quantization)

            if flag:
                info=f"模型{model}加载成功"

            return gr.update(
                value=info), gr.update(choices=model_list, value=model if model_list else None),
        except Exception as e:
            print(e)
            return gr.update(value=f"模型加载出错，错误是：{e}"),gr.update()

def  load_embedding(embedding_type,embedding_dir,embedding,local_embedding):
    embedding=  local_embedding if local_embedding  else embedding
    embedding_dir = embedding_dir if embedding_dir else default_model_dir[embedding_type]
    embedding_dir = normalize_path(embedding_dir)
    embedding_path = os.path.join(embedding_dir,embedding)

    if embedding_type in ["OllamaEmbedding","HuggingFaceEmbedding"] and embedding in models_cache[embedding_type].get(embedding_dir, {}):
        return gr.update(value=f"嵌入模型{embedding}加载成功"),gr.update(value=embedding )
    else:
        try:
            flag=True
            model_list=get_model_list(embedding_dir,embedding_type)
            if embedding not in model_list: #下载
                flag,info = download_embedding(embedding_type,embedding,embedding_dir)
                if not flag:
                    return gr.update(value=info), gr.update()
                model_list.append(embedding)
            # print("-2")
            embedding_model,*_=get_embedding(embedding_type=embedding_type,embedding=embedding,embedding_dir=embedding_dir)

            embedding_model.embed_query("1")  # 真正的预加载，在get_embedding预加载，会导致使用Chroma 传入embeddingFunction就初始化
            #我们需要调用才初始化，降低显存占用
            models_cache[embedding_type][embedding_dir][embedding] = [embedding_model]
            if flag:
                info=f"模型{embedding}加载成功"

            # model_list=get_model_list(embedding_dir,embedding_type)
            return gr.update(value=info),gr.update(choices=model_list,value=embedding if model_list else None)


        except Exception as error:
            return gr.update(value=f"模型加载出错，错误是：{error}"),gr.update()


def clear_history(state):

    ChatAgents[state["session_id"]].clear_history()

# def keep_alive(request:gr.Request):
#     if request:
#         ChatAgents[request.session_hash].keep_alive()
def update_search_config(search_type: str):
    if search_type == "余弦相似度":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    elif search_type == "MMR算法":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

def update_model_dir(is_model_dir):
    if is_model_dir:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
def update_embedding_model_dir(is_embedding_model_dir):
    if is_embedding_model_dir:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
def start_llamafactory_webui(request:gr.Request,fine_tuning):
    #需要指定端口，避免和rag应用冲突
    # 启动 llamafactory-cli webui 命令
    def is_port_open(host, port, timeout=1.0):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except Exception:
            return False

    session_id=request.session_hash
    if fine_tuning=="启动llamaFactory微调":
        try:
            gr.Info("✅ LLaMA-Factory WebUI 正在后台启动",duration=20)
            os.environ["GRADIO_SERVER_PORT"]="7880"  # 设置环境变量，指定端口

            llama_processes[session_id]=subprocess.Popen(
                ["llamafactory-cli", "webui"],
                stdout=open("fine_tuning/llama_webui.log", "w"),
                stderr=subprocess.STDOUT
            )
            for _ in range(60): #20s内判断是否成功启动
                if is_port_open("127.0.0.1", 7880):
                    gr.Info("✅ LLaMA-Factory WebUI 已在后台启动：http://localhost:7880")
                    print("成功")
                    return gr.update(value="关闭llamaFactory微调")
                time.sleep(1)
            print("超时")
            gr.Error("❌ WebUI 启动超时，请检查日志 llama_webui.log")
            return gr.update(value="启动llamaFactory微调")
        except Exception as e:
            gr.Error(f"❌ LLaMA-Factory启动出错：{str(e)}")
            return gr.update(value="启动llamaFactory微调")
    elif fine_tuning == "关闭llamaFactory微调":
        try:
            proc = llama_processes.get(session_id)
            if proc:
                proc.terminate()  # 或：proc.send_signal(signal.SIGINT)
                proc.wait() #阻塞当前主线程，直到子线程proc退出
                gr.Info(f"✅ LLaMA-Factory WebUI 已成功关闭")
                del llama_processes[session_id]
            else:
                gr.Warning(f"⚠️  没有找到对应的 WebUI 进程")
            return gr.update(value="启动llamaFactory微调")
        except Exception as e:
            gr.Error(f"❌  关闭llamaFactory Web UI失败：{e}")
            return gr.update(value="关闭llamaFactory微调")
    return None


def Rag_assistant_block(state):
    with (gr.Column(visible=False) as demo):
        knowledgeDB = KnowledgeDB()
        DB = gr.State(knowledgeDB)  # 创建一个状态变量（为每个会话维护一个知识库），用于存储知识库对象

        llm_model_maxtokens_dict = gr.State( copy.deepcopy(LLM_MODEL_MAXTOKENS_DICT) )

        # 创建界面
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):

                theme_block(scale=2)
                gr.Row(scale=2)
            gr.Column(scale=6)          #为了使图片出现在合适的位置，用column占位
            with gr.Column(scale=2):
                gr.Image(value=LOGO_PATH, height=100, scale=1, show_label=False, show_download_button=False,
                         container=False)
                title = gr.Markdown(f"""<center><b>用户</b>：</center>    """,)

            gr.Column(scale=5)
            with gr.Column(scale=2):

                logout_button = gr.Button("退出", scale=2,  # min_width=button_size["small"],
                                          variant="huggingface")
                gr.Row(scale=2)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=800,
                    show_copy_button=True,
                    show_share_button=True,
                    avatar_images=(USER_PATH, AI_PATH),
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                        {"left": "\\[", "right": "\\]", "display": True},
                        {"left": "\\(", "right": "\\)", "display": False}
                    ]
                )

                # 创建一个文本框组件，用于输入 prompt。
                with gr.Row():
                    with gr.Column(scale=8):
                        question_input = gr.Textbox(label="用户输入", placeholder="输入你的问题...")
                        with gr.Row():
                           # 主要内容宽一些
                           with gr.Column(scale=5):
                               gr.Markdown("💡**支持python自定义函数、实例方法、和langchain工具类(BaseTool, BaseToolkit)**")
                           with gr.Column(scale=1):  # 右侧按钮占用较小空间
                                registerToolButton=gr.Button("📌注册工具", scale=1,size="sm")  # 小按钮
                        function_call = gr.Code(label="函数调用(选填)",
                                               # info="",
                                                   interactive=True,lines=5,language="python",show_line_numbers=True,autocomplete=True)

                        inside_function=gr.Dropdown(label="可选的工具调用(选填)",interactive=True,multiselect=True,
                                                   choices=list(tools.keys()))
                        info_box = gr.Textbox(label="提示信息", interactive=False)
                    with gr.Column(scale=2):
                        with gr.Row():
                            mode=gr.Radio(label="模式",choices=["聊天模式","智能体模式","指定模式"],value="聊天模式")
                        with gr.Row():
                            is_with_history = gr.Checkbox(label="使用记忆", value=False)
                        with gr.Row():
                            agent_output_mode=gr.Radio(label="输出模式",choices=["标准输出","React输出"],value="标准输出",visible=False)
                        with gr.Row():
                            chat_mode = gr.CheckboxGroup(label="聊天配置", choices=chat_config[:-1], value=None)
                            special_mode = gr.Radio(label="可选功能",choices=["文本摘要"], value=None,visible=False)


                        rag_config = gr.Radio(label="搜索配置", choices=["快速搜索", "高级搜索"], value="快速搜索",
                                              visible=False)
                        reranker_config=gr.Radio(label="重排序设置",choices=["交叉编码器重排","LLM重排"],value="LLM重排",visible=False)
                        web_config=gr.Radio(label="联网搜索引擎",choices=["duckduckgo","serperapi"],value="serperapi",visible=False)#弃用不选择搜索api默认选择duduckGo，使其不可见
                        # abstract_mode = gr.Radio(label="摘要生成方式", value="短文本精读",
                        #                          choices=["短文本精读", "长文本略读", "长文本精读"], visible=False)
                        #

                # abstract_button=gr.Button(value="生成文档摘要",variant="primary")
                with gr.Row():
                    chat_button = gr.Button("Chat", variant="primary")
                    # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                    clear = gr.ClearButton(
                        components=[chatbot], value="Clear", variant="primary")


            with gr.Column(scale=2):
                model_argument = gr.Accordion("LLM参数配置", open=True)

                with model_argument:
                    temperature = gr.Slider(0,
                                            1,
                                            value=0.01,
                                            step=0.01,
                                            label="llm temperature",
                                            interactive=True)

                    top_k_llm = gr.Slider(1,
                                          10,
                                          value=3,
                                          step=1,
                                          label="top_k",
                                          interactive=True,
                                          visible=False)
                    top_p = gr.Slider(0,
                                      1,
                                      value=0.95,
                                      step=0.01,
                                      label="top_p",
                                      interactive=True,
                                      visible=False)
                    max_tokens = gr.Slider(256,
                                           4096,  # 随模型输出能力不同而改变，具体看globals
                                           value=2048,
                                           step=256,
                                           label="模型最大输出tokens",
                                           interactive=True
                                           )
                    history_len = gr.Slider(0,
                                            10,
                                            value=3,
                                            step=1,
                                            label="history length",
                                            interactive=True)

                chat_model_select = gr.Accordion("聊天模型选择")  #

                with chat_model_select:
                    model_type = gr.Dropdown(
                        list(llm_model_maxtokens_dict.value.keys()),
                        label="companys",
                        value="OPENAI",
                        interactive=True)

                    llm = gr.Dropdown(choices=llm_model_maxtokens_dict.value["OPENAI"].keys(),
                                      label="large language model",
                                      value=INIT_LLM, #gpt3.5-turbo
                                      interactive=True)

                    llm_dir = gr.Textbox(label="模型目录", visible=False, interactive=True,lines=2,
                                         value=None)
                    local_llm = gr.Textbox(label="本地模型",
                                           info="支持模型名称\n模型不存在时，可以选择是否根据模型名称下载",
                                           value="", interactive=True)
                    quantization_config = gr.Radio(label="量化配置",choices=quantization_choices["HuggingFace"],value="fp16",visible=False,interactive=True)

                    with gr.Row():

                        is_model_dir=gr.Checkbox(label="指定本地模型目录",value=False,visible=False)
                        is_download_llm = gr.Button(value="加载模型",visible=False)
                        is_empty_cache=gr.Button(value="清空显存",visible=True
                                                 #size="sm",
                                                 #min_width=button_size["small"]
                                                 )

                    llm_key = gr.Textbox(label="llm key", value=None, type="password", visible=False)  # 只对用户可见
                    base_url = gr.Textbox(label="llm base url", value=None, visible=False)

                template = gr.Textbox(label="提示词模板", info="System Message",
                                      placeholder="使用f字符串的格式填写提示词模板",
                                      value=default_template, interactive=True, lines=15)  # 允许用户自定义提示词
              #
                fine_tuning=gr.Button(value="启动llamaFactory微调",variant="huggingface")

        with gr.Row():
            # 上传文件
            with gr.Column(scale=3):
                file = gr.File(label='请选择知识库目录', file_count='multiple',
                               file_types=['.txt', '.md', '.docx','.doc','.pdf'])
                with gr.Row():
                    init_db = gr.Button("💨添加文件")  # 初次添加文件
                    upd_db = gr.Button("更新文件")  # 更新已有文件
                    del_db = gr.Button("🗑️ 删除文件")  # 删除文件
                msg_db = gr.Textbox(label="提示信息", value=None, interactive=False, visible=True, scale=3)

            # 数据库选择和文件搜索：
            with gr.Column(scale=2):
                # 既是输出也是输出组件，可以显示当前已经存在的知识库，也可以指定检索的文件
                embedding_model_select = gr.Accordion("Embedding模型")
                with embedding_model_select:
                    embedding_type = gr.Dropdown(EMBEDDING_MODEL_DICT,
                                                 label="companys",
                                                 value="OPENAI")
                    embedding = gr.Dropdown(EMBEDDING_MODEL_DICT["OPENAI"],
                                            label="Embedding model",
                                            value="text-embedding-ada-002",
                                            interactive=True)
                    embedding_model_dir = gr.Textbox(label="embedding模型目录", visible=False, interactive=True,
                                                     value=None)
                    local_embedding_model = gr.Textbox(label="本地embedding模型",
                                                       info="支持模型名称\n模型不存在时，可以选择是否根据模型名称下载",
                                                       value="", interactive=True, visible=False)

                    with gr.Row():
                        is_download_embedding = gr.Button(value="加载模型", visible=False)
                        is_embedding_model_dir = gr.Checkbox(label="指定本地模型目录", value=False,
                                                             visible=False)
                search_file = gr.Textbox(label="文件名", value=None)
                search_button = gr.Button("查询文件")

            with gr.Column(scale=2):
                embedding_config = gr.Accordion("Embedding参数配置", open=True)
                with embedding_config:
                    top_k_query = gr.Slider(1,
                                            10,
                                            value=3,
                                            step=1,
                                            label="top_k",
                                            interactive=True)
                    score_threshold = gr.Slider(0,
                                                1,
                                                value=0.4,
                                                step=0.01,
                                                label="score threshold",
                                                interactive=True)
                    websearch_max_results=gr.Slider(1,
                                                   10,
                                                   value=3,
                                                   step=1,
                                                   label="websearch_max_results",
                                                   interactive=True,
                                                   visible=True)
                    fetch_k = gr.Slider(5,
                                        30,
                                        value=20,
                                        step=1,
                                        label="fetch_k",
                                        interactive=True,
                                        visible=False)
                    lambda_mult = gr.Slider(0,
                                            1,
                                            value=0.5,
                                            step=0.01,
                                            label="lambda_mult",
                                            visible=False,
                                            interactive=True)
                search_type = gr.Radio(choices=["余弦相似度", "MMR算法"], value="余弦相似度", label="检索方式",
                                       interactive=True)
                key_input = gr.Accordion(label="embedding模型API_KEY")
                with key_input:
                    embedding_key = gr.Textbox(label="embedding key", value=None, type="password", visible=False)
                    # 只对spark embedding可见
                    spark_app_id = gr.Textbox(label="spark app id", value=None, type="password", visible=False)
                    spark_api_secret = gr.Textbox(label="spark api secret", value=None, type="password", visible=False)
                    wenxin_secret = gr.Textbox(label="wenxin api secret", value=None, type="password", visible=False)




        with gr.Row():  # 文件列表

            db_file = gr.CheckboxGroup(
                choices=DB.value.files.get("text-embedding-ada-002", []),
                label="🗂️文件列表",
                interactive=True,
                scale=8
            )

        with gr.Row():  # 翻页功能
            all_button = gr.Button(value="✔全选", scale=1,
                                   #size="sm",
                                 #  min_width=button_size["small"]
                                   )
            cancel_button = gr.Button(value="❌取消",scale=1,
                                      #size="sm", min_width=button_size["small"]
                                      )
            gr.Column(scale=8)
            last_page = gr.Button(value="🔼上一页", scale=1,
                                #  size="sm",
                                 # min_width=button_size["small"]
                                  )
            page = gr.Textbox(label="页数", scale=1,
                             # min_width=button_size["small"]
                              )  #

            next_page = gr.Button(value="🔽下一页", scale=1,
                                  #size="sm",
                                  #min_width=button_size["small"]
                                  )


        model_type.change(update_llm_dropdown, inputs=[model_type,llm_dir,llm_model_maxtokens_dict] ,outputs=[llm, top_p, top_k_llm,llm_dir,
                                                                                            local_llm,is_model_dir,is_download_llm])  # 更新LLM下拉选项
        llm.change(update_llm_config, inputs=[model_type, llm,llm_model_maxtokens_dict,mode,chat_mode,llm_dir], outputs=[max_tokens,chat_mode,quantization_config])
        embedding_type.change(update_embedding_dropdown, inputs=[embedding_type, state,embedding_model_dir],
                              outputs=[embedding, spark_app_id, spark_api_secret, wenxin_secret, local_embedding_model,
                                       is_embedding_model_dir,is_download_embedding,embedding_model_dir])
        embedding.change(update_file, inputs=[embedding_type,embedding,embedding_model_dir, DB], outputs=[db_file, msg_db,DB])
        # 用户才有key显示，当key变化了储存在知识库内供API使用；游客不显示默认不会change，所以DB储存的key是缺省的None，解析env的key
        embedding_key.change(fn=update_embedding_key, inputs=[embedding_key, DB], outputs=[DB])
        template.change(fn=update_template, inputs=[template, state])

        wenxin_secret.change(fn=update_wenxin_secret, inputs=[wenxin_secret, DB], outputs=[DB])
        spark_app_id.change(fn=update_spark_app_id, inputs=[spark_app_id, DB], outputs=[DB])
        spark_api_secret.change(fn=update_spark_api_secret, inputs=[spark_api_secret, DB], outputs=[DB])

        search_type.change(update_search_config, inputs=[search_type], outputs=[top_k_query, score_threshold,
                                                                                fetch_k, lambda_mult])

        llm_dir.submit(update_model_list,inputs =[llm_dir,model_type,llm_model_maxtokens_dict],
                       outputs=[llm])
        embedding_model_dir.submit(update_embedding_list,inputs=[embedding_model_dir,embedding_type],outputs=[embedding])
        is_empty_cache.click(empty_gpu_memory,outputs=[info_box]) #清空显存时显示模型目录
        all_button.click(choose_all, inputs=[DB, embedding], outputs=[db_file])
        cancel_button.click(cancel_all, inputs=[], outputs=[db_file])
        is_model_dir.change(update_model_dir,inputs=[is_model_dir],outputs=[llm_dir])
        is_embedding_model_dir.change(update_embedding_model_dir,inputs=[is_embedding_model_dir],outputs=[embedding_model_dir])

        is_download_llm.click(load_llm,inputs=[model_type,llm_dir,llm,local_llm,quantization_config,llm_model_maxtokens_dict],outputs=[info_box,llm])
        is_download_embedding.click(load_embedding,inputs=[embedding_type,embedding_model_dir,embedding,local_embedding_model]
                                    ,outputs=[msg_db,embedding])
        mode.change(update_mode, inputs=[mode, llm,inside_function],
                    outputs=[chat_mode, special_mode, function_call, inside_function, agent_output_mode]) \
            .then(lambda mode, chat_mode, rag_config:
                  (gr.update(visible=(mode == "聊天模式" and "使用知识库" in chat_mode)),
                   gr.update(
                       visible=(mode == "聊天模式" and "使用知识库" in chat_mode and rag_config == "高级搜索"))
                   ), inputs=[mode, chat_mode, rag_config], outputs=[rag_config, reranker_config])

        chat_mode.change(lambda mode, chat_mode, rag_config:
                  (gr.update(visible=(mode == "聊天模式" and "使用知识库" in chat_mode)),
                   gr.update(
                       visible=(mode == "聊天模式" and "使用知识库" in chat_mode and rag_config == "高级搜索"))
                   ), inputs=[mode, chat_mode, rag_config], outputs=[rag_config, reranker_config])
        rag_config.change(
            lambda mode, chat_mode, rag_config: gr.update(visible=(mode == "聊天模式" and "使用知识库" in chat_mode
                                                                   and rag_config == "高级搜索")),
            inputs=[mode, chat_mode, rag_config], outputs=[reranker_config])




        init_db.click(create_db_info,  #
                      inputs=[file, embedding_type, embedding, DB,
                              embedding_key, base_url, spark_app_id, spark_api_secret, wenxin_secret,local_embedding_model,embedding_model_dir],
                      outputs=[msg_db, db_file, DB])
        fine_tuning.click(start_llamafactory_webui,inputs=[fine_tuning],outputs=[fine_tuning])#启动llamaFactory应用
        del_db.click(delete_db, inputs=[db_file, embedding_type, embedding, DB,local_embedding_model,embedding_model_dir], outputs=[msg_db, db_file, DB])
        upd_db.click(update_db, inputs=[db_file, file, embedding_type, embedding, DB,local_embedding_model,embedding_model_dir], outputs=[msg_db, db_file, DB])
        search_button.click(search_db, inputs=[search_file, embedding, local_embedding_model,DB], outputs=[msg_db, db_file])
        # 设置按钮的点击事件。当点击时，调用上面定义的 chat_qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。


        temp_msg=gr.State("")#发送消息后马上清空，后续answer输入从temp_msg，实现“发送”文本的前端效果
        registerToolButton.click(lambda fc,ins,states: update_tool_list(fc,ins,states),inputs=[function_call,inside_function,state],
                              outputs=[inside_function])
        chat_button.click(lambda msg: (msg,""),inputs=[question_input],outputs=[temp_msg,question_input]).then(lambda fc,ins,states: update_tool_list(fc,ins,states),inputs=[function_call,inside_function,state],
                              outputs=[inside_function]
                          ).then(chat_qa_chain_answer, inputs=[
            temp_msg, mode,chatbot, model_type, llm, llm_dir,llm_model_maxtokens_dict,quantization_config,
            embedding_type, embedding,embedding_model_dir,
            temperature, top_k_llm, history_len, max_tokens, top_p,
            top_k_query, score_threshold, fetch_k, lambda_mult, search_type, chat_mode, special_mode, is_with_history,
            llm_key, base_url, DB, file, db_file, rag_config, web_config,  websearch_max_results,inside_function,state,agent_output_mode,
        reranker_config],
                          outputs=[info_box, chatbot])

        question_input.submit(lambda msg: (msg,""),inputs=[question_input],outputs=[temp_msg,question_input]).then(lambda fc,ins,states: update_tool_list(fc,ins,states),
                                                                    inputs=[function_call,inside_function,state],outputs=[inside_function]).then(chat_qa_chain_answer, inputs=[
            temp_msg, mode,chatbot, model_type, llm, llm_dir,llm_model_maxtokens_dict,quantization_config,
            embedding_type, embedding,embedding_model_dir,temperature, top_k_llm, history_len, max_tokens, top_p,
            top_k_query, score_threshold, fetch_k, lambda_mult, search_type, chat_mode, special_mode, is_with_history,
            llm_key, base_url, DB, file, db_file, rag_config, web_config,  websearch_max_results,inside_function,state,agent_output_mode
        ,reranker_config],
                          outputs=[info_box, chatbot])

        # 点击后清空后端存储的聊天记录
        clear.click(clear_history, inputs=[ state])  # 不同游客的agent也是要隔离的



        gr.Markdown("""提醒：<br>    
                         1. 使用知识库检索时请先上传文件。
                         2. OpenAI API采用的是第三方服务器，可能存在无法使用的问题<br>
                         3.Gemini 和GROK API调用需要科学上网 <br>
                         4.API调用失败可能是额度用完或服务器不再维护该模型
                         """)

    return MyBlock(demo, ("logout_button", logout_button),
                   ("llm_key", llm_key),
                   ("base_url", base_url),
                   ("embedding_key", embedding_key),
                   ("spark_app_id", spark_app_id),
                   ("spark_api_secret", spark_api_secret),
                   ("title", title),
                   ("knowledgeDB", DB),
                   ("db_file", db_file),
                   ("wenxin_secret", wenxin_secret),
                   )  # 返回所有存在value的组件，在页面跳转过程之中，需要清空value


def run_rag_assistant(state, login_block: MyBlock, register_block: MyBlock, app_block: MyBlock):
    #  state=gr.State({"username":{},"logged_in":False})
    log_button2 = register_block["log_button2"]
    login_button1: gr.Button = login_block["login_button1"]
    reg_button1 = login_block["reg_button1"]
    guest_button = login_block["guest_button"]
    reg_button2 = register_block["reg_button2"]
    logout_button = app_block["logout_button"]

    # 这里代码可以优化为以block为单位更新内部组件，然后再返回新的block，而不是组件更新。
    login_button1.click(
        login,
        inputs=[login_block["login_username"], login_block["login_password"], state, app_block["knowledgeDB"]],
        outputs=[login_block["login_output"], state, login_block.block, app_block.block,
                 app_block["llm_key"], app_block["base_url"], app_block["embedding_key"], app_block["title"],
                 app_block["knowledgeDB"], app_block["db_file"]]
    )  # 登录

    log_button2.click(loginToRegister, inputs=[],
                      outputs=[register_block.block, login_block.block, login_block["login_output"]])  # 返回登录界面

    guest_button.click(guest_login, inputs=[state, app_block["knowledgeDB"]],
                       outputs=[login_block["login_output"]
                           , state, login_block.block, app_block.block, app_block["title"], app_block["knowledgeDB"],
                                app_block["db_file"]])

    reg_button1.click(loginToRegister, inputs=[],
                      outputs=[login_block.block, register_block.block, login_block["login_output"]])  # 登录跳转到注册界面
    reg_button2.click(register, inputs=[register_block["reg_username"], register_block["reg_password1"],
                                        register_block["reg_password2"]],
                      outputs=[register_block["reg_output"], register_block.block, login_block.block,
                               login_block["login_output"]])  # 注册成功后跳转到登录界面

    logout_button.click(
        logout,
        inputs=[state],
        outputs=[state, app_block.block, login_block.block, login_block["login_output"]]
    )
