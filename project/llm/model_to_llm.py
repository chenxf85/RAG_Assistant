import sys

# from trl.commands.scripts.ppo import model_name
from utils.checkPort import getValidPort

from globals import normalize_path
sys.path.append("")
from openai import OpenAIError
#导入chatopenai，ErnieBotChat,
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatTongyi,ChatLlamaCpp
from langchain_ollama import ChatOllama
from globals import OLLAMA_PORT_MAP
import socket
import time

from llm.call_llm import parse_llm_api_key
from llm.call_llm import parse_llm_api_base
from langchain_deepseek import ChatDeepSeek
from globals import LLM_MODEL_MAXTOKENS_DICT,SPARK_MODEL_DICT
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import re
import subprocess
from typing import Tuple, Dict
from modelscope import snapshot_download
import torch
import gradio as gr
from model.model_list import  get_model_list
import multiprocessing
from globals import models_cache
from transformers import BitsAndBytesConfig


def download_huggingface_model(model: str,model_dir) -> (bool,str):
    """
    下载 HuggingFace 模型

    Args:
        model_name: 要下载的模型名称
        model_dir: 本地保存的模型目录
    Returns:
        bool: 下载是否成功
    """
    try:
        # 尝试下载模型
        snapshot_download(model,cache_dir=model_dir)
        return True,f"模型{model}下载成功，保存在{model_dir}"
    except Exception as e:
        return False,f"模型{model}下载失败: {str(e)}"


def download_ollama_model(model_name: str,model_dir:str) -> (bool,str):
    """
    下载 Ollama 模型

    Args:
        model_name: 要下载的模型名称
        model_dir: 本地保存的模型目录
        
    Returns:
        bool: 下载是否成功
    """

    # def pull_and_alias_model(base_model: str, alias_name: str):
    #     # 1. 拉取原始模型

    try:
        env = os.environ.copy()

        if model_dir in OLLAMA_PORT_MAP:
            port= OLLAMA_PORT_MAP[model_dir]
            env["OLLAMA_MODELS"] = model_dir
            env["OLLAMA_HOST"] = f"http://127.0.0.1:{port}"
        else:
            port = getValidPort(port=11500, max_port=11600)
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



        process = subprocess.Popen(
        ["ollama", "pull", model_name],
            env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        bufsize=1
        )
        for line in process.stdout:
            line = line.strip()
            # 如果是进度信息就刷新当前行
            #  if "%" in line or "Pulling" in line or "Downloading" in line:
            print(f"\r{line}", end="", flush=True)  # 单行刷新
        #   else:
        # print(f"\n{line}")  # 非进度信息正常打印

        process.wait()
        print(f"\n✅ {model_name} 下载完成")

        return True,f"模型{model_name}下载成功，保存在{model_dir}"
    except Exception as e:
        return False,f"下载 Ollama 模型失败: {str(e)}"
def download_llamacpp_model(model_name: str, model_dir: str) -> (bool,str):
    try:
        # 尝试下载模型
        snapshot_download(model_name,cache_dir=model_dir)
        print(f"\n✅ {model_name} 下载完成")

        return True,f"模型{model_name}下载成功，保存在{model_dir}"
    except Exception as e:
        return False,f"下载 Ollama 模型失败: {str(e)}"

def download_llm(model_type: str, model,model_dir:str,
                 ) :
    """
    下载模型并更新模型列表

    Args:
        model_type: 模型类型
        model: 当前选择的模型
        local_llm: 本地模型名称或路径
        model_dir:本地模型保存的目录
        download_model: 是否下载模型
        llm_model_maxtokens_dict:模型最大token
    Returns:
        Tuple[Dict, Dict]: (更新后的模型列表, 选中的模型)
    """
    if model_type == "HuggingFace":
        return download_huggingface_model(model, model_dir=model_dir)
    elif model_type == "Ollama":
        return download_ollama_model(model, model_dir=model_dir)
    elif model_type == "llama_cpp":
        return download_llamacpp_model(model, model_dir=model_dir)


def get_ollama_llm(model: str,
                   model_dir: str,
                   temperature: float = 0.7,
                   top_k: int = 5,
                   top_p: float = 0.95,
                   max_tokens: int = 2048,
                   reasoning: bool = False,
                 ):
    """
    启动指定目录下的 Ollama 服务并连接模型
    """

    env = os.environ.copy()
    if model_dir in OLLAMA_PORT_MAP:
        port= OLLAMA_PORT_MAP[model_dir]
        env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    else: #目录改变
        port=getValidPort(port=11500,max_port=11600)

        # 若未启动 Ollama 服务，则启动一个新的
        OLLAMA_PORT_MAP[model_dir] = port
        env["OLLAMA_MODELS"] = model_dir
        env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
        print(f"启动新的 Ollama 服务（模型目录: {model_dir}）...")
        subprocess.Popen(["ollama", "serve"], env=env,
                         stdout=subprocess.DEVNULL,  # 屏蔽标准输出
                         stderr=subprocess.DEVNULL  # 屏蔽错误输出
                         )
        time.sleep(2)  # 等待模型启动
    model_dir=normalize_path(model_dir)
    if model not in models_cache["Ollama"].get(model_dir,{}):
        # 启动一个后台进程并保持运行
        cmd = ["ollama", "run", model]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        base_url=f"127.0.0.1:{port}"      #
        os.environ["OLLAMA_HOST"] = base_url  # 优先级比输入参数base_url更高
        llm_model = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens,
            top_k=top_k,
            top_p=top_p,
            reasoning=reasoning
        )
        models_cache["Ollama"][model_dir][model]=[llm_model]
    else:
        models_cache["Ollama"][model_dir][model][0].temperature=temperature
        models_cache["Ollama"][model_dir][model][0].num_predict=max_tokens
        models_cache["Ollama"][model_dir][model][0].top_k=top_k
        models_cache["Ollama"][model_dir][model][0].top_p=top_p
        models_cache["Ollama"][model_dir][model][0].reasoning=reasoning
        llm_model=models_cache["Ollama"][model_dir][model][0]

    return llm_model
def get_HuggingFace_llm(model_dir,model_name,model_path,quantization,
                        max_tokens,temperature,top_k,top_p):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dir=normalize_path(model_dir)
    if model_name not in models_cache["HuggingFace"].get(model_dir, {}):
        if quantization == "q4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map=device,
            )
        elif quantization == "q8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                #  llm_int8_has_fp16_weight=True  #这个会使得模型加载失败

            )
            llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map=device,
            )
        elif quantization == "fp16":
            llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,

                trust_remote_code=True,
                device_map=device,
            )
        else:
            raise ValueError(f"不支持的量化配置: {quantization}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, torch_dtype=torch.float16,
                                              device_map=device, trust_remote_code=True)
        models_cache["HuggingFace"][model_dir][model_name] = [tokenizer, llm, None, None]

    else:
        tokenizer, llm = models_cache["HuggingFace"][model_dir][model_name][:2]


    pipe = pipeline(
        "text-generation",
        model=llm,
        tokenizer=tokenizer,
        max_length=max_tokens if max_tokens else 2048,
        temperature=temperature,
        top_k=top_k if top_k else 5,
        top_p=top_p if top_p else 0.95,
        #  device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    # llm_model=HuggingFacePipeline(pipeline=pipe)
    llm_model = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe),
                                tokenizer=tokenizer)
    models_cache["HuggingFace"][model_dir][model_name][2] = pipe
    models_cache["HuggingFace"][model_dir][model_name][3] = llm_model

    return  tokenizer,llm,pipe, llm_model

def get_llamcpp_llm(model,model_dir,quantization,
                    temperature=0.0,
                    max_tokens=4096,
                    ):
    def find_gguf_file(model_path: str, quant: str) -> str:
        quant = quant.lower()

        def is_quant_match(f, quant):
            f = f.lower()
            if not f.endswith(".gguf"):
                return False
            parts = re.split(r"[-_.]", f)
            return any(p.startswith(quant) for p in parts)

        print(f"模型路径为{model_path}")
        candidates = [f for f in os.listdir(model_path) if is_quant_match(f, quant)]
        if not candidates:
            raise FileNotFoundError(f"没有找到包含量化标志 '{quant}' 的gguf文件")

        return os.path.join(model_path, candidates[0])

    model_name = model + "_" + quantization
    model_dir = normalize_path(model_dir)
    model_path = os.path.join(model_dir, model)  # guff文件夹，不是模型文件路径
    if model_name in models_cache["llama_cpp"].get(model_dir, {}):
        model_path = models_cache["llama_cpp"][model_dir][model_name][1]
        models_cache["llama_cpp"][model_dir][model_name][0].temperature=temperature
        models_cache["llama_cpp"][model_dir][model_name][0].max_tokens=max_tokens
        models_cache["llama_cpp"][model_dir][model_name][0].n_ctx=max_tokens*2

        llm_model=models_cache["llama_cpp"][model_dir][model_name][0]
        print(f"llm_model1:{sys.getrefcount(llm_model)}")
    else:
        model_path = find_gguf_file(model_path, quantization)  # 生成具体量化guff模型文件路径
        llm_model = ChatLlamaCpp(
            temperature=temperature,
            model_path=model_path,
            n_ctx=max_tokens * 2,  # 上下文长度，这里简单设置
            n_gpu_layers=-1,
            n_batch=300,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            max_tokens=max_tokens,
            n_threads=multiprocessing.cpu_count() - 1,

            #   verbose=True,
        )
        models_cache["llama_cpp"][model_dir][model_name] = [llm_model, model_path]
        print(f"llm_model2:{sys.getrefcount(llm_model)}")

    return llm_model

def model_to_llm(model_type:str=None,model:str=None,model_dir:str=None,quantization:str=None,
                 temperature:float=0.0,max_tokens:int =4096,top_k:int =None,top_p:int=None,api_key:str=None,api_base:str=None,
                 is_reasoning=False):
       # model_path = os.path.join(model_dir, model)
        if model_type=="HuggingFace":
            # 使用HuggingFace本地模型
           # download_llm(model_type,model,local_llm,model_dir,download_model,llm_models)
            model_path = os.path.join(model_dir, model)
            model_name = model+"_"+quantization  if quantization  else model

            try:
                    # 如果指定了量化配置，使用BitsAndBytesConfig加载模型
                tokenizer,llm,pipe,llm_model=get_HuggingFace_llm(model_dir,model_name,model_path,quantization,
                                              max_tokens,temperature, top_k, top_p)

                return llm_model
            except Exception as e:
                del tokenizer,llm,pipe,llm_model
                torch.cuda.empty_cache()
                raise ValueError(f"加载模型失败: {str(e)}")

        elif model_type=="Ollama":
           # is_reasoning= None if is_reasoning else False #Ollama None执行默认推理，对于Qwen3模型，默认推理是在think标签中
            try:

                llm_model=get_ollama_llm(model=model,
                        model_dir=model_dir,
                        temperature=temperature,
                top_k=top_k if top_k else 5,
                top_p=top_p if top_p else 0.95,
                max_tokens=max_tokens if max_tokens else 2048,
                reasoning=is_reasoning) #指定目录下，需要重新生成ollama实例

                return llm_model
            except Exception as e:
                del llm_model
                torch.cuda.empty_cache()
                raise ValueError(f"加载模型失败: {str(e)}")

        elif model_type=="llama_cpp":

            try:
                llm_model=get_llamcpp_llm(model,model_dir,quantization, temperature, max_tokens)
                print(f"llm_model3:{sys.getrefcount(llm_model)}")
                return llm_model
            except Exception as e:

                torch.cuda.empty_cache()
                raise ValueError(f"加载模型失败: {str(e)}")

        api_key = parse_llm_api_key(model_type, model) if not api_key else api_key

        api_base = parse_llm_api_base(model_type,model) if not api_base else api_base


        if model_type=="QWEN":
            llm = ChatDeepSeek(model=model, temperature=temperature, max_tokens=max_tokens,
                             top_p=top_p,api_key=api_key,api_base=api_base,streaming=True,
                               extra_body={"enable_thinking":is_reasoning}) #混合推理只适用于qwen3模型(商用与开源)
        elif model_type=="ZHIPUAI":
            is_reasoning="enabled"  if is_reasoning else "disabled"
            llm = ChatDeepSeek(model=model, temperature=temperature, max_tokens=max_tokens,api_key=api_key,
                             api_base=api_base,streaming=True,extra_body = {"thinking": {
                "type": is_reasoning
            }}
                               ) #混合推理仅适用于glm4.5及以上

        elif model_type=="BAICHUAN":
            llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens,top_p=top_p,
                   api_key=api_key,base_url=api_base,streaming=True) #支持top-p,k
        else:
            if model_type == "SPARK":
                model = SPARK_MODEL_DICT[model]
            llm = ChatDeepSeek(model=model, temperature=temperature, max_tokens=max_tokens, api_key=api_key,
                             api_base=api_base, streaming=True)
        # else:
        #     if model_type == "SPARK":
        #         model = SPARK_MODEL_DICT[model]
        #     llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens,api_key=api_key,
        #                      base_url=api_base,streaming=True)

        return llm

        #ChatDeepSeek在ChatOpenAI的基础上返回了推理内容，所以这里同一使用ChatDeepSeek



