from globals import  LLM_MODEL_MAXTOKENS_DICT,OLLAMA_PORT_MAP
import os
import json
import copy
import subprocess
import ollama

# def read_model_files(root,model_name: str,model_type,llm_maxtokens_dict: dict = None):
#     config_path = os.path.join(root, "config.json")
#     with open(config_path, "r", encoding="utf-8") as f:
#         config_data = json.load(f)
#     max_tokens = config_data.get("max_position_embeddings", 8192)
#     llm_maxtokens_dict[model_type][model_name] = (max_tokens, max_tokens)


import subprocess
from utils.checkPort import getValidPort


def get_ollama_list(model_dir):
    """
    使用 subprocess.Popen 获取指定端口上的 Ollama 模型列表（异步方式）

    Args:
        port: Ollama 服务端口（默认11434）

    Returns:
        list[str]: 当前可用模型名称列表
    """
    try:
        # 通过 Popen 启动 ollama list 子进程
        env=os.environ.copy()
        if model_dir in OLLAMA_PORT_MAP:
            port = OLLAMA_PORT_MAP[model_dir]
            env["OLLAMA_MODELS"] = model_dir
            env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
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

        process = subprocess.Popen(
            ["ollama", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        # 读取输出结果
        stdout, stderr = process.communicate(timeout=5)

        if process.returncode != 0:
            print(f"[警告] 端口 {port} 的 Ollama 服务连接失败: {stderr.strip()}")
            return []

        # 解析输出
        lines = stdout.strip().splitlines()
        if not lines or len(lines) < 2:
            return []

        # ollama list 输出示例:
        # MODEL             ID              SIZE    MODIFIED
        # mistral:latest    a1b2c3d4        4.1GB   2024-04-01 10:30:00
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])

        return models

    except subprocess.TimeoutExpired:
        process.kill()
        print(f"[错误] 获取 Ollama 模型列表超时（端口 {port}）")
        return []
    except FileNotFoundError:
        print("[错误] 未找到 ollama 命令，请确认 Ollama 已正确安装。")
        return []
    except Exception as e:
        print(f"[错误] 获取 Ollama 模型列表失败: {e}")
        return []


def get_model_list(model_dir: str,model_type: str = "HuggingFace",llm_maxtokens_dict: dict = None):
    '''
    读取model_dir下最后一级目录的模型名称，返回一个模型列表，
    并生成 model_list.json 缓存 max_position_embeddings 信息
    '''
    if model_type=="Ollama":
        llm_maxtokens_dict[model_type]={}
        for model in get_ollama_list(model_dir):
            llm_maxtokens_dict[model_type][model] = (8192,16384) #Ollma模型信息没有包含最大的输出和窗口长度，这里给一个默认值8192
        return llm_maxtokens_dict
    elif model_type=="HuggingFace" :
        llm_maxtokens_dict[model_type] = {}
        for root, dirs, files in os.walk(model_dir):
            config_path = os.path.join(root, "config.json")
            configuration_path=os.path.join(root, "configuration.json")
            model_name = os.path.relpath(root, model_dir).replace("\\", "/")
            if os.path.exists(config_path) :  # safetensors
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                    max_tokens = config_data.get("max_position_embeddings", 8192)
                    llm_maxtokens_dict[model_type][model_name] = (max_tokens, max_tokens)
                except Exception as e:
                    print(f"[警告] 读取 {config_path} 失败: {e}")
                    continue
            elif os.path.exists(configuration_path):  # llama.cpp
                llm_maxtokens_dict[model_type][model_name] = (4096, 8192)
        return llm_maxtokens_dict
    elif model_type=="llama_cpp":
        llm_maxtokens_dict[model_type] = {}
        for root, dirs, files in os.walk(model_dir):
            config_path = os.path.join(root, "params")
            configuration_path = os.path.join(root, "configuration.json")
            model_name = os.path.relpath(root, model_dir).replace("\\", "/")
            if os.path.exists(config_path):  # safetensors
                try:
                    with open(config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)

                    max_tokens = config_data.get("num_predict", 4096)
                    context_window=config_data.get("num_ctx",8192)

                    llm_maxtokens_dict[model_type][model_name] = (max_tokens,context_window, )
                except Exception as e:
                    print(f"[警告] 读取 {config_path} 失败: {e}")
                    continue
            elif os.path.exists(configuration_path):  # llama.cpp
                llm_maxtokens_dict[model_type][model_name] = (4096, 8192)
        return llm_maxtokens_dict
    elif model_type=="OllamaEmbedding":
        return get_ollama_list(model_dir)
    elif model_type=="HuggingFaceEmbedding":
        model_list=[]
        for root, dirs, files in os.walk(model_dir):
            config_path = os.path.join(root, "config.json")
            configuration_path = os.path.join(root, "configuration.json")
            model_name = os.path.relpath(root, model_dir).replace("\\", "/")
            if os.path.exists(config_path) and os.path.exists(configuration_path):  # safetensors
                model_list.append(model_name)

        return model_list








