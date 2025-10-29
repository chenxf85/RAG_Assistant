
#用户登录后可以获取自己的数据库，并且选择上传自己的key，选择模型，也可以选择默认使用本地提供的key；

import sys
import os
#sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 登录逻辑处理
import gradio as gr
from user.sign_up import load_users, save_user
import traceback
import logging
from database.create_db import KnowledgeDB
from Agent.agent import Agent
from database.gen_files_list import init_db
from globals import DEFAULT_PERSIST_PATH,ChatAgents,default_model_dir,empty_gpu_memory
#导入随机函数
import random
from model.model_list import get_model_list
def login(username, password, state,knowledgeDB:KnowledgeDB,request:gr.Request):
    try :
        users = load_users() #json文件中读取用户信息
        # 检查用户名和密码是否匹配
        if username in users and users[username] == password:
            # 登录成功，修改状态
            id=request.session_hash
            state["logged_in"] = True
            state["username"] = username
            state["session_id"]=id

            # 返回提示信息、更新状态、隐藏登录界面、显示主应用界面，显示embedding_key和llm_key
            knowledgeDB.reset(username)
            knowledgeDB.get_dbs_file()
         #
            ChatAgents[id]=Agent(id) #不用username作为key，因为gr.unload的回调函数不能传入输入组件，无法通过state获得username，无法清理特定资源


            return "登录成功！", state, gr.update(visible=False), gr.update(visible=True),\
            gr.update(visible=True),gr.update(visible=True),gr.update(visible=True),\
                gr.update(value=f""" <b><center>用户：{state["username"]}</b></center> 
                    """) ,\
                knowledgeDB,gr.update(choices=knowledgeDB.files.get("text-embedding-ada-002",[]))
            #针对rag_assistant增加的，非一般性代码
        else:
            # 登录失败，保持注册和登录界面可见
            return "用户名或密码错误！", state, gr.update(visible=True), gr.update(visible=False),\
                gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(),knowledgeDB, gr.update()#针对rag_assistant增加的，非一般性代码
    except Exception as e:
        error_info = traceback.format_exc()  # 将完整的错误输出为字符串
        logging.basicConfig(filename="log.txt",
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                            level=logging.ERROR)  # 设置日志级别，只有调用该函数才会将大于等于level的日志信息保存到项目根目录下的log文件中;不调用，则输出到控制台
        logging.error(f"用户登录发生错误：\n{error_info}")  # 错误输出到log.txt, 存在于项目根目录
        print(error_info)
        return f"用户登录失败：{str(error_info)}",state,gr.update(visible=True), gr.update(visible=False),\
            gr.update(visible=False),gr.update(visible=False),gr.update(visible=False),gr.update(),knowledgeDB,gr.update(),\



# 退出登录逻辑
def logout(request:gr.Request,state):
   # ChatAgents[state["username"]].clear_all_history()
    session_id=state["session_id"]
    empty_gpu_memory(request)
    ChatAgents.pop(session_id, None)
    print(f"用户退出，已经释放{session_id}的资源")
    state["logged_in"] = False
    state["username"] = ""
    # 退出后显示注册和登录界面，隐藏应用


    return (state, gr.update(visible=False), gr.update(visible=True),
            ""
                #,gr.update(visible=True),gr.update(visible=True)
    ) # 清空登录输入框

def loginToRegister():

    # 退出后显示注册和登录界面，隐藏应用界面
    return  gr.update(visible=False), gr.update(visible=True),""
def guest_login(state,knowledgeDB:KnowledgeDB,request:gr.Request):
    try:

        state["logged_in"] = True

        knowledgeDB.reset("guest")
        id =request.session_hash
        state["session_id"] = id
        knowledgeDB.get_dbs_file()



      # f"guest_{len(ChatAgents)}保证游客在ChatAgents的key不一样的
        guest_id=random.randint(0, 1000000)
        while   guest_id in ChatAgents:
            guest_id = random.randint(0, 1000000)

        username=f"guest_{guest_id}"

        state["username"] = username
        ChatAgents[id] = Agent(id)


            #guest本地不保存数据库，所以每次登录应该是看不到文件列表的；这里是为了便于测试，加入了该功能。
                                               #后续修改，删除knowledgeDB，注释这一行即可，并且不返回DB
        # 登录成功，修改状态,隐藏登陆界面，显示主应用界面
        return "登录成功！", state, gr.update(visible=False), gr.update(visible=True) ,\
              gr.update(value=f"""   <center><b>用户：游客</b></center> 
                            """),knowledgeDB,gr.update(choices=knowledgeDB.files.get("text-embedding-ada-002",[]))


    except Exception as e:
        error_info = traceback.format_exc()  # 将完整的错误输出为字符串
        logging.basicConfig(filename="log.txt",
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                            level=logging.ERROR)  # 设置日志级别，只有调用该函数才会将大于等于level的日志信息保存到项目根目录下的log文件中;不调用，则输出到控制台
        logging.error(f"游客登录发生错误：\n{error_info}")  # 错误输出到log.txt, 存在于项目根目录
        print(error_info)

        return f"游客登录失败：{str(e)}", state,gr.update(visible=True), gr.update(visible=False),\
            gr.update(),gr.update(),gr.update()

