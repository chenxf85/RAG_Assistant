import copy

from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self

import logging,traceback
from prompt.prompt import default_template
from typing import Optional,Literal
import time

from globals import LLM_MODEL_MAXTOKENS_DICT
import sys

import copy
from Agent.tools import tools
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from  llm.model_to_llm import model_to_llm
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from pydantic import Field,BaseModel
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_community.tools import Tool,StructuredTool




class Agent(): #能否state化呢？
    #todo
    """
    存储问答 Chain 的对象

    - chat_qa_chain_self: 以 (model, embedding) 为键存储的带历史记录的问答链。

    """
    def __init__(self,id):
        self.id=id
        self.chat_qa_chains= {}
        self.prompts=default_template
   #     self.active_time=time.time() #当长时间不活跃，定时释放agent资源时用到。（处理因为网络断开等会话非正常结束的情况）
        self.chat_history=[]#agent的chat_histroy
        self.chat_history_LCEL = []
        self.memory=InMemorySaver()

        self.tools=copy.deepcopy(tools)

    def get_tool_dict(self,tool_names):
        tools={}
        for name, val in self.tools.items():
            if name in tool_names:
                if isinstance(val, dict): #工具集
                    for tool_name,func in val.items():
                        tools[tool_name] = func
                else:
                    tools[name] = val  #工具
        return tools

    def chat_qa_chain_self_answer(self, question: str, chat_history: list = [], model_type: str = "OPENAI", model:str="gpt-3.5-turbo",model_dir:str=None,context_window:int=8192,quantization_config:Optional[str]=None,
                                  embedding_type:str="OPENAI",embedding: str ="text-embedding-ada-002",embedding_dir:str=None,
                                  temperature: float = 0.0, top_k_llm: int = 4, history_len: int = 3,max_tokens:int=1024,top_p=0.5,#llm参数
                                  top_k_query:int=4,score_threshold:float=0.8,  fetch_k:int=20,lambda_mult:float=0.5,search_type:str="similarity_score_threshold",
                                  chat_mode:dict[str,bool]=None,is_abstract:bool=False,is_with_history:bool=False,
                                  api_key: str = None, api_base: str = None,
                                  DB=None,upload_files:Optional[list[str]]=[],rag_files:Optional[list[str]]=[],
                                  rag_config:str="快速搜索",web_config="duckduckgo",websearch_max_results=3,tool_names=None,
                                  reranker_config:str="LLM重排"
                                  ):

        """
        调用带历史记录的问答链进行回答;
        """



        if not question and not is_abstract:
            yield "请输入问题", chat_history
            return

        if question:
           # question=question.split("/think")[0] if question.endswith("/think") else question.split("/no_think")[0]
            self.chat_history.append([question, ""])
            yield "正在生成回答", self.chat_history



        chains=self.chat_qa_chains
        chat_history_LCEL = self.get_chat_history_LCEL(history_len)
        try:
            tools = self.get_tool_dict(tool_names)
            model_name=model+"_"+quantization_config if model_type in ["HuggingFace","llama_cpp"] and \
            quantization_config else model
            if (model_name, embedding) not in chains:
                chains[(model_name, embedding)] = \
                    Chat_QA_chain_self(model_type=model_type, model=model,model_dir=model_dir,context_window=context_window,
                                       chat_history=chat_history,embedding_dir=embedding_dir,
                                       embedding_type=embedding_type, embedding=embedding,
                                       DB=DB)


            chain = chains[(model_name, embedding)]

            output_stream=chain.answer(question=question, prompts=self.prompts,temperature=temperature, top_k_llm=top_k_llm, max_tokens=max_tokens,quantization_config=quantization_config,
                                             top_p=top_p,top_k_query=top_k_query, score_threshold=score_threshold,
                                             fetch_k=fetch_k,lambda_mult=lambda_mult, search_type=search_type,history_len=history_len,
                                             api_key=api_key,api_base=api_base,chat_mode=chat_mode,is_abstract=is_abstract,is_with_history=is_with_history,
                                             rag_files=rag_files,upload_files=upload_files,rag_config=rag_config,web_config=web_config,max_results=websearch_max_results,DB=DB,tools=tools,
                                       reranker_config=reranker_config,chat_history_LCEL=chat_history_LCEL)
            for info,answer in output_stream:
                if isinstance(answer,list):
                    self.chat_history.append(answer)
                else:
                    self.chat_history[-1][1] += answer

                yield info,self.chat_history
            self.chat_history_LCEL.append(("human", question))
            answer=self.chat_history[-1][1].split("<b>回答:</b><br>")[-1] #剔除参考资料片段，网络搜索结果，避免模型上下文过大
            self.chat_history_LCEL.append(("ai", answer))
            return


        except Exception as e:
            error_info = traceback.format_exc()  # 将完整的错误输出为字符串
            logging.basicConfig(filename="log.txt",
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                                level=logging.ERROR)  # 设置日志级别，只有调用该函数才会将大于等于level的日志信息保存到项目根目录下的log文件中;不调用，则输出到控制台
            logging.error(f"chat_process发生错误：\n{error_info}")  # 错误输出到控制台，存在于项目根目录
            print(error_info)
            yield e, self.chat_history
            return


    def get_rag_tool(self,chain,docs=None,**kwargs):
        '''
        将workflow内部搭建的rag chain, creat_doc_chain 作为agent可用的调用工具
        没有放在tools文件定义，是因为每次调用的rag_tool会根据用户输入的搜索参数动态调整
        '''

        kwargs["chat_mode"]={"is_rag":True}
       # kwargs["rag_config"]="快速搜索"
        # Runnable对象工具化
        class ToolInput(BaseModel):
            query:str=Field(description="用户提出的问题，直接提取输入input，无需任何修改")
            rag_config:Literal["快速搜索","高级搜索"]=(
                Field(description="检索配置，根据用户的要求，选择搜索模式；"
                                  "如果用户输入侧重搜索时间短，选择快速搜索；如果侧重高精度的搜索，选择准确搜索"
                      "当用户输入未体现搜索要求或者准确搜索失败时，采用快速搜索",
                      default="快速搜索"))

        def rag_tool_func(query: str,rag_config:Literal["快速搜索","高级搜索"]="快速搜索") -> str:
            # 这里确保直接使用用户查询
            kwargs['rag_config']=rag_config
            kwargs["reranker_config"]="LLM重排"  #如果采用准确搜索，默认用LLM重排；

            query_change_chain1, query_change_chain2=chain.create_query_change_chain(llm=chain.llm)
            chat_history_LCEL = self.get_chat_history_LCEL(kwargs["history_len"]) if kwargs["is_with_history"] else []
            # if kwargs["is_with_history"] or docs:
            #     query1=query_change_chain1.invoke({"input": query,
            #                                      #  "chat_history":chat_history_LCEL,  因为在调用工具之前已经把上下文传给agent，这一步可以根据记忆改写查询，这里跳过
            #                                        "context":docs}) #问题改写：结合上传文档和上下文改写
            # else:
            #     query1={"input":query}
            chain.get_retriever(**kwargs)  #
            query={"input": query}
            if kwargs["rag_config"]=="快速搜索": #单一查询
                return chain.create_rag_chain().invoke(query).get("rag_context", "")
            elif kwargs["rag_config"]=="高级搜索": #多重查询，改写问题

                #输出隔离问题。在agent调用工具的过程中，如果工具包含了LLM调用，可能会导致上下文污染（将工具LLM输出结果传入agent的上下文）
                queries=query_change_chain2.invoke(query,config={"callbacks": []}).get("queries",[query])
                rag_context= chain.create_rag_chain().invoke({"queries": queries},config={"callbacks": []}).get("rag_context", "")
                return rag_context
            else:
                raise ValueError("rag_config参数错误，只能是快速搜索或者准确搜索")


        rag_tool = StructuredTool.from_function(
            func= rag_tool_func,
          #  coroutine= rag_tool_func,
            name="RAG_TOOL",
            description="这是一个知识库检索的工具。当无法回答用户问题时使用，在数据库中检索相关结果以增强输出。",
            args_schema=ToolInput
        )

        return rag_tool
    def get_docs_output(self,chain,**kwargs):
        upload_files=kwargs.get("upload_files",False)

        doc_output={}
        if upload_files:
            new_docs = chain.get_docs(upload_files)
            doc_tool = chain.create_docs_chain("context")
            doc_output=doc_tool.invoke({"context":new_docs})

        return doc_output

    def agent_answer(self, query, chat_history, **kwargs):
        if not query:
            yield "请输入问题", chat_history
            return
        else:
            info = "正在输出回答"
            # query = query.split("/think")[0] if query.endswith("/think") else query.split("/no_think")[0]
            self.chat_history.append([query, ""])
            yield info, self.chat_history

        model = kwargs["model"]
        embedding = kwargs["embedding"]
        chains = self.chat_qa_chains
        model_type = kwargs["model_type"]
        model_dir = kwargs["model_dir"]
        context_window = kwargs["context_window"]
        embedding_type = kwargs["embedding_type"]
        output_mode=kwargs["agent_output_mode"]
        embedding_dir=kwargs["embedding_dir"]
        DB = kwargs["DB"]
        quantization_config=kwargs["quantization_config"]

        if model_type in ["HuggingFace","llama_cpp"] and "qwen3" in model.lower():
            if kwargs["chat_mode"]["is_reasoning"]:
                query+="/think"
            else:
                query += "/no_think"
        model_name = model + "_" + quantization_config if model_type in ["HuggingFace", "llama_cpp"] and \
                                                          quantization_config else model
        if (model_name, embedding) not in chains:
            chains[(model_name, embedding)] = \
                Chat_QA_chain_self(model_type=model_type, model=model, model_dir=model_dir,
                                   context_window=context_window,
                                   chat_history=[],  # 这里只是调用chain的工具没有用到这个chat_history
                                   embedding_type=embedding_type, embedding=embedding,embedding_dir=embedding_dir,
                                   DB=DB)
        chain = chains[(model_name, embedding)]

        model = model_to_llm(
            model_type=model_type,
            model=model,
            model_dir=model_dir,  # 只在本地模型时起作用 ?
            quantization=quantization_config,  # 只在本地模型时起作用 ?
            temperature=kwargs["temperature"],
            top_k=kwargs["top_k_llm"],
            top_p=kwargs["top_p"],
            max_tokens=kwargs["max_tokens"],
            api_key=kwargs["api_key"],
            api_base=kwargs["api_base"],
            is_reasoning=kwargs["chat_mode"]["is_reasoning"],
        )
        chain.llm=model


        doc_output = self.get_docs_output(chain, **kwargs)
        config = {"configurable": {"thread_id": self.id, "context": doc_output.get("context", "")}}
        tools = self.flatten_tools(tool_names=kwargs["tools"], chain=chain,docs=doc_output, **kwargs)

        def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
            context = config["configurable"].get("context", "")

            system_msg = (f"当提供了工具列表的时候，你是可以调用工具回答问题。"
                          f"你需要根据用户问题，识别用户的意图，将复杂任务分解\n"
                          f"比如当用户要求在知识库检索时，你必须调用知识库检索工具;否则不需要使用知识库工具"
                          f"为多个子问题。根据子问题，调用不同的工具最终完成问题.\n"
                
                              f"这里检测到用户上传的文本(如果为空则忽略)：{context}\n"
                          
                          f"如果未提供工具，则直接输出你的回答"
                          )
            return [{"role": "system", "content": system_msg}]+state["messages"]

        tools_num = 1
        chat_history_LCEL=self.get_chat_history_LCEL(kwargs["history_len"]) if kwargs["is_with_history"] else []
        #print(chat_history_LCEL)

        if kwargs.get("rag_files", []):
            query = "使用知识库检索，回答下列问题：\n" + query+"\n如果结果不够充实，可以使用其他工具" # 把知识库检索要求放在用户输入，问题改写可能遗漏这一要求；或许考虑在prompt中加入

        if kwargs["is_with_history"]:
            agent = create_react_agent(model, tools,  prompt=prompt) if \
                hasattr(model,"bind_tools") else  create_react_agent(model,prompt=prompt)
            chunks = agent.stream(
                {"messages":  chat_history_LCEL+[("human", query)]},
                config=config,
                stream_mode="messages"
            )
        else:
            agent = create_react_agent(model, tools, prompt=prompt) if  hasattr(model,"bind_tools") else  create_react_agent(model,prompt=prompt)
            chunks = agent.stream(
                {"messages": [("human", query)]},
                config=config,
                stream_mode="messages"
            )

        self.chat_history[-1][1] += "<b>回答:</b><br>"
        try:
            if output_mode == "React输出":
                start_reasoning = True
                for chunk in chunks:
                    chunk=chunk[0]
                    print(chunk)
                    if isinstance(chunk, ToolMessage):
                        start_reasoning = True
                        if not self.chat_history[-1][1].endswith("<br>")  and not self.chat_history[-1][1].endswith("\n"):
                            self.chat_history[-1][1] +="<br>"  # 去掉末尾的<br>
                        self.chat_history[-1][1]+="-------------------------------------------<b>Action</b>-------------------------------------------<br>"
                        self.chat_history[-1][1] += f"第{tools_num}次工具调用，{chunk.name}的输出结果为{chunk.content}\n"

                        tools_num += 1
                        yield info, self.chat_history
                    elif isinstance(chunk, AIMessage) and chunk.content :

                        if start_reasoning:
                            if not self.chat_history[-1][1].endswith("<br>") and not self.chat_history[-1][1].endswith("\n"):
                                self.chat_history[-1][1] += "<br>"
                            self.chat_history[-1][1]+="----------------------------------------<b>Reasoning</b>------------------------------------------<br>"
                            start_reasoning = False

                        self.chat_history[-1][1]+=chunk.content

                        yield info, self.chat_history

            elif output_mode == "标准输出":
                last_AIchunk=False
                for chunk in chunks:
                    chunk = chunk[0]
                    print(chunk)
                    if isinstance(chunk, AIMessage):
                        self.chat_history[-1][1] += chunk.content
                        last_AIchunk = True
                    else:
                        if last_AIchunk:
                            self.chat_history[-1][1] += "<br>"
                            last_AIchunk = False
                    yield info, self.chat_history
        except Exception as e:
            error_info = traceback.format_exc()  # 将完整的错误输出为字符串
            logging.basicConfig(filename="log.txt",
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                                level=logging.ERROR)  # 设置日志级别，只有调用该函数才会将大于等于level的日志信息保存到项目根目录下的log文件中;不调用，则输出到控制台
            logging.error(f"agent_answer发生错误：\n{error_info}")  # 错误输出到控制台，存在于项目根目录
            print(error_info)
            yield e, self.chat_history
            return

        self.chat_history_LCEL.append(("human", query))
        answer=\
        self.chat_history[-1][1]. split("----------------------------------------<b>Reasoning</b>------------------------------------------<br>")[-1]
        self.chat_history_LCEL.append(("ai", answer))

        yield "回答完成", self.chat_history

    def flatten_tools(self,tool_names,chain,docs,**kwargs)->list:
        all_tools = []
        for name,val in self.tools.items():
            if name in tool_names:
                if isinstance(val, dict):
                    all_tools.extend(val.values())
                else:
                    if name=="RagTool":
                        all_tools.append(self.get_rag_tool(chain,docs,**kwargs))
                    else:
                        all_tools.append(val)
        return all_tools

    def clear_history(self):
        self.chat_history.clear()
        self.chat_history_LCEL.clear()
        self.memory.delete_thread(self.id)

    def get_chat_history_LCEL(self, history_len):  # 用于LCEL invoke方法

        """
          返回chatPomomptTemplata可用的message形式，注意一次记忆包括一组question和answer，而self.chat_history
           self.chat_history.append(("human",question))
           self.chat_history.append(("ai", answer))
           所以记忆长度应该为：len(self.chat_history)/2
            """

        if not self.chat_history_LCEL:
            return []
        elif len(self.chat_history) < history_len:
            return self.chat_history_LCEL
        else:
            return self.chat_history_LCEL[-history_len * 2:]

    # def to_history_lcel(self):  # 当传入chat_history是list[tuple(str,str)],需要转为LCEL格式：
    #     chat_history_LCEL = []
    #     if not self.chat_history:
    #         return []
    #     for question, answer_ui in self.chat_history:
    #         chat_history_LCEL.append(("human", question))
    #         answer = self.extract_answer(answer_ui)  # 剔除参考资料片段
    #         chat_history_LCEL.append(("ai", answer))
    #     return chat_history_LCEL

