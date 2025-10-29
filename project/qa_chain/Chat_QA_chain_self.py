
from langchain_chroma.vectorstores import Chroma
import  trafilatura
import sys,os
sys.path.append(os.getcwd())
#import magic
from llm.model_to_llm import model_to_llm
from langchain_core.messages import AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from globals import only_reason_models
from qa_chain.File_Browse_Chain import LCELBrowser
from collections import defaultdict
from utils.formatPackage import addFull

from database.get_scores import get_scores
from langchain_core.prompts import format_document
import os
from utils.fileProcess import file_loader, get_docs

from langchain.schema import StrOutputParser, BaseOutputParser
from langchain.retrievers.multi_query import LineListOutputParser


from langchain_core.retrievers import (
    BaseRetriever,
    RetrieverOutput,
)
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableBranch, RunnableLambda
import requests

from bs4 import BeautifulSoup
from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_PROMPT,
)
from operator import attrgetter

from langchain_core.documents import Document
from prompt.prompt import _validate_prompt
from langchain_core.messages      import ToolMessage
from langchain_core.callbacks.manager import (

    CallbackManagerForRetrieverRun,
)


import json
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, DocumentCompressorPipeline
from utils.reranker import CrossEncoderReranker2,LLMReranker2 #对langchain内置的reranker的改写，分别实现交叉编码器和LLM重排

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import torch
from utils.ensembleRetriever import EnsembleRetriever2

from sentence_transformers import CrossEncoder
from duckduckgo_search import DDGS


from globals import only_reason_models
from utils.replace_think_tag import replace_think_tag_stream
from transformers import AutoTokenizer

class EmptyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return []

    def _aget_relevant_documents(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return []


class NoOutputParser(BaseOutputParser[str]):
    """Parse outputs that could return a null string of some sort."""

    no_output_str: str = "NO_OUTPUT"

    def parse(self, text: str) -> str:
        cleaned_text = text.strip()
        if cleaned_text == self.no_output_str:
            return ""
        return cleaned_text



class Chat_QA_chain_self:
    """"
    带历史记录的问答链
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - chat_history：历史记录，输入一个列表，默认是一个空列表
    - history_len：控制保留的最近 history_len 次对话
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火
    - api_key：星火、百度文心、OpenAI、智谱都需要传递的参数
    - Spark_api_secret：星火秘钥
    - Wenxin_secret_key：文心秘钥
    - embeddings：使用的embedding模型
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）
    """


    # if rank_model.config.pad_token_id is None:
    # if rank_model.config.pad_token_id is None:
    #     rank_model.config.pad_token_id = rank_model.config.eos_token_id

    ranker_model= None
    def __init__(self, **kwargs):
        self.model_type = kwargs["model_type"]
        self.model = kwargs["model"]
        self.llm= None #在param_set中初始化
        self.model_dir = kwargs['model_dir']
        self.embedding_type = kwargs["embedding_type"]
        self.embedding = kwargs["embedding"]
        self.embedding_dir = kwargs["embedding_dir"]

        self.total_tokens: list[int] = []  # 每轮对话的token数量
        self.sql_path = os.path.join(kwargs["DB"].persist_directory, self.embedding_type, "chroma.sqlite3")

        self.vectordb: Chroma = kwargs["DB"].get_vectordb(self.embedding_type, self.embedding,self.embedding_dir)
        self.context_window=kwargs["context_window"]
        self.summarizer=None #在param_set中初始化

        self.documents = kwargs["DB"].get_docs(self.embedding)
        #   self.document_variable_names = ["context", "rag_context"]
        self.scores = []
        self.model_path= os.path.join(self.model_dir, self.model)

    @classmethod
    def get_reranker_model(cls):
        reranker_path = "model/reranker/Qwen/Qwen3-Reranker-0___6B"
        if cls.ranker_model is None:
            tokenizer = AutoTokenizer.from_pretrained(reranker_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token  # 用 eos_token 作为 pad_token
                # 读取reranker_peth下config.json中的eos_token_id
                # json文件转为字典
            with open(os.path.join(reranker_path, "config.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
            pad_token_id = data.get("pad_token_id", data.get("eos_token_id", 151645))

            model_kwargs = {'device': "cuda:0" if torch.cuda.is_available() else "cpu",
                            "tokenizer_kwargs": {"pad_token": tokenizer.pad_token},
                            "config_kwargs": {"pad_token_id": pad_token_id}}  # CrossEncoder的参数
            cls.ranker_model = HuggingFaceCrossEncoder(model_name=reranker_path,
                                             model_kwargs=model_kwargs,

                                             )

    def param_set(self, **kwargs):

        self.llm = model_to_llm(
            model_type=self.model_type,
            model=self.model,
            model_dir=self.model_dir, #只在本地模型时起作用 ?
            quantization=kwargs["quantization_config"],  #只在本地模型时起作用 ?
            temperature=kwargs["temperature"],
            top_k=kwargs["top_k_llm"],
            top_p=kwargs["top_p"],
            max_tokens=kwargs["max_tokens"],
            api_key=kwargs["api_key"],
            api_base=kwargs["api_base"],
            is_reasoning=kwargs["chat_mode"]["is_reasoning"]
        )
        # print(f"self.llm:{sys.getrefcount(self.llm)}")
        self.get_prompt(**kwargs)
        self.summarizer = LCELBrowser(llm=self.llm,
                                          token_max=self.context_window)  # 这里的token_max是指每个文档的最大token数])
        # print(f"self.llm:{sys.getrefcount(self.llm)}")

    def get_retriever(self, **kwargs):
        '''
        高级RAG：采用多查询召回，重排，上下文压缩和混合检索
        '''
    #    self.vectordb = kwargs["DB"].get_vectordb(self.embedding_type, self.embedding)  # 和底层数据库同步
     #   is_retriever=   kwargs["chat_mode"]["is_rag"] or kwargs["chat_mode"]["is_web_search"]
        if not kwargs["chat_mode"]["is_rag"]:
            self.retriever = EmptyRetriever()
        else:
            # 基础检索器配置
            search_kwargs = {
                "k": kwargs["top_k_query"]
            }

            if kwargs["search_type"] == "similarity_score_threshold":
                if kwargs["rag_config"] == "快速搜索":
                    search_type = kwargs["search_type"]
                    search_kwargs["score_threshold"] = kwargs["score_threshold"]  # 快速搜索时这个值是向量检索相似度阈值
                else:
                    search_type = "similarity"
            elif kwargs["search_type"] == "mmr":
                search_type = kwargs["search_type"]
                search_kwargs["fetch_k"] = kwargs["fetch_k"]
                search_kwargs["lambda_mult"] = kwargs["lambda_mult"]

            # 如果用户指定了文件，则加上过滤器:
            if kwargs.get("rag_files"):
                search_kwargs["filter"] = {"source": {"$in": kwargs["rag_files"]}}
                self.documents = kwargs["DB"].get_docs(self.embedding, kwargs["rag_files"])
            else:
                self.documents = kwargs["DB"].get_docs(self.embedding)

            # 1. 创建基础向量检索器
            base_retriever = self.vectordb.as_retriever(
                search_type = search_type,
                search_kwargs = search_kwargs
            )

            if kwargs["rag_config"] == "快速搜索":

                self.retriever = self.create_chat_retriever(
                    base_retriever,
                     **kwargs
                )
            elif kwargs["rag_config"] == "高级搜索":
                # 2. 创建 BM25 文本检索
                # todo
                self.bm25_retriever = BM25Retriever.from_documents(
                    self.documents,
                    k=kwargs["top_k_query"]
                )
                # 3. 创建混合检索器:融合向量检索和文本检索
                ensemble_retriever = EnsembleRetriever2(
                    retrievers=[base_retriever, self.bm25_retriever],
                    weights=[0.8, 0.2],
                    top_n=kwargs["top_k_query"]
                )

                # 4. 后处理：创建重排序器和上下文压缩器
                # 重排序器
                top_n = kwargs["top_n"] if kwargs.get("top_n", False) else min(kwargs["top_k_query"], 5)
                if kwargs["reranker_config"]=="交叉编码器重排":
                    self.get_reranker_model()
                    self.reranker = CrossEncoderReranker2(model=self.ranker_model, top_n=top_n,
                                                      score_threshold=kwargs["score_threshold"])  # raranker得分的阈值
                else:
                    self.rerankerLLM=LLMReranker2(llm=self.llm,top_n=top_n,score_threshold=kwargs["score_threshold"])
                # 上下文压缩器

                output_parser = NoOutputParser()
                template = """根据以下文本和问题，从文本中提取和问题相关的内容。如果文本和问题内容有关，则返回原内容。
                                                如果文本和问题无关，则返回{no_output_str}。
                                                以下是问题：
                                                {{question}}
                                                以下是文本：
                                                {{context}}
                                                注意！！禁止回答问题"""
                extractor_template = template.format(no_output_str=output_parser.no_output_str)
                extractor_prompt = PromptTemplate(
                    template=extractor_template,
                    input_variables=["question", "context"],
                    output_parser=output_parser,
                )
                # llm = model_to_llm(
                #     model_type="ZHIPUAI",
                #     model="glm-4-long",
                #
                #     temperature=0
                # ) if not self.llm else self.llm

                compressor = LLMChainExtractor.from_llm(llm=self.llm, prompt=extractor_prompt)  #上下文压缩处理
                if kwargs["reranker_config"]=="交叉编码器重排":
                    pipeline_compressor = DocumentCompressorPipeline(transformers=[self.reranker])  #重排序
                else:
                    pipeline_compressor = DocumentCompressorPipeline(transformers=[self.rerankerLLM]) #串联多个压缩器
                # compression_retriever = ContextualCompressionRetriever(
                #      base_compressor=pipeline_compressor, base_retriever=ensemble_retriever
                #  )  # 具有重排序和上下文压缩处理的混合检索器
                self.retriever = self.create_chat_retriever( #没有用上下文压缩检索器，是因为要和多查询结合
                ensemble_retriever,
                    compressor=pipeline_compressor, **kwargs
                )


    def create_chat_retriever(self, base_retriever, compressor=None, **kwargs):

        def retriever_chain(inputs: dict, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:

            # 并发执行多个检索任务
            if not compressor:
                query = inputs.get("query",inputs["input"])
                query = query.split("/think")[0] if query.endswith("/think")\
                    else query.split("/no_think")[0]
                docs = base_retriever.invoke(query)
                # print(f"所有文档{docs}\n")
                self.scores = get_scores(query, self.vectordb, **kwargs)
                return list(docs)
            else:
                queries = inputs["queries"]
                results =  [base_retriever.invoke(query, config={"callbacks": run_manager.get_child()})
                         for query in queries]
            #    results = await asyncio.gather(*tasks)

                # 扁平化所有文档，并去重
                all_docs = [doc for docs in results for doc in docs]
                unique_docs = []
                seen = set()
                # print(f"文档数量:{len(all_docs)}\n 所有文档{all_docs}\n")
                for doc in all_docs:
                    key = (doc.page_content, doc.metadata.get("source", ""))
                    if key not in seen:
                        seen.add(key)
                        unique_docs.append(doc)
                # print(f"文档数量:{len(unique_docs)}\n 去重文档{unique_docs}\n")

                if unique_docs and compressor:
                    compressed_docs = compressor.compress_documents(
                        unique_docs,
                        query=" ".join(queries),
                      #  callbacks=run_manager.get_child()
                    )
                    # print(f"文档数量:{len(compressed_docs)}\n 提取文档{compressed_docs}\n")
                    self.scores = self.reranker.scores if kwargs["reranker_config"]=="交叉编码器重排" else self.rerankerLLM.scores

                    return list(compressed_docs)
                else:
                    return []


        retrieve_documents = RunnableLambda(retriever_chain).with_config(run_name="chat_retriever_chain")

        return retrieve_documents


    @staticmethod
    def create_docs_chain( document_variable_name="context"):
        _document_prompt = DEFAULT_DOCUMENT_PROMPT

        def format_docs(inputs: dict) -> str:
            return "\n\n".join(
                format_document(doc, _document_prompt)
                for doc in inputs[document_variable_name]
            )

        deal_docs_chain = RunnablePassthrough.assign(**{document_variable_name: format_docs}).with_config(
            run_name="format_inputs")
        return deal_docs_chain
    @staticmethod
    def create_query_change_chain(llm=None, prompt1=None, prompt2=None,k_queries=1):

        #  output_parser = StrOutputParser() if k_queries == 0 else LineListOutputParser()
        # print(f"query_answer:{sys.getrefcount(llm)}")
        llm = model_to_llm(
            model_type="ZHIPUAI",
            model="glm-4.5-flash",
            temperature=0
        ) if not llm else llm  # 用于问题改写的llm，不是用于对话的llm
        # print(f"query_answer:{sys.getrefcount(llm)}")
        if not prompt1:
            condense_question_system_template = (
                "你是一个帮助生成搜索查询的助手。下面你将会看到用户的最新问题，对话历史和上传的文档"
                "该问题可能依赖于之前的上下文。"
                "使其在没有上下文的情况下也能被理解。"
                "不需要回答问题，只需在必要时进行改写，如果已经是完整问题就原样返回。"
                "以下是上传文档的内容:{context}"
            )
            prompt1 = ChatPromptTemplate.from_messages([
                ("system", condense_question_system_template),
                ("placeholder", "{chat_history}"),
                ("human", "{input}")
            ])
        if not prompt2:
            condense_question_system_template = (
                "你是一个帮助生成搜索查询的助手。下面你将会看到用户结合历史修改的查询"
                "为了丰富用户检索文档的多样性，你需要拓展查询"
                f"请将这个问题从不同角度改写成{k_queries}个独立的、完整的搜索查询，但不要回答问题"
                "使其在没有上下文的情况下也能被理解。"
                "注意！！禁止回答问题。只需在必要时进行改写"
            )
            prompt2 = ChatPromptTemplate.from_messages([
                ("system", condense_question_system_template),
                ("placeholder", "{chat_history}"),
                ("human", "{query}")
            ])
        llm_chain1 = prompt1 | llm | StrOutputParser()
        llm_chain2 = prompt2 | llm | LineListOutputParser()
        import copy
        def query_change1(inputs: dict):
            # 根据历史和上传的文档改写查询
            deal_inputs=copy.deepcopy(inputs)
            deal_inputs["input"]= inputs["input"].split("/think")[0] if inputs["input"].endswith("/think") else inputs["input"].\
                                                                                                                split("/no_think")[0]
            query = llm_chain1.invoke(deal_inputs)
            return query

        def query_change2(inputs: dict):
            # 多查询改写
            inputs["query"]= inputs.get("query",inputs["input"])
            queries = llm_chain2.invoke(inputs,config={"callbacks": []})
            queries.append(inputs["query"])
            query = inputs["input"].split("/think")[0] if inputs["input"].endswith("/think") \
                else inputs["input"].split("/no_think")[0]
            queries.append(query)
            queries=list(dict.fromkeys(queries))
            #queries去重




            return queries

        query_change_chain1 = RunnablePassthrough.assign(query=query_change1).with_config(
            run_name="query_change_chain1"
        )
        query_change_chain2 = RunnablePassthrough.assign(queries=query_change2).with_config(
            run_name="query_change_chain2"
        )
        return query_change_chain1 , query_change_chain2

    @staticmethod
    def create_web_search_chain(**kwargs):

        def fetch_readable_text(url: str, snippet: str = "") -> str:
            """
            尝试提取网页正文，失败时返回 snippet。
            """
            try:
                # downloaded = trafilatura.fetch_url(url)
                # if downloaded:
                #     extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                #     print(extracted)
                #     if extracted and len(extracted) > 50:
                #         print("trafilatura成功抓取")
                #         return f"摘要:{snippet}\n{extracted}"
                #     else:
                #         print("无法爬取或爬取太短,返回摘要")
                #         return snippet
                # else:
                #     print("无效爬取")
                #     return snippet
                return snippet #这个包爬取太慢，这里直接跳过
            except Exception:
                # print("返回摘要")
                return snippet

        def web_search(inputs: dict) -> str:
            """
            联网搜索函数。
            支持 'duckduckgo' 与 'serperapi' 两种搜索模式。
            返回HTML格式的搜索结果（适合Gradio展示）。
            """
            query = inputs["query"] if inputs.get("query", False) else inputs["input"]

            # ✅ DuckDuckGo 搜索
            if kwargs["web_config"] == "duckduckgo":
                from duckduckgo_search import DDGS

                with DDGS() as ddgs:
                    results = ddgs.text(query, max_results=kwargs.get("max_results", 5))
                    texts = ""
                    for r in results:
                        title, url, snippet = r["title"], r["href"], r["body"]
                        contents = fetch_readable_text(url, snippet)
                        texts += f'<details><summary><a href="{url}" target="_blank">{title}</a></summary><br><p>' \
                                 f'{contents[:500]}</p><br></details>'
                return texts

            # ✅ SerperAPI 搜索
            elif kwargs["web_config"] == "serperapi":
                SERPER_API_KEY = "354c335007f57fb4ee5cf2024d884bde966ee4b3"

                url = "https://google.serper.dev/search"
                headers = {
                    "X-API-KEY": SERPER_API_KEY,
                    "Content-Type": "application/json",
                }
                payload = {
                    "q": query,
                    "num": kwargs.get("max_results", 5),
                    "hl": "zh-cn"
                }

                response = requests.post(url, headers=headers, json=payload, timeout=10)
                if response.status_code != 200:
                    raise Exception(f"SerperAPI 调用失败: {response.status_code}, {response.text}")

                data = response.json()
                results = data.get("organic", [])
                texts = ""
                for r in results:
                    title = r.get("title", "")
                    link = r.get("link", "")
                    snippet = r.get("snippet", "")
                    contents = fetch_readable_text(link, snippet)
                    texts += f'<details><summary><a href="{link}" target="_blank">{title}</a></summary><br><p>' \
                             f'{contents[:500]}</p><br></details>'

                return texts

            else:
                raise ValueError(f"web_config 必须为 'duckduckgo' 或 'serperapi'，当前为 {kwargs['web_config']}")

        web_search_chain = RunnablePassthrough.assign(web_context=web_search).with_config(
            run_name="web_search_chain"
        )
        return web_search_chain

    def create_rag_chain(self,document_variable_name="rag_context"):
        if not isinstance(self.retriever, BaseRetriever):
            retrieval_docs: Runnable[dict, RetrieverOutput] = self.retriever
        else:
            retrieval_docs = (lambda x: x["input"]) | self.retriever
        def get_ref_ui(inputs: dict) -> str:
            """
            根据问题返回检索到的文档,与内容
            """
            #   answer = result["answer"]
            #    docs = result.get("context", [])
            docs = inputs[document_variable_name]
            ref_ui = defaultdict(list)
            for doc in docs:
                source = doc.metadata.get("source")
                page_content = doc.page_content.strip()
                if doc.metadata.get("page"):
                    page = doc.metadata.get("page") + 1  # 返回的page从0开始
                    # print(f"{source},{page_content},{page}\n\n")

                else:
                    page = "未知"

                if source and page_content and page:
                    ref_ui[source].append((page, page_content))

            references_ui = ""

            for filename, contents in ref_ui.items():
                references_ui += f"<details><summary>{filename}</summary>"
                for i, ele in enumerate(contents):
                    page = ele[0]
                    content = ele[1]
                    if content[-1] != "\n" or content[-1] != "<br>":
                        content += "\n"
                    content = addFull(content)
                    references_ui += f"<p><b>片段{i + 1}(页码{page})：</b>{content}</p>"
                references_ui += "</details>"
                references_ui += "<!-- 这是一个 HTML 注释,防止模型回答的markdown输出错误渲染 -->"
            # 仅在 UI 上合并，但 answer 本身不包含这些内容

            return references_ui

        rag_chain = RunnablePassthrough.assign(
            **{document_variable_name: retrieval_docs.with_config(run_name="retrieve_documents")},
        ).assign(
            **{document_variable_name:RunnableLambda(get_ref_ui).with_config(
                run_name="form_ref")})  # .assign \

        return rag_chain

    def create_tool_chain(self,tools):
        ##prompt需要将检索的结果传递给llm，所以提示词需要有相应的占位符
        #这里tool_call的同理也可以这么做，但是因为提供了Toolmessage作为模型输入，所以也可以在create_chat_chain时将ToolMessage与ChatPromptTemplate合并
        #但是这种方法不方便输出Tool调用结果，于是采用同web_search_chain，一样的方法
        if not hasattr(self.llm, "bind_tools"):
            return RunnableLambda(lambda x: x)

        llm_with_tools = self.llm.bind_tools(list(tools.values()))
        print(list(tools.values()))
        get_tool_calls = llm_with_tools | attrgetter("tool_calls") #1.工具调用返回tool_call列表

        # 2. 工具输出
        def call_tools(tool_calls):
            tool_outputs = []
            for tool_call in tool_calls:
                tool_name=tool_call["name"]
                tool_fn = tools.get(tool_name)
                if not tool_fn:
                    continue
                msg = tool_fn.invoke(tool_call)
                tool_outputs.append(
                    (tool_name,msg)
                )
           # print(tool_outputs)
            return tool_outputs

        def show_tool_outputs(inputs:dict)->str:
            input=inputs["input"]
            tool_result = ""
            tool_calls=get_tool_calls.invoke(input) #工具调用
          #  print(llm_with_tools.invoke(input))
        #    print(tool_calls)
            tool_outputs=call_tools(tool_calls) #工具输出
            for  tool_name,tool_message in tool_outputs:
                tool_result += f"<details><summary>{tool_name}</summary>"
                tool_result += f"<p>{tool_message.content}</p></details>"

        #    print(tool_result)
            return tool_result


        # 3. 组装 tool response
        tool_chain = RunnablePassthrough.assign(tool_result=show_tool_outputs).with_config(run_name="tool_message")


        return tool_chain


    def create_chat_chain(self, llm=None,prompt1=None,prompt2=None,document_variable_names=None, k_queries=1,**kwargs):
        '''
        document_variable_names:list[str],分别是upload_files和rag_file在prompt的占位词
        '''
        # 集成了chat，rag，文档上传的自定义链；根据create_stuff_chain 和create_retrieve_chain 改写


        if not document_variable_names:
            document_variable_names = ["context", "rag_context"]

        query_change_chain1, query_change_chain2 = self.create_query_change_chain(llm,prompt1, prompt2)

        deal_docs_chain = self.create_docs_chain(document_variable_names[0])
        web_search_chain = self.create_web_search_chain(**kwargs)
        rag_chain=self.create_rag_chain(document_variable_names[1])
        tool_chain = self.create_tool_chain(kwargs["tools"])
        _validate_prompt(document_variable_names, self.qa_prompt)
        _document_prompt = DEFAULT_DOCUMENT_PROMPT  # 默认是page_content
        _output_parser = StrOutputParser()


        #     (answer=deal_rag_chain.with_config(run_name="rag_chain"))
        simple_chain = (self.qa_prompt | self.llm
                        #| _output_parser
                        )  # without rag

   #     is_retriever =kwargs["chat_mode"]["is_rag"] or kwargs["chat_mode"]["is_web_search"]
        chain =   RunnableBranch((lambda x:x.get(document_variable_names[0], False), deal_docs_chain), lambda x: x  )| \
            (RunnableBranch((lambda x: x.get(document_variable_names[0], False) or x.get("chat_history",False),query_change_chain1),
                               lambda x:x)
                | RunnableBranch((lambda x :kwargs["chat_mode"]["is_rag"]and kwargs["rag_config"]=="高级搜索", query_change_chain2),
                               lambda x:x)

                | RunnableBranch(
            (lambda x: kwargs["chat_mode"]["is_web_search"], web_search_chain),
            lambda x: x )
                |RunnableBranch(
            (lambda x: kwargs["chat_mode"]["is_rag"], rag_chain),
            lambda x: x  )
                |RunnableBranch((lambda x :kwargs.get("tools",False),tool_chain.with_config(run_name="chat_chain")), lambda x: x)
            .assign(answer=simple_chain))

        return chain

    def answer(self, question, document_variable_names=None,chat_history_LCEL=[], **kwargs):

        print(kwargs["chat_mode"])
        self.param_set(**kwargs)
        # print(f"answer:{sys.getrefcount(self.llm)}")
        document_variable_names = ["context",
                                   "rag_context"] if document_variable_names is None else document_variable_names

        if kwargs["is_abstract"]:  # 文本摘要模式
            upload_files= [] if not kwargs["upload_files"] else kwargs["upload_files"]
            rag_files=[] if not kwargs["rag_files"] else kwargs["rag_files"]
            files = upload_files + rag_files
            abstract_answer = self.abstract_answer(files=files, chat_history_LCEL=[],**kwargs)
            for info, answer in abstract_answer:
                yield info, answer

        else:
            self.get_retriever(**kwargs)
            # print(f"answer:{sys.getrefcount(self.llm)}")
            chat_answer = self.chat_answer(question, document_variable_names, chat_history_LCEL,**kwargs)
            # print(f"answer:{sys.getrefcount(self.llm)}")
            for info, answer in chat_answer:
                yield info, answer




    def get_docs(self,upload_files):
        new_docs = []
        if upload_files:
            loaders = []

            docs = defaultdict(list)
            for file in upload_files:
                file_loader(file, loaders)
            for loader in loaders:
                if loader:
                    file_name = loader[0]
                    doc = loader[1].load()
                    docs[file_name].extend(doc)

            #    file_names = ",".join(list(docs.keys()))
            page_contents = []
            for i, file in enumerate(docs.items()):  #
                file_name = file[0]  # key
                subdocs = file[1]  # value
                page_content = f"<b>文档{i + 1}{file_name}的内容：</b><br>" \
                               + "".join([doc.page_content for doc in subdocs])
                #   num_tokens += self.llm.get_num_tokens(page_content)
                page_contents.append(page_content)
                new_docs.append(Document("<br>".join(page_contents)))
        return new_docs
    def chat_answer(self, question, document_variable_names=None,chat_history_LCEL=[], **kwargs):

        # 5. 构建总链路（Retrieval Chain）
        if not document_variable_names:
            document_variable_names = ["context", "rag_context"]
        chat_chain = self.create_chat_chain(llm=self.llm,document_variable_names=document_variable_names, **kwargs)
       # tool_chain = self.create_tool_chain(kwargs["tools"])
        chain_config = {"input": question}
        if self.model_type in ["HuggingFace","llama_cpp"] and "qwen3" in self.model.lower():
            if kwargs["chat_mode"]["is_reasoning"]:
                chain_config["input"]+="/think"
            else:
                chain_config["input"] += "/no_think"
        chain_config["chat_history"] = chat_history_LCEL if kwargs["is_with_history"] else []
        chain_config["web_context"] = []
        upload_files = kwargs["upload_files"]
        chain_config["tool_result"] = []

        new_docs=self.get_docs(upload_files)
        chain_config[document_variable_names[0]] = new_docs

        chain_config[document_variable_names[1]] = []
        # print(f"chat_answer:{sys.getrefcount(self.llm)}")
        for info, answer in self.stream( chat_chain,  chain_config, document_variable_names,
                                               **kwargs):

            yield info, answer


    def stream(self,  chat_chain,chain_config, document_variable_names, **kwargs):
        answer = ""

        info = "正在输出回答"


        i = 0

        start_reason=True
        start_answer=True
        end_rest=""
        state = {"buffer": "", "in_think": False}
        for chunk in chat_chain.stream(chain_config):
            i = i + 1
            print(f"{i}:{chunk},"
                  )
            ans=""
            web_context = chunk.get("web_context", [])
            rag_context = chunk.get(document_variable_names[1], [])
            tool_result = chunk.get("tool_result", [])
            delta=chunk.get("answer", AIMessageChunk(content=""))
            if web_context:
                ans= f"<b>网络搜索结果：</b><br>{web_context} <br> "
                yield info, ans

            if tool_result:
                ans += f"<b>工具调用：</b><br>{tool_result } <br> "
                yield info, ans
            if rag_context:
                ans= f"<b>参考资料：</b><br>{rag_context} <br> "
                info = f"{self.scores}"
                yield info, ans
                continue
            #深度思考输出只针对部分模型启用：ollama qwen3系列和globals列出的reason_models
            if (kwargs["chat_mode"]["is_reasoning"] or self.model in only_reason_models) and start_reason: #reasoning_content没输出完，接着输出推理内容
                #思考内容两种处理方式<think>在content中，或者出现在additional_kwargs中
                if delta.additional_kwargs.get("reasoning_content", False):
                    delta=delta.additional_kwargs["reasoning_content"]
                    start_reason = False
                    ans +="<b>思考：</b><br>"+delta
                    yield info, ans
                    continue
                else:
                    delta=delta.content
                    delta,state,end_reason,end_rest=replace_think_tag_stream(delta,state) #返回剔除think标签，处理过后的文本
                    ans += delta
                    start_reason = not end_reason
                    # if end_reason:
                    #     ans +="<br>"
                    yield info ,ans
                    continue
            if start_answer:
                if delta.content:#开始输出正式回答
                    delta=delta.content
                    start_answer = False
                    if  not start_reason: #和推理内容换行
                        ans +="<br>"
                    ans += f"<b>回答:</b><br>"
                    if end_rest:#/think留下的残片
                        ans +=end_rest
                        end_rest=""
                    ans += delta
                    yield info, ans
                elif  delta.additional_kwargs.get("reasoning_content",""): #reasoning_content没输出完，接着输出推理内容
                    delta= delta.additional_kwargs["reasoning_content"]
                    ans +=delta
                    yield info, ans
                continue

            delta = delta.content
            ans += delta
            answer += delta
            yield info, ans

        if not kwargs["chat_mode"]["is_rag"]:
            yield "回答已经完成", ""
        # print(f"stream:{sys.getrefcount(self.llm)}")




    def abstract_answer(self, files: list[str],chat_history_LCEL=[], **kwargs):
        '''
             Input:
             1.docs:上传文档（file_path)解析为Document对象,或者选择文件列表（已经向量化的文件），根据Document。metedata["source"]=file_name,筛选
             2.mode:["stuff","map_reduce","refine"] #from langchain.chains.summarize import load_summarize_chain
             内部仍然是v0.1的chain使用方法，为了和新版本同步，这里重新用LCEL实现
             '''
        docs = defaultdict(list)
        for file in files:
            if os.path.exists(file): #上传文档路径
                loaders = []

                file_loader(file, loaders)
                for loader in loaders:
                    if loader:
                        file_name = loader[0]
                        doc = loader[1].load()
                        docs[file_name].extend(doc)
            else:  #文件列表的文件名
                docResults= self.vectordb.get(where={"source":file})
                #print(docResults) #这个对于txt文件的document打印不出来？但是有被检索到
                page_contents=docResults["documents"]
                metadatas=docResults["metadatas"]
                if docs.get(file,False):
                    docs[file].extend([Document(page_content=page_content, metadata=metadata)
                      for page_content,metadata in zip(page_contents, metadatas)])
                else:
                    docs[file]=[Document(page_content=page_content, metadata=metadata)
                      for page_content,metadata in zip(page_contents, metadatas)]

        file_names = ",".join(list(docs.keys()))


        yield "", [f"生成关于文件{file_names}的摘要",f"<b>回答:</b><br>"]
        if kwargs["is_with_history"]:
            for info, ans in self.summarizer.astream_summary(docs,
                                                                      chat_history_LCEL):

                yield info, ans
        else:
            for info, ans in self.summarizer.astream_summary(docs, []):
                yield info, ans

    def get_prompt(self, **kwargs):
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", kwargs["prompts"]),
            ("placeholder", "{chat_history}"),
            ("human", "问题：{input}，输出答案")
        ])



    def extract_answer(self, answer_with_ref_ui: str) -> str:
        split_marker = "<br>"
        return answer_with_ref_ui.split(split_marker)[-1]