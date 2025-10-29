from typing import List, Literal, AsyncIterator, Iterator

from langchain.prompts import  ChatPromptTemplate


from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter

from typing import Annotated, List, Literal, TypedDict

from langchain_core.documents import Document


from prompt.prompt import (
    Map_Prompt,
    Reduce_Prompt,
    Stuff_Prompt,
    Refine_Prompt,
    Initial_Refine_Prompt,
)
from qa_chain.MapReduceChain import MapReduceChain
from qa_chain.RefineChain import RefineChain

SummaryMethod = Literal["stuff", "map_reduce", "refine"]


class LCELBrowser:
    '''
    文件浏览链，可以阅读用户上传的文档
    当未传入prompt，则默认是一个文本摘要器。

    '''

    def __init__(self, llm=None, method: SummaryMethod = "stuff", token_max=100000, recursion_limit=10,
                 document_variable_name="context", map_prompt=Map_Prompt, reduce_prompt=Reduce_Prompt,
                 stuff_prompt=Stuff_Prompt, refine_prompt=Refine_Prompt, initial_refine_prompt=Initial_Refine_Prompt):
        self.llm = llm
        self.method = method
        self.token_max = token_max
        self.recursion_limit = recursion_limit
        self.document_variable_name = document_variable_name
        # 创建各种提示模板
        self.map_prompt = map_prompt
        self.reduce_prompt = reduce_prompt

        self.stuff_prompt = stuff_prompt
        self.refine_prompt = refine_prompt

        self.initial_refine_prompt = initial_refine_prompt

        self.max_refine = 5
        # 创建基础链
        self._create_chains()

    def _create_chains(self):
        """创建不同的摘要处理链"""

        self.stuff_chain = create_stuff_documents_chain(self.llm, self.stuff_prompt,
                                                        )
        self.map_reduce_chain = create_map_reduce_chain(self.llm, self.token_max, self.map_prompt, self.reduce_prompt,
                                                      )
        self.refine_chain = create_refine_chain(self.llm, self.initial_refine_prompt, self.refine_prompt,
                                                )

    def _get_chain(self):
        if self.method == "stuff":
            return self.stuff_chain
        elif self.method == "map_reduce":
            return self.map_reduce_chain
        elif self.method == "refine":
            return self.refine_chain
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    def length_function(self, docs: str) -> int:
        """Get number of tokens for input context."""
        try:
            if hasattr(self.llm, "get_num_tokens"):
                return self.llm.get_num_tokens(docs)
            else:
                return int(len(docs)*0.4)
        except Exception as e:
            print(f"Error in length_function: {e}")
            return int(len(docs)*0.4)
    def astream_summary(self, docs: dict[str, list[Document]],  chat_history_LCEL=[]) -> \
            Iterator:
        """异步流式生成文档摘要"""

        for i, file in enumerate(docs.items()):  #
            #支持多文档摘要
            file_name = file[0]
            subdocs = file[1]
            print(len(subdocs))
            page_contents = f"<b>文档{file_name}的内容：</b><br>"  \
                           + "\n\n".join([doc.page_content for doc in subdocs])
            new_docs = [Document(page_contents)]
            length=self.length_function(page_contents)
            print(length)
            if  length< self.token_max * 0.7: #粗略估计token
                self.set_params(method="stuff")
                split_docs = new_docs
            else:
                text_splitter= RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=self.token_max * 0.7, chunk_overlap=0
                )
                split_docs = text_splitter.split_documents(new_docs)
                print("文档数量"+str(len(split_docs)))
                if (len(split_docs)) < self.max_refine:
                    self.set_params(method="refine")
                else:
                    self.set_params(method="map_reduce")

            chain = self._get_chain()
            isFirst=True
            for answer in chain.stream({"context": split_docs, "chat_history": chat_history_LCEL,"context_num":len(split_docs)}):
                if isFirst:
                    i=i+1
                    answer= f"<b>{i}.{file_name}文档摘要：</b><br>"+answer
                    isFirst=False
                yield f"正在生成{file_name}的回答", answer

            yield f"完成{file_name}的回答", "<br>-------------------------------<br>"

    # 阅读文档
    def set_params(self, llm=None, method: SummaryMethod = None):
        """设置摘要方法"""

        self.llm = llm if llm else self.llm
        self.method = method if method else self.method


def create_map_reduce_chain(llm, token_max=100000, map_prompt: ChatPromptTemplate = Map_Prompt,
                            reduce_prompt: ChatPromptTemplate = Reduce_Prompt):
    return MapReduceChain(llm, token_max, map_prompt, reduce_prompt)


def create_refine_chain(llm, initial_refine_prompt: ChatPromptTemplate = Initial_Refine_Prompt,
                        refine_prompt: ChatPromptTemplate = Refine_Prompt):
    return RefineChain(llm, initial_refine_prompt, refine_prompt)
