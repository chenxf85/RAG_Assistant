

from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser


from typing import Annotated, List, Literal, TypedDict


from langchain_core.documents import Document

from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig

from prompt.prompt import _validate_prompt
from prompt.prompt import (
    Map_Prompt,
    Reduce_Prompt,
    Stuff_Prompt,
    Refine_Prompt,
    Initial_Refine_Prompt,
)

class RefineChain:
    class State(TypedDict):
        context: List[Document]
        index: int
        summary: str
        chat_history: List[tuple]

    def __init__(self, llm, initial_refine_prompt: ChatPromptTemplate = Initial_Refine_Prompt,
                 refine_prompt: ChatPromptTemplate = Refine_Prompt):
        self.llm = llm
        self.initial_refine_prompt = initial_refine_prompt
        self.refine_prompt = refine_prompt

        # Initial summary
        self.initial_refine_prompt = initial_refine_prompt
        self.initial_summary_chain = initial_refine_prompt | llm | StrOutputParser()
        self.refine_prompt = refine_prompt

        self.refine_summary_chain = refine_prompt | llm | StrOutputParser()
        self.refine_chain = self.create_chain()

    def create_chain(self):
        graph = StateGraph(self.State)
        graph.add_node("generate_initial_summary", self.generate_initial_summary)
        graph.add_node("refine_summary", self.refine_summary)

        graph.add_edge(START, "generate_initial_summary")
        graph.add_conditional_edges("generate_initial_summary", self.should_refine)
        graph.add_conditional_edges("refine_summary", self.should_refine)
        return graph.compile()

    def stream(self, input_dict):


        context_num=input_dict["context_num"]
        for step in self.refine_chain.stream(input_dict, stream_mode="messages"):
            # 只输出最后一个step的summary

            print(step)

            if  step[1]['langgraph_step']==context_num: #只输出最终摘要
                yield step[0].content


    def generate_initial_summary(self, state: State, config: RunnableConfig):
        new_dict = state.copy()

        new_dict["context"] = state["context"][0].page_content
        new_dict["chat_history"] = state["chat_history"]
        summary = self.initial_summary_chain.invoke(
            new_dict,
            config,
        )
        return {"summary": summary, "index": 0}

    # And a node that refines the summary based on the next document

    def refine_summary(self, state: State, config: RunnableConfig):
        content = state["context"][state["index"]].page_content
        summary = self.refine_summary_chain.invoke(
            {"existing_summary": state["summary"], "new_text": content,"chat_history": state["chat_history"]},
            config,
        )

        return {"summary": summary, "index": state["index"] + 1}

    # Here we implement logic to either exit the application or refine
    # the summary.
    def should_refine(self, state: State) -> Literal["refine_summary", END]:

        print(f"文本数量{len(state["context"])}")
        if state["index"]+1 >= len(state["context"]):
            return END
        else:
            return "refine_summary"
