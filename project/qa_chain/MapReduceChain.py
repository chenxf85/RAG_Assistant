

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import StrOutputParser



import operator
from typing import Annotated, List, Literal, TypedDict

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from prompt.prompt import _validate_prompt
from prompt.prompt import (
    Map_Prompt,
    Reduce_Prompt
)
from copy import deepcopy
class MapReduceChain:
    class OverallState(TypedDict):
        # Notice here we use the operator.add
        # This is because we want combine all the summaries we generate
        # from individual nodes back into one list - this is essentially
        # the "reduce" part

        context: List[Document]
        summaries: Annotated[list, operator.add]
        collapsed_summaries: List[Document]
        final_summary: str
        chat_history: List[tuple]

        # This will be the state of the node that we will "map" all
        # documents to in order to generate summaries

    class SummaryState(TypedDict):
        content: str
        chat_history: List[tuple]

    def __init__(self, llm, token_max=100000, map_prompt: ChatPromptTemplate = Map_Prompt,
                 reduce_prompt: ChatPromptTemplate = Reduce_Prompt):

        # token_max 和模型上下文窗口匹配，避免超出；此外，map_chain生成的初步摘要累积也不能超过token_max，
        # 但是存在情况就是怎么也无法将长文档拆分然后压缩到token_max内，此时需要判断recursion_limit达到上限，或者超时，认为模型无法处理长文档。
        self.llm = llm
        self.token_max = token_max
        self.recursion_limit = 10

        self.map_prompt = map_prompt
        self.reduce_prompt = reduce_prompt

        self.map_chain = self.map_prompt | self.llm | StrOutputParser()
        self.reduce_chain = self.reduce_prompt | self.llm | StrOutputParser()
        self.map_reduce_chain = self.create_chain()

    def stream(self, input_dict: dict[str, any]):  # 为了和langchain astream统一接口"
        # 按照placeholders从input_dict中取出对应值,检查输入是否有效

        new_dict = deepcopy(input_dict)
        new_dict["recursion_limit"] = self.recursion_limit
        new_dict["context"] = input_dict["context"]

        for chunk in self.map_reduce_chain.stream(
                new_dict, stream_mode="messages"
        ):
            print(chunk)
            if (chunk[1]["langgraph_node"] == "generate_final_summary"):
                yield chunk[0].content


    def create_chain(self):
        graph = StateGraph(self.OverallState)
        graph.add_node("generate_summary", self.generate_summary)  # same as before
        graph.add_node("collect_summaries", self.collect_summaries)
        graph.add_node("collapse_summaries", self.collapse_summaries)
        graph.add_node("generate_final_summary", self.generate_final_summary)

        # Edges:
        graph.add_conditional_edges(START, self.map_summaries,
                                #    ["generate_summary"]
                                    )
        graph.add_edge("generate_summary", "collect_summaries")
        graph.add_conditional_edges("collect_summaries", self.should_collapse)
        graph.add_conditional_edges("collapse_summaries", self.should_collapse)
        #  graph.add_edge("collect_summaries", "generate_final_summary")  直连就是不判断是否大于num_token直接输出
        graph.add_edge("generate_final_summary", END)
        return graph.compile()

    def length_function(self, documents: List[Document]) -> int:
        """Get number of tokens for input context."""
        try:
            if hasattr(self.llm, "get_num_tokens"):

                return sum(self.llm.get_num_tokens(doc.page_content) for doc in documents)
            else:
                return int(sum(len(doc.page_content) for doc in documents)*0.4)
        except Exception as e:
            print(f"Error in length_function: {e}")
            return int(sum(len(doc.page_content) for doc in documents)*0.4)

    # This will be the overall state of the main graph.
    # It will contain the input document contents, corresponding
    # summaries, and a final summary.

    # Here we generate a summary, given a document
    def generate_summary(self, state: SummaryState) -> OverallState:

        # Remove the content key to avoid confusion
        response = self.map_chain.invoke({"context":state["content"],"chat_history":state["chat_history"]})
        return {"summaries": [response]}

    # Here we define the logic to map out over the documents
    # We will use this an edge in the graph
    def map_summaries(self, state: OverallState):
        # We will return a list of `Send` objects
        # Each `Send` object consists of the name of a node in the graph
        # as well as the state to send to that node

        page_contents = [doc.page_content for doc in state["context"]]
        print(f"现在有多少；{len(page_contents)}")

        new_state = state.copy()
        new_state.pop("context", None)
        return [
            Send("generate_summary", {**new_state, **{"content": content}}) for content in page_contents
        ]

    def collect_summaries(self, state: OverallState):
        return {
            "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
        }

    # Add node to collapse summaries
    def collapse_summaries(self, state: OverallState):
        doc_lists = split_list_of_docs(
            state["collapsed_summaries"], self.length_function, self.token_max
        )
        results = []
        for doc_list in doc_lists:
            results.append(acollapse_docs(doc_list, self.reduce_chain.ainvoke,chat_history=state["chat_history"]))

        return {"collapsed_summaries": results}

    # This represents a conditional edge in the graph that determines
    # if we should collapse the summaries or not
    def should_collapse(self,
                        state: OverallState,
                        ) -> Literal["collapse_summaries", "generate_final_summary"]:
        num_tokens = self.length_function(state["collapsed_summaries"])
        print(f"Checking collapse: {num_tokens} tokens\n")
        if num_tokens > self.token_max:
            return "collapse_summaries"
        else:
            return "generate_final_summary"

    # Here we will generate the final summary

    def generate_final_summary(self, state: OverallState):
        response = self.reduce_chain.invoke({"summaries": state["collapsed_summaries"],"chat_history":state["chat_history"]})
        return {"final_summary": response}

