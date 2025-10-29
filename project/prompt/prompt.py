from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
)
from typing import Optional,Union
Map_Prompt = ChatPromptTemplate.from_messages(
    [("system", """请为以下内容生成一个简洁的摘要:\\n\\n
                      {context}
                      简洁摘要:"""
      ),
     ("placeholder", "{chat_history}")
     ]
)
Reduce_Prompt = ChatPromptTemplate.from_messages(
    [("human", """以下是一系列文档的摘要:\\n\\n
                   {summaries} \\n\\n
                  将这些摘要合并成一个连贯的最终摘要: """
      ),
     ("placeholder", "{chat_history}")
     ]
)
Stuff_Prompt = ChatPromptTemplate.from_messages(
    [("human", """请为以下内容生成摘要:\\n\\n
                      {context}
                      摘要:"""
      ),
     ("placeholder", "{chat_history}")
     ]
)
Refine_Prompt = ChatPromptTemplate.from_messages(
    [("human",
      """这是已有的摘要:
                {existing_summary}

                请基于以下新文本完善和改进上述摘要:
                {new_text}

                改进后的摘要:"""
      ),
     ("placeholder", "{chat_history}")
     ]
)
Initial_Refine_Prompt = ChatPromptTemplate.from_messages(
    [("human", """请为以下文本生成初始摘要:
            {context}

            初始摘要:"""
      ),
     ("placeholder", "{chat_history}")
     ]
)
#
# default_template= {"chat":"""你是一个友好且智慧的AI助手，能够根据上下文准确回答用户的问题，并提供详细的解释。
# 请根据上下文回答最后的问题。如果遇到复杂的问题，请你按照步骤逐步回答，在回答中展现你的逻辑性并引导用户进行多轮对话\n\n""",
#                            "rag":"""使用上下文来回答最后的问题。
# 如果你不知道答案，就说你不知道，不要试图编造答案。
# 尽量使答案详细清晰。总是在回答的最后说“谢谢你的提问！”。
# 与问题有关的上下文如下：
# {context}
#                                     """,
#                             "react":""
#                     }

default_template= """你是一个AI助手，能够根据上下文准确回答用户的问题，尽量使答案详细清晰。
如果遇到复杂的问题，请你按照步骤逐步回答，在回答中展现你的逻辑性并引导用户进行下一轮对话。
如果文档内容，知识库和网络搜索结果不能回答问题，可以尝试使用工具。如果最后仍然不知道答案，就说你不知道，不要试图编造答案。
以下是你可以使用的信息：
1.用户可能已经上传了文档如下（需要在输出中简单对文档进行概要）:
{context}
如果为空，则可以忽略
2.根据用户提问在数据库中检索到了相关信息：
{rag_context} 
3.下面是互联网搜索的结果：
{web_context}。
4.下面是工具调用的结果：
{tool_result}
"""


def _validate_prompt(document_variable_names:Union[list[str],str], prompt:ChatPromptTemplate) -> None:
    if isinstance(document_variable_names, str):
        document_variable_names=[document_variable_names]
    for document_variable_name in document_variable_names:
        if document_variable_name not in prompt.input_variables:
            raise ValueError(
                f"Prompt must accept {document_variable_name} as an input variable. "
                f"Received prompt with input variables: {prompt.input_variables}"
            )