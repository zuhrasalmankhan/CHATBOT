import os
from dotenv import load_dotenv
load_dotenv()
from langchain_together import ChatTogether
from langchain.chat_models import init_chat_model
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from Tavilyfile import query_tavily  # your Tavily function
#from RAGfile import query_RAG      # your RAG function


TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-oss-20b")


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def tavily_search(question: str) -> str:
    """Search the web for latest/current information."""
    return query_tavily(question)

@tool
def rag_search(question: str) -> str:
    """Retrieve information from internal PDF knowledge base."""
    return "Done"
    #return query_RAG(question)


tools = [tavily_search, rag_search]

system_prompt = """
You are an intelligent assistant. Decide how to answer:
- If the user asks for latest/current info or news, call `tavily_search`.
- If the user asks about the company documents/PDFs, call `rag_search`.
- Otherwise, answer directly.
Always give clear, concise answers.
"""


llm = ChatTogether(model="openai/gpt-oss-20b", temperature=0.7)
llm_with_tools = llm.bind_tools(tools)

def chatbot_node(state: State):
    # Add system prompt at the start
    msgs = [{"role": "system", "content": system_prompt}] + state["messages"]
    msg = llm_with_tools.invoke(msgs)
    return {"messages": [msg]}

memory = MemorySaver()
builder = StateGraph(State)

builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(checkpointer=memory)

print(builder.nodes)   # shows all nodes
print(builder.edges)   # shows all edges



# config = {"configurable": {"thread_id": "agent_thread"}}

# # Ask for latest news (should hit Tavily)
# state = graph.invoke(
#     {"messages":[{"role":"user","content":"What are the latest updates about AI regulation?"}]},
#     config=config
# )
# print(state["messages"][-1].content)

# # Ask for PDF info (should hit RAG)
# state = graph.invoke(
#     {"messages":[{"role":"user","content":"What is written in section 3.4 of our annual report?"}]},
#     config=config
# )
# print(state["messages"][-1].content)


