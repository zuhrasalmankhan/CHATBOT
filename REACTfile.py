import os
from dotenv import load_dotenv
load_dotenv()

from langchain_together import ChatTogether
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage

from Tavilyfile import query_tavily  # updated Tavily function

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-oss-20b")


class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def tavily_search(question: str) -> str:
    """Search the web for latest/current information."""
    result = query_tavily(question)
    print("[DEBUG Tavily output]:", result)  # <-- see exactly what it returns
    return result

@tool
def rag_search(question: str) -> str:
    """Retrieve information from internal PDF knowledge base."""
    return "Done"  # stub for now


tools = [tavily_search, rag_search]

system_prompt = """
You are an intelligent assistant. Decide how to answer:
- If the user asks for latest/current info or news, call `tavily_search`.
- If the user asks about the company documents/PDFs, call `rag_search`.
- Otherwise, answer directly.
Always give clear, concise answers.
"""

llm = ChatTogether(model=MODEL_NAME, temperature=0.7)
llm_with_tools = llm.bind_tools(tools)

def chatbot_node(state: State):
    msgs = [SystemMessage(content=system_prompt)] + state["messages"]
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
print(builder.nodes)
print(builder.edges)

config = {"configurable": {"thread_id": "agent_thread"}}

state = graph.invoke(
    {"messages": [{"role": "user", "content": "What are the latest updates about AI regulation?"}]},
    config=config
)
print(state["messages"][-1].content)
