import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_together import ChatTogether
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition

from Tavilyfile import query_tavily

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

@tool
def tavily_search(query: str) -> str:
    """Search the internet for latest information."""
    return query_tavily(query)

@tool
def rag_lookup(query: str) -> str:
    """Search company documents and PDFs."""
    return "NONE"

tools = [tavily_search, rag_lookup]

llm_with_tools = ChatTogether(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    together_api_key=TOGETHER_API_KEY,
).bind_tools(tools)

system_prompt = """You are an intelligent assistant. Decide how to answer:
- If the user asks for latest/current info or news, call `tavily_search`.
- If the user asks about the company documents/PDFs, call `rag_lookup`.
- Otherwise, answer directly.
Always give clear, concise answers.
"""

graph = StateGraph(MessagesState)

def chatbot_node(state: MessagesState) -> Dict[str, Any]:
    msgs = [SystemMessage(content=system_prompt)] + state["messages"]
    msg = llm_with_tools.invoke(msgs)
    return {"messages": [msg]}

graph.add_node("chatbot", chatbot_node)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("chatbot")
graph.add_conditional_edges("chatbot", tools_condition)
graph.add_edge("tools", "chatbot")
graph.add_edge("chatbot", END)

compiled_graph = graph.compile()

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    state = {"messages": []}

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        # Add user message
        state["messages"].append(HumanMessage(content=user_input))

        # Invoke the graph
        state = compiled_graph.invoke(state, config=config)

        # Print AI responses
        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                print(f"AI: {msg.content}")
