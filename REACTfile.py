import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from RAGfile import RAGTool

# Gemini wrapper
from langchain_google_genai import ChatGoogleGenerativeAI  

from Tavilyfile import query_tavily

load_dotenv()

GEMINI_API_KEY = os.getenv("Gemini_API")
rag_tool = RAGTool("netsol_report.pdf")  
# -------------------- Tools -------------------- #
@tool
def tavily_search(query: str) -> str:
    """Search for the latest information using Tavily API and return a brief summary."""
    result = query_tavily(query)
    return f"Latest info: {result[:300]}..."

@tool
def rag_lookup(query: str) -> str:
    """Retrieve relevant information from company PDFs."""
    results = rag_tool.retrieve(query)
    if not results:
        return "No relevant content found."
    return "Top match: " + results[0]["text"][:300]


tools = [tavily_search, rag_lookup]

# -------------------- LLM with Tools -------------------- #
llm_with_tools = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # lighter model, higher quota
    google_api_key=GEMINI_API_KEY,
).bind_tools(tools)


# -------------------- System Prompt -------------------- #
system_prompt = """
You are a helpful and concise AI assistant.

Decision rules for tool use:
- If the user asks for the latest/current information or news, call `tavily_search`.
- If the user asks about the contents of Netsol_report.pdf, call `rag_lookup`.
- If `rag_lookup` returns no relevant information, answer directly based on your knowledge.
- Otherwise, answer directly without using a tool.

Response guidelines:
- Always answer clearly and directly.
- Summarize tool results in 2–3 sentences.
- Do not repeat the question unless necessary.
- If you don’t know something, say so briefly.
- When the user shares personal info (like their name), acknowledge it naturally (e.g., "Nice to meet you, Zee!").
"""




# -------------------- Graph Setup -------------------- #
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

# -------------------- CLI Loop -------------------- #
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    state = {"messages": []}

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        state["messages"].append(HumanMessage(content=user_input))
        state = compiled_graph.invoke(state, config=config)

        # Only print unique AI messages
        seen = set()
        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                if msg.content.strip() and msg.content not in seen:
                    print(f"AI: {msg.content.strip()}")
                    seen.add(msg.content)
