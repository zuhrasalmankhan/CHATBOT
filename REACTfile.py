from langchain_together import ChatTogether
from langchain.agents import initialize_agent, AgentType
from langchain.agents import load_toolsu

llm = ChatTogether(model="mistral-7b-instruct")

#
tools = load_tools([], llm=llm)  


agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Step 4: Ask a question
result = agent.run("What is the capital of Pakistan?")
print(result)
