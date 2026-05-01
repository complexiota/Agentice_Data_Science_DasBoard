import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from src.agent.tools import analyze_data, clean_data_tool, train_model_tool, get_data_summary

def get_agent():
    """Creates and returns the LangGraph ReAct agent and its memory checkpointer."""
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")
        
    llm = ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.1
    )
    
    tools = [
        analyze_data,
        clean_data_tool,
        train_model_tool,
        get_data_summary
    ]
    
    memory = MemorySaver()
    agent = create_react_agent(llm, tools, checkpointer=memory)
    
    return agent
