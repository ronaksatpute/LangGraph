from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int) -> int:
    """Add two numbers"""
    return a + b

@tool   
def multiply(a:int, b:int) -> int:
    """Multiply two numbers"""
    return a * b

@tool
def subtract(a:int, b:int) -> int:
    """Subtract two numbers"""
    return a - b

tools = [add, multiply, subtract]
tool_node = ToolNode(tools)

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def model_call(state:AgentState)->AgentState:
    system_prompt = SystemMessage(content=
    "You are my AI Assistant. Please answer my query to the best of your ability"
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state:AgentState)->str:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tool_node",
        "end": END
    },
)

graph.add_edge("tool_node", "our_agent")

agent = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

user_input = {"messages": [("user", "Add 7 and -9. Also what is 2 * 3? Also, tell me an AI joke")]}
print_stream(agent.stream(user_input,stream_mode="values"))