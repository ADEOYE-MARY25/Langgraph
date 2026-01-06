# Imports
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from IPython.display import Image, display
from typing import Literal
import os




# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found! Please set it in your .env file.")


# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,  # Lower temperature for more precise tool usage
    api_key=openai_api_key
)

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Use this tool when you need to perform calculations.
    
    Args:
        expression: A mathematical expression like "2 + 2" or "15 * 37"
        
    Returns:
        The calculated result as a string
        
    Examples:
        - "2 + 2" returns "4"
        - "100 / 5" returns "20.0"
        - "2 ** 10" returns "1024"
    """
    try:
        # Evaluate the expression safely
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error calculating: {str(e)}"


# # Test the calculator tool
# result = calculator.invoke({"expression": "123 * 456"})
# print(f"123 * 456 = {result}")

# result2 = calculator.invoke("2 ** 10")

@tool
def text_analyzer(text: str) -> str:
    """
    Analyze text and return statistics about it.
    Use this tool when you need to analyze or count things in text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Statistics about the text (characters, words, sentences)
        
    Examples:
        - "Hello world" returns character count, word count, etc.
    """
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    
    return f"""Text Analysis:
- Characters: {char_count}
- Words: {word_count}
- Sentences: {sentence_count}
- First 50 chars: {text[:50]}..."""

# Test the text analyzer
# test_text = "Hello! This is a test. How are you today?"
# result = text_analyzer.invoke({"text": test_text})


# Create a list of tools
tools = [calculator, text_analyzer]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Test: Does LLM decide to call calculator?
# response2 = llm_with_tools.invoke([HumanMessage(content="What is 234 * 567?")])

# System prompt that encourages tool usage
sys_msg = SystemMessage(content="""You are a helpful assistant on food Education with access to tools.
Only use tools when necessary - for simple conversation food educational questions, answer directly.
for non-educational related question - say i can only assist you on food educational question. 
                        
When asked to perform calculations, use the calculator tool.
When asked to analyze text, use the text_analyzer tool.
shzhjhdz
                        
                        """)

def assistant(state: MessagesState) -> dict:
    """
    Assistant node - decides whether to use tools or answer directly.
    """
    messages = [sys_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """
    Decide next step based on last message.
    
    If LLM called a tool → go to 'tools' node
    If LLM provided final answer → go to END
    """
    last_message = state["messages"][-1]
    
    # Check if LLM made tool calls
    if last_message.tool_calls:
        return "tools"
    
    # No tool calls - we're done
    return "__end__"


builder1 = StateGraph(MessagesState)

# Add nodes
builder1.add_node("assistant", assistant)
builder1.add_node("tools", ToolNode(tools))  # ToolNode executes tool calls automatically

# Define edges
builder1.add_edge(START, "assistant")
builder1.add_conditional_edges(
    "assistant",
    should_continue,
    {"tools": "tools", "__end__": END}
)
builder1.add_edge("tools", "assistant")  # After tools, go back to assistant

# Add memory
memory1 = MemorySaver()
agent = builder1.compile(checkpointer=memory1)


# Visualize the agent graph
try:
    display(Image(agent.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f"Could not display graph: {e}")
    print("Graph structure: START → assistant → [conditional] → tools → assistant → END")


# Helper function
def run_agent(user_input: str, thread_id: str = "user1"):
    """
    Run the agent and display the conversation.
    """
    print(f"\n{'='*70}")
    print(f"👤 User: {user_input}")
    print(f"{'='*70}\n")
    
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    
    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            continue  # Already printed
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                print(f"🤖 Agent: [Calling tool: {message.tool_calls[0]['name']}]")
            else:
                print(f"🤖 Agent: {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"🔧 Tool Result: {message.content[:100]}..." if len(message.content) > 100 else f"🔧 Tool Result: {message.content}")
    


while True:
    query=input("user:")
    response= run_agent(query, thread_id="4001")
    response


# # Get full message history
# result = agent.invoke(
#     {"messages": [HumanMessage(content="What is 15 * 25?")]},
#     config={"configurable": {"thread_id": "inspect_session"}}
# )

# print("\n📋 FULL MESSAGE HISTORY:\n")
# for i, msg in enumerate(result["messages"], 1):
#     print(f"{i}. {type(msg).__name__}")
#     if isinstance(msg, AIMessage) and msg.tool_calls:
#         print(f"   Tool Call: {msg.tool_calls[0]['name']}({msg.tool_calls[0]['args']})")
#     elif isinstance(msg, ToolMessage):
#         print(f"   Content: {msg.content}")
#     elif hasattr(msg, 'content'):
#         print(f"   Content: {msg.content}")
#     print()

#     @tool
#     def coin_flip() -> str:
#         """
#         Flip a coin and return heads or tails.
        
#         Use this when the user wants a random choice or coin flip.
        
#         Returns:
#             Either "Heads" or "Tails"
#         """
#         import random
#         return random.choice(["Heads", "Tails"])

# # Rebuild agent with 3 tools
# tools_v2 = [calculator, text_analyzer, coin_flip]
# llm_with_tools_v2 = llm.bind_tools(tools_v2)

# def assistant_v2(state: MessagesState) -> dict:
#     messages = [sys_msg] + state["messages"]
#     response = llm_with_tools_v2.invoke(messages)
#     return {"messages": [response]}

# builder_v2 = StateGraph(MessagesState)
# builder_v2.add_node("assistant", assistant_v2)
# builder_v2.add_node("tools", ToolNode(tools_v2))
# builder_v2.add_edge(START, "assistant")
# builder_v2.add_conditional_edges("assistant", should_continue, {"tools": "tools", "__end__": END})
# builder_v2.add_edge("tools", "assistant")

# agent_v2 = builder_v2.compile(checkpointer=MemorySaver())

# # Test coin flip
# result = agent_v2.invoke(
#     {"messages": [HumanMessage(content="Flip a coin for me!")]},
#     config={"configurable": {"thread_id": "coin_session"}}
# )

# for msg in result["messages"]:
#     if isinstance(msg, AIMessage) and not msg.tool_calls:
#         print(f"🤖 Agent: {msg.content}")






