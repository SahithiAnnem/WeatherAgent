import os
from typing import TypedDict, List
from datetime import datetime, timedelta

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
# lamgraph is a library within langchain.. so it uses msot of the langchain components to function
from pydantic import BaseModel, Field

# --- START: MODIFIED SECTION ---
# Replace "YOUR_ACTUAL_GEMINI_API_KEY" with your real API key
# For security, consider using environment variables for production deployments.
GEMINI_API_KEY = "AIzaSyCdSV-j0Hd1bdBF87f9jdgUTj4ch19snm8"
    

# 1. Define the Agent's State
class AgentState(TypedDict):
    """
    Represents the state of our agent in LangGraph.
    It will contain a list of messages (conversation history).
    """
    messages: List[BaseMessage]

# 2. Define a data a schema for Tool input (with Pydantic for input validation)
class WeatherInput(BaseModel):
    """Input for the get_current_weather tool."""
    location: str = Field(description="The city and state, e.g., San Francisco, CA")
    date: str = Field(description="The date for which to get the weather, in YYYY-MM-DD format (e.g., 2025-06-19)")

@tool(args_schema=WeatherInput)
def get_current_weather(location: str, date: str) -> str:
    """
    Fetches the current weather for a specified location and date.
    This is a mock tool for demonstration purposes.
    """
    print(f"\n--- Mock Tool Call: get_current_weather(location='{location}', date='{date}') ---")
    
    
    current_date = datetime(2025, 6, 19).strftime("%Y-%m-%d") # Hardcoded for consistent demo output
    # If you want it to always be today's date from the machine's perspective, use:
    # current_date = datetime.now().strftime("%Y-%m-%d")

    if "san francisco" in location.lower():
        if date == current_date:
            return f"The weather in {location} today ({date}) is sunny with a temperature of 70°F."
        else:
            return f"I don't have historical/future weather data for {location} on {date}. But San Francisco is generally mild."
    elif "new york" in location.lower():
        if date == current_date:
            return f"The weather in {location} today ({date}) is cloudy with a temperature of 65°F with a chance of rain."
        else:
             return f"I don't have historical/future weather data for {location} on {date}. New York can be unpredictable."
    elif "mason, ohio" in location.lower():
        if date == current_date:
            return f"The weather in {location} today ({date}) is 75°F and partly cloudy."
        else:
            return f"I don't have historical/future weather data for Mason, Ohio on {date}."
    else:
        return f"Sorry, I don't have weather information for {location}."

# Initialize the Gemini LLM
# Pass the API key directly to the google_api_key parameter
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5, google_api_key=GEMINI_API_KEY)

# Bind the tool to the LLM
llm_with_tools = llm.bind_tools([get_current_weather])


# 3. Define Nodes:

def call_llm(state: AgentState):
    """Node to invoke the LLM with dynamic tool routing based on user input."""
    messages = state["messages"]
    user_input = messages[-1].content if isinstance(messages[-1], HumanMessage) else ""
    dynamic_llm = get_llm_for_input(user_input)

    print(f"\n--- Node: call_llm (Invoking Gemini) ---")
    # for i, msg in enumerate(messages):
     #   print(f"  Message {i}: type={type(msg)}, content='{getattr(msg, 'content', 'N/A')}', tool_calls='{getattr(msg, 'tool_calls', 'N/A')}', tool_call_id='{getattr(msg, 'tool_call_id', 'N/A')}'")

    response = dynamic_llm.invoke(messages)
    return {"messages": [response]}

def call_tool(state: AgentState):
    """
    Node to execute a tool call requested by the LLM.
    """
    print(f"\n--- Node: call_tool (Executing tool call from Gemini) ---")
    # print(f"\n--- Node: calling tool")
    messages = state["messages"]
    last_message = messages[-1] # latest msg
    #print(f"\n--- Last Message: {last_message} ---")
    
    
    
    tool_outputs = []
    for tool_call in last_message.tool_calls:
        if tool_call['name'] == "get_current_weather":
            try:
                output = get_current_weather.invoke(tool_call['args'])
                tool_outputs.append(ToolMessage(content=str(output), tool_call_id=tool_call['id']))
            except Exception as e:
                tool_outputs.append(ToolMessage(content=f"Error executing tool {tool_call.name}: {e}", tool_call_id=tool_call.id))
        else:
            tool_outputs.append(ToolMessage(content=f"Unknown tool: {tool_call.name}", tool_call_id=tool_call.id))
            
    return {"messages": state["messages"]+tool_outputs}

# 4. Define the Conditional Edge (Router)

def should_continue(state: AgentState) -> str:
    """
     agent should continue by calling a tool
    or finish by responding directly.
    """
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        print("\n--- Decision: CONTINUE (Gemini suggested tool calls) ---")
        return "continue"
    else:
        print("\n--- Decision: END ---")
        return "end"

# Helper: dynamically bind LLM with the appropriate tool behavior
def get_llm_for_input(user_input: str):
    if "weather" in user_input.lower() or "temperature" in user_input.lower():
        return llm.bind_tools([get_current_weather], tool_choice="get_current_weather")
    else:
        return llm.bind_tools([get_current_weather], tool_choice="auto")
# 5. Assemble the Graph:

workflow = StateGraph(AgentState)
workflow.add_node("llm", call_llm)
workflow.add_node("tool", call_tool)
workflow.set_entry_point("llm")
workflow.add_conditional_edges(
    "llm",
    should_continue,
    {
        "continue": "tool",
        "end": END
    }
)
workflow.add_edge("tool", "llm")
app = workflow.compile()
#langgraph studio is a part of langsmith platform
# Visualize the graph (optional, requires graphviz: pip install graphviz)
try:
    from IPython.display import Image, display
    image_data = app.get_graph().draw_mermaid_png()
    image_path = "langgraph_workflow.png" # Define the filename
    with open(image_path, "wb") as f:
        f.write(image_data)
    print(f"\n--- Graph visualization saved to {os.path.abspath(image_path)} ---")
    # The following display() will still print the object in a basic terminal
    # but would render in a Jupyter environment.
    display(Image(image_data)) 
except ImportError:
    print("\nInstall graphviz to visualize the graph: pip install graphviz")
    print("You might also need to install the graphviz system package.")

print("\n--- Agent Initialized ---")

# --- Test Runs ---

# Using the context date: Thursday, June 19, 2025
TODAY_DATE_STR = "2025-06-19"
TOMORROW_DATE_STR = "2025-06-20" # Explicitly setting based on context

print("\n--- Running Agent (Gemini): Weather in San Francisco Today ---")
inputs_sf = {"messages": [HumanMessage(content=f"What's the weather in San Francisco today ({TODAY_DATE_STR})?")]}
for s in app.stream(inputs_sf):
    print(s)


print("\n--- Running Agent (Gemini): Weather in New York Tomorrow ---")
inputs_ny = {"messages": [HumanMessage(content=f"What is the current weather in New York on 2025-06-20 using your weather tool?")]}
for s in app.stream(inputs_ny):
    print(s)

print("\n--- Running Agent (Gemini): Weather in Mason, Ohio (Current Location Context) ---")
inputs_mason = {"messages": [HumanMessage(content=f"What's the weather like in Mason, Ohio today ({TODAY_DATE_STR})?")]}
for s in app.stream(inputs_mason):
    print(s)

print("\n--- Running Agent (Gemini): General question (no tool needed) ---")
inputs_general = {"messages": [HumanMessage(content="Tell me a fun fact about giraffes.")]}
for s in app.stream(inputs_general):
    print(s)
