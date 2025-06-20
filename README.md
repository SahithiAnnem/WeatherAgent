# Pydantic Weather Agent with LangGraph and Gemini

This project demonstrates a conversational agent built using Python, Langchain (specifically LangGraph), Pydantic, and Google's Gemini large language model. The agent can understand user queries, determine if a tool is needed (in this case, to get weather information), validate tool inputs using Pydantic, execute the tool, and provide a response.

## Overview

The agent is designed to:
1.  Engage in a conversation.
2.  Identify when a query requires fetching weather information.
3.  Use a (mock) `get_current_weather` tool to retrieve this information.
4.  Leverage Pydantic to define and validate the inputs (location and date) for the weather tool.
5.  Utilize Google's Gemini model for natural language understanding and response generation.
6.  Employ LangGraph to define and manage the agent's execution flow (e.g., deciding whether to call the LLM, a tool, or end the conversation).

## Technologies Used

*   **Python**: The core programming language.
*   **Langchain**: A framework for developing applications powered by language models.
    *   **LangGraph**: A Langchain library module for building stateful, multi-actor applications with LLMs, used here to define the agent's workflow as a graph.
    *   `langchain_core`: For core abstractions like messages and tools.
    *   `langchain_google_genai`: For integration with Google's Gemini models.
*   **Pydantic**: A data validation library used to define the expected input schema for our `get_current_weather` tool, ensuring the LLM provides data in the correct format.
*   **Google Gemini API**: The large language model (LLM) used for understanding queries and generating responses. The script uses `gemini-1.5-flash`.
*   **Requests**: (Implicitly used by Langchain or could be used directly for API calls, though your `pydantic_agent.py` uses the Langchain integration).
*   ## Why LangGraph for this Agent?

LangGraph is a powerful extension of Langchain specifically designed for building stateful, multi-actor applications, including complex agents. Here's why it's a great fit for this Pydantic-enabled weather agent:

*   **State Management**: LangGraph excels at managing the agent's state throughout the conversation. In our `pydantic_agent.py`, the `AgentState` (a `TypedDict`) explicitly defines what information (like the list of `messages`) is passed between different steps (nodes) of the agent's operation. This makes it easy to track the conversation history and context.

*   **Cyclical Workflows**: Conversational agents often require cyclical workflows – for example, an LLM call might lead to a tool call, and the result of that tool call is then fed back to the LLM for a final response. LangGraph is built to handle these cycles naturally. Our graph has an edge from the `tool` node back to the `llm` node, allowing for this iterative refinement. This is a key advantage over simpler chain-like structures (like basic LCEL) which are typically acyclic.

*   **Explicit Control Flow**: The graph structure, with its defined nodes (`llm`, `tool`) and edges (including the `conditional_edges` based on `should_continue`), makes the agent's decision-making process very explicit and easy to understand. You can clearly see how the agent moves from one state to another.

*   **Modularity and Reusability**: Each node in a LangGraph (e.g., `call_llm`, `call_tool`) is a regular Python function. This promotes modularity, making it easier to develop, test, and potentially reuse these components in other agents or applications.

*   **Robust Tool Integration**: LangGraph seamlessly integrates with Langchain's tool-using capabilities. The `llm_with_tools` binding and the `call_tool` node demonstrate how the agent can decide to use a tool, pass validated arguments (thanks to Pydantic), execute it, and incorporate the results back into the conversation.

*   **Debugging and Observability**:
    *   **Streaming**: The `app.stream()` method allows you to see the outputs of each step in the graph as they happen, which is invaluable for debugging and understanding the agent's internal state changes.
    *   **Visualization**: LangGraph can generate visual representations of the agent's workflow (like the `langgraph_workflow.png` your script creates). This visual aid helps in understanding the agent's logic and control flow at a glance.
    *   **LangSmith Integration (Beyond this script)**: While not explicitly implemented in this basic example, LangGraph is designed to integrate tightly with LangSmith, a platform for debugging, testing, evaluating, and monitoring LLM applications. This provides even deeper insights into agent behavior in more complex scenarios.

*   **Flexibility for Complex Agents**: While this weather agent is relatively simple, LangGraph provides the building blocks for creating much more sophisticated agents, including those that might involve multiple LLMs, human-in-the-loop steps, or interactions between several specialized "sub-agents."

In essence, LangGraph provides the structure and control needed to build reliable and understandable agents that can engage in multi-step reasoning and tool use, making it a significant step up for developing more advanced LLM-powered applications.

## How Pydantic Enhances the Agent

Pydantic plays a crucial role by:
1.  **Clear Schema Definition**: `WeatherInput` clearly defines what information (`location` and `date`) the `get_current_weather` tool expects.
2.  **Input Validation**: When the LLM decides to use the tool, Langchain (using the Pydantic schema) automatically validates the arguments (`args`) provided by the LLM for the tool call. If the LLM provides malformed data (e.g., a missing field, wrong data type), Pydantic would raise an error, which can be handled gracefully. This makes the tool usage more robust.
3.  **Improved LLM Prompting**: By providing the Pydantic schema to the LLM (via `bind_tools`), the LLM gets a structured understanding of how to call the tool correctly, increasing the likelihood of successful tool invocations.

This agent serves as a great starting point for building more complex, tool-using agents with reliable input handling thanks to Pydantic and the flexible workflow management of LangGraph.

## Workflow Explained (Simple Terms)

Imagine you're talking to the agent:

1.  **You Ask**: You send a message, like "What's the weather in San Francisco today?"
2.  **Agent Thinks (LLM)**: The Gemini model (the "brain" of the agent) reads your message.
3.  **Needs a Tool?**:
    *   If your question is about weather, Gemini decides it needs to use the `get_current_weather` tool.
    *   It figures out the `location` ("San Francisco") and `date` ("today") needed for the tool. Pydantic helps ensure these details are correctly structured.
4.  **Tool Works**: The `get_current_weather` tool is called with the location and date. (In this script, it's a *mock* tool, meaning it provides pre-defined weather data for specific locations/dates rather than fetching live data).
5.  **Agent Responds (LLM again)**: Gemini takes the weather information from the tool and formulates a friendly answer for you.
6.  **No Tool Needed?**: If you ask something general, like "Tell me a fun fact," Gemini answers directly without using the weather tool.

This process is managed by LangGraph, which directs the flow between the LLM and any tools.

## Key Code Components in `pydantic_agent.py`

*   **`GEMINI_API_KEY`**: **Important!** You need to replace the placeholder with your actual Gemini API key. For security, it's best to use environment variables in production.
*   **`AgentState` (TypedDict)**: Defines the structure of the agent's memory, primarily holding the list of messages in the conversation.
*   **`WeatherInput` (Pydantic BaseModel)**:
    *   Defines the expected input for the `get_current_weather` tool: `location` (string) and `date` (string in "YYYY-MM-DD" format).
    *   Pydantic automatically validates that the data passed to the tool matches this schema.
*   **`@tool(args_schema=WeatherInput)` decorator**:
    *   Registers the `get_current_weather` function as a tool that the Langchain agent can use.
    *   The `args_schema=WeatherInput` part tells Langchain to use our Pydantic model for validating the arguments provided by the LLM when it wants to call this tool.
*   **`get_current_weather(location: str, date: str) -> str`**:
    *   The mock function that simulates fetching weather data. It returns hardcoded responses based on the input location and date.
*   **`ChatGoogleGenerativeAI`**: Initializes the Gemini LLM (using `gemini-1.5-flash` in this script).
*   **`llm.bind_tools([get_current_weather])`**: Makes the LLM aware of the `get_current_weather` tool and its Pydantic schema, enabling it to decide when and how to call it.
*   **Graph Nodes (`call_llm`, `call_tool`)**:
    *   `call_llm`: This function invokes the Gemini LLM with the current conversation history. The LLM might respond directly or suggest a tool call.
    *   `call_tool`: If the LLM suggests a tool call, this function executes the appropriate tool (e.g., `get_current_weather`) with the arguments provided by the LLM.
*   **Conditional Edge (`should_continue`)**:
    *   This function inspects the last message from the LLM. If the message contains tool call requests, it routes the workflow to the `call_tool` node. Otherwise, it ends the current cycle.
*   **`StateGraph`**:
    *   The LangGraph object where nodes (like `llm` and `tool`) and edges (the paths between nodes, including conditional ones) are defined to create the agent's execution flow.
    *   `workflow.set_entry_point("llm")`: Specifies that the process always starts by calling the LLM.
    *   `app = workflow.compile()`: Compiles the graph into a runnable application.
*   **Graph Visualization**: The script attempts to generate a PNG image of the workflow graph (`langgraph_workflow.png`) using `draw_mermaid_png()`.


