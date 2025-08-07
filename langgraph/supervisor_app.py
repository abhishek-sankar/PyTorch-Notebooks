import os
from typing import Dict, Any, List
from dotenv import load_dotenv

from langgraph_supervisor.supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from agents import WeatherAgent, CalculatorAgent

load_dotenv()


@tool
def get_weather_info(location: str) -> str:
    """Get weather information for a specific location"""
    weather_agent = WeatherAgent()
    return weather_agent.get_weather(location)


@tool  
def calculate_math(expression: str) -> str:
    """Perform mathematical calculations"""
    calc_agent = CalculatorAgent()
    return calc_agent.calculate(expression)


def create_supervisor_system():
    """Create and configure the supervisor with sub-agents"""
    
    # Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Create agents with tools
    weather_agent = create_react_agent(
        llm, 
        tools=[get_weather_info],
        state_modifier="You are a weather assistant. Help users get weather information for any location."
    )
    # Set the name attribute for the supervisor
    weather_agent.name = "weather_agent"
    
    calculator_agent = create_react_agent(
        llm,
        tools=[calculate_math], 
        state_modifier="You are a calculator assistant. Help users perform mathematical calculations."
    )
    # Set the name attribute for the supervisor
    calculator_agent.name = "calculator_agent"
    
    # Create supervisor workflow
    supervisor_workflow = create_supervisor(
        agents=[weather_agent, calculator_agent],
        model=llm,
        prompt="""You are a helpful supervisor that coordinates between specialized agents.

Available agents:
- weather_agent: Handles weather-related questions for any location
- calculator_agent: Performs mathematical calculations

Route weather questions to weather_agent and math questions to calculator_agent.
For general questions, respond directly."""
    )
    
    # Compile the workflow with memory checkpointer for conversation history
    checkpointer = MemorySaver()
    supervisor = supervisor_workflow.compile(checkpointer=checkpointer)
    
    return supervisor


def main():
    """Test the supervisor locally"""
    supervisor = create_supervisor_system()
    
    print("LangGraph Supervisor Demo")
    print("Available agents: weather_agent, calculator_agent")
    print("Try asking: 'What's the weather in New York?' or 'Calculate 15 + 25'")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        
        try:
            # Create message list for supervisor
            messages = [HumanMessage(content=user_input)]
            
            # Get response from supervisor
            response = supervisor.invoke({"messages": messages})
            
            # Extract the response content
            if isinstance(response, dict) and "messages" in response:
                last_message = response["messages"][-1]
                if hasattr(last_message, 'content'):
                    print(f"Supervisor: {last_message.content}\n")
                else:
                    print(f"Supervisor: {last_message}\n")
            else:
                print(f"Supervisor: {response}\n")
                
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()