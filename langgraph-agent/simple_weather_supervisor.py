"""
Simple Weather Agent Supervisor for testing streaming
"""
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather(location: str) -> str:
    """Get weather information for a specific location"""
    # Simulate API call delay
    import time
    time.sleep(2)
    
    # Mock weather data
    weather_data = {
        "new york": "Sunny, 72°F",
        "london": "Cloudy, 15°C", 
        "tokyo": "Rainy, 22°C",
        "paris": "Partly cloudy, 18°C"
    }
    
    # Normalize location to lowercase for case-insensitive matching
    location_key = location.lower().strip()
    return weather_data.get(location_key, f"Weather data not available for {location}")

@tool 
def get_forecast(location: str, days: int = 3) -> str:
    """Get weather forecast for a location"""
    import time
    time.sleep(1)
    
    return f"{days}-day forecast for {location}: Mix of sun and clouds, temps 15-25°C"

class SimpleWeatherSupervisor:
    def __init__(self):
        print("Initializing Simple Weather Supervisor...")
        
        # Create LLM
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Create weather agent
        self.weather_agent = create_react_agent(
            self.llm,
            tools=[get_weather, get_forecast],
            name="weather_agent"
        )
        
        # Create supervisor workflow
        supervisor_workflow = create_supervisor(
            agents=[self.weather_agent],
            model=self.llm,
            prompt="""You are a helpful supervisor that coordinates weather requests.

Available agents:
- weather_agent: Handles weather-related questions for any location

Route weather questions to weather_agent. Be helpful and provide detailed responses."""
        )
        
        # Compile with memory
        checkpointer = MemorySaver()
        self.app = supervisor_workflow.compile(checkpointer=checkpointer)
        
        print("Simple weather supervisor initialized!")
    
    def get_weather_stream(self, query: str):
        """Stream weather request processing"""
        print(f"SUPERVISOR: Processing weather query: {query}")
        
        try:
            step_count = 0
            final_result = None
            
            # Stream the actual LangGraph execution
            for chunk in self.app.stream(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": f"weather_{hash(query)}"}}
            ):
                step_count += 1
                print(f"SUPERVISOR: Step {step_count}: {list(chunk.keys())} - YIELDING NOW")
                
                # Extract meaningful progress from actual chunk data
                for node_name, node_data in chunk.items():
                    if isinstance(node_data, dict) and "messages" in node_data:
                        messages = node_data["messages"]
                        if messages:
                            last_msg = messages[-1]
                            if hasattr(last_msg, 'content') and last_msg.content:
                                # This is actual progress - yield it immediately
                                progress_data = {
                                    "type": "progress",
                                    "step": step_count,
                                    "node": node_name,
                                    "content": last_msg.content[:200] + "..." if len(str(last_msg.content)) > 200 else last_msg.content
                                }
                                print(f"SUPERVISOR: YIELDING PROGRESS: {progress_data}")
                                yield progress_data
                                
                                # Force a small delay to ensure streaming
                                import time
                                time.sleep(0.1)
                
                # Store final result
                final_result = chunk
            
            # Extract final response
            if final_result:
                for node_data in final_result.values():
                    if isinstance(node_data, dict) and "messages" in node_data:
                        messages = node_data["messages"]
                        if messages:
                            final_message = messages[-1]
                            final_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
                            
                            yield {
                                "type": "complete",
                                "success": True,
                                "result": final_content,
                                "steps": step_count
                            }
                            return
            
            # Fallback if no proper result found
            yield {
                "type": "complete",
                "success": False,
                "error": "No valid response generated"
            }
            
        except Exception as e:
            print(f"SUPERVISOR ERROR: {e}")
            import traceback
            traceback.print_exc()
            yield {
                "type": "complete", 
                "success": False,
                "error": str(e)
            }

if __name__ == "__main__":
    supervisor = SimpleWeatherSupervisor()
    
    # Test streaming
    print("\nTesting weather query streaming...")
    for progress in supervisor.get_weather_stream("What's the weather like in New York?"):
        if progress.get("type") == "progress":
            print(f"PROGRESS: Step {progress['step']}")
        elif progress.get("type") == "complete":
            print(f"COMPLETE: {progress['result']}")
            break