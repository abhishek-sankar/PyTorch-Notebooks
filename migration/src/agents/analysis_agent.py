"""
Analysis Agent - LLM-driven project analysis using ReAct agent
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import InMemorySaver
from ..tools import all_tools
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Configure detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisAgent:
    def __init__(self, **kwargs):
        self.model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        self.checkpointer = InMemorySaver()
        
        # Create ReAct agent with analysis prompt
        self.agent = create_react_agent(
            model=self.model,
            tools=all_tools,
            prompt=self._create_analysis_prompt,
            checkpointer=self.checkpointer
        )
    
    def _create_analysis_prompt(self, state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
        """Create system prompt for project analysis"""
        system_msg = """You are an expert Java project analysis agent for migration to Java 21.

IMPORTANT: You must provide a FINAL ANSWER after your analysis. Do not get stuck in loops.

Your task is to thoroughly analyze a Java project and provide migration recommendations.

ANALYSIS APPROACH:
1. READ pom.xml to understand current Java version, dependencies, frameworks
2. SEARCH source code for migration patterns (javax imports, JUnit usage, etc.)
3. IDENTIFY specific migration challenges and blockers
4. SUGGEST optimal sequence of OpenRewrite recipes
5. ASSESS migration complexity and risk
6. PROVIDE FINAL ANSWER with recommendations

Available tools:
- read_pom, get_java_version, list_dependencies - for Maven analysis
- search_files, list_java_files, read_file - for source code analysis
- mvn_rewrite_discover - to discover actually available OpenRewrite recipes
- run_command, mvn_compile - for project validation

SYSTEMATIC PROCESS:
1. Start with pom.xml analysis (current Java version, dependencies)
2. Search for javax imports that need Jakarta migration
3. Check for JUnit 4 vs 5 usage
4. Look for Spring Boot version
5. Identify deprecated APIs
6. Recommend migration sequence
7. Assess risk and complexity
8. PROVIDE FINAL ANSWER

You MUST end with a FINAL ANSWER that includes specific OpenRewrite recipes to run.
Example: "FINAL ANSWER: Recommended OpenRewrite migration sequence: 1. Apply recipe org.openrewrite.java.migrate.Java8toJava11..."

Take as many tool calls as needed for thorough analysis."""

        return [{"role": "system", "content": system_msg}] + state["messages"]
    
    def analyze_project(self, project_path: str) -> str:
        """Analyze a Java project for migration using ReAct reasoning"""
        print(f"\n🔍 [ANALYSIS AGENT] Starting analysis for: {project_path}")
        start_time = datetime.now()
        
        try:
            context = f"""ANALYZE THIS JAVA PROJECT FOR MIGRATION TO JAVA 21:

PROJECT PATH: {project_path}

CRITICAL: OpenRewrite recipes must be configured in pom.xml, NOT in YAML files.

Please perform a comprehensive analysis:
1. Examine the pom.xml for current Java version and dependencies
2. Search source code for patterns that need migration
3. Use mvn_rewrite_discover to find actually available OpenRewrite recipes
4. Identify frameworks in use (Spring Boot, JUnit, etc.)
5. Suggest specific OpenRewrite recipes that actually exist
6. Assess migration complexity and provide recommendations
7. PROVIDE A FINAL ANSWER with exact recipe names for pom.xml configuration

IMPORTANT: 
- Use mvn_rewrite_discover to get real available recipes
- Recommend recipes that will be configured in pom.xml (not YAML)
- You MUST provide a FINAL ANSWER with specific recipe names

Start by reading the pom.xml to understand the current state."""

            config = {
                "configurable": {
                    "thread_id": f"analysis_{hash(project_path)}"
                },
                "recursion_limit": 35  # Keep higher limit as requested
            }
            
            print(f"🤖 [ANALYSIS AGENT] Starting ReAct reasoning...")
            print(f"🔧 [ANALYSIS AGENT] Streaming with detailed logging...")
            
            # Use streaming to show detailed progress
            step_count = 0
            for chunk in self.agent.stream(
                {"messages": [{"role": "user", "content": context}]},
                config
            ):
                step_count += 1
                print(f"\n📊 [STEP {step_count}] ==========================================")
                
                # Print detailed chunk information
                if chunk and isinstance(chunk, dict):
                    for node_name, node_data in chunk.items():
                        print(f"🏗️  [NODE] {node_name}")
                        
                        if 'messages' in node_data and node_data['messages']:
                            for i, msg in enumerate(node_data['messages']):
                                print(f"  📝 [MESSAGE {i+1}]:")
                                
                                # Show LLM thinking/content
                                if hasattr(msg, 'content') and msg.content:
                                    content = str(msg.content)
                                    if len(content) > 300:
                                        print(f"    💭 Content: {content[:300]}...")
                                    else:
                                        print(f"    💭 Content: {content}")
                                
                                # Show tool calls with parameters
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    for j, tool_call in enumerate(msg.tool_calls):
                                        print(f"    🔧 [TOOL CALL {j+1}]:")
                                        print(f"       Tool: {getattr(tool_call, 'name', 'unknown')}")
                                        
                                        # Show parameters
                                        if hasattr(tool_call, 'args'):
                                            args = tool_call.args
                                            print(f"       Args: {args}")
                                        elif hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                                            args = tool_call.function.arguments
                                            print(f"       Args: {args}")
                                        
                                # Show tool responses
                                if hasattr(msg, 'name') and msg.name:
                                    print(f"    🔄 Tool Response from {msg.name}:")
                                    if hasattr(msg, 'content'):
                                        response = str(msg.content)
                                        if len(response) > 200:
                                            print(f"       {response[:200]}...")
                                        else:
                                            print(f"       {response}")
            
            print(f"\n📊 [ANALYSIS AGENT] Completed streaming, getting final result...")
            
            # Get final result
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": context}]},
                config
            )
            
            elapsed = datetime.now() - start_time
            print(f"⏱️ [ANALYSIS AGENT] Completed in {elapsed.total_seconds():.2f} seconds")
            
            # Log the tool calls from the final result
            if result and "messages" in result:
                print(f"📊 [ANALYSIS AGENT] Processing {len(result['messages'])} messages")
                tool_call_count = 0
                
                for i, msg in enumerate(result["messages"]):
                    # Log tool calls
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_call_count += 1
                            print(f"\n🔧 [TOOL CALL #{tool_call_count}]:")
                            if hasattr(tool_call, 'function'):
                                print(f"Tool: {tool_call.function.name}")
                                print(f"Args: {tool_call.function.arguments}")
                            elif isinstance(tool_call, dict):
                                print(f"Tool: {tool_call.get('name', 'unknown')}")
                                print(f"Args: {tool_call.get('args', {})}")
                            print(f"{'='*50}")
                    
                    # Show final thinking if it's the last message and has content
                    if i == len(result["messages"]) - 1 and hasattr(msg, 'content') and msg.content:
                        print(f"\n🧠 [FINAL LLM RESPONSE]:")
                        print(f"{msg.content}")
                        print(f"{'='*80}")
                
                print(f"📊 [ANALYSIS AGENT] Total tool calls made: {tool_call_count}")
                
                # Extract final result
                last_message = result["messages"][-1]
                final_result = self._extract_message_content(last_message)
                return final_result
            else:
                return "Analysis agent completed but no response found"
            
        except Exception as e:
            elapsed = datetime.now() - start_time
            print(f"❌ [ANALYSIS AGENT] Failed after {elapsed.total_seconds():.2f} seconds: {str(e)}")
            logger.error(f"Analysis agent error: {str(e)}", exc_info=True)
            return f"Analysis agent failed: {str(e)}"
    
    def _extract_message_content(self, msg) -> str:
        """Extract content from a message object"""
        if hasattr(msg, 'content'):
            return str(msg.content)
        elif isinstance(msg, dict) and 'content' in msg:
            return str(msg['content'])
        else:
            return str(msg)