"""
Execution Agent - LLM-driven OpenRewrite recipe execution using ReAct agent
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

class ExecutionAgent:
    def __init__(self, **kwargs):
        self.model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        self.checkpointer = InMemorySaver()
        
        # Create ReAct agent with execution prompt
        self.agent = create_react_agent(
            model=self.model,
            tools=all_tools,
            prompt=self._create_execution_prompt,
            checkpointer=self.checkpointer
        )
    
    def _create_execution_prompt(self, state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
        """Create system prompt for recipe execution"""
        system_msg = """You are an expert OpenRewrite recipe execution agent for Java migration.

IMPORTANT: You must complete the execution efficiently. Do not get stuck in loops.

Your task is to execute migration recipes systematically and handle any issues that arise.

EXECUTION APPROACH:
1. PREPARE - Set up OpenRewrite configuration and validate project state
2. EXECUTE - Run recipes in optimal order
3. VALIDATE - Check compilation after each major step
4. HANDLE ERRORS - Fix any issues that arise during execution
5. COMMIT - Save progress at key checkpoints
6. PROVIDE FINAL RESULT

Available tools:
- add_openrewrite_plugin, configure_openrewrite_recipes, add_rewrite_dependency - for pom.xml plugin setup
- mvn_rewrite_discover, mvn_rewrite_run, mvn_rewrite_run_recipe, mvn_rewrite_dry_run - for recipe execution
- mvn_compile, mvn_test - for validation
- git_status, git_add_all, git_commit - for progress tracking
- All file and command tools for troubleshooting

CRITICAL: OpenRewrite MUST be configured in pom.xml, not YAML files. This is the only way that works.

EXECUTION STRATEGY:
1. Check if OpenRewrite plugin exists in pom.xml
2. If not present, add plugin using add_openrewrite_plugin
3. Configure plugin with recipes using configure_openrewrite_recipes (adds recipes AND dependencies)
4. Run mvn rewrite:run to execute the recipes
5. Validate with mvn compile
6. Handle any compilation errors
7. Commit successful changes
8. Report completion status

IMPORTANT: Use configure_openrewrite_recipes to set up both active recipes AND the rewrite-migrate-java dependency in one step.

You MUST provide a final status report when complete.
Take as many tool calls as needed to execute the recipes properly."""

        return [{"role": "system", "content": system_msg}] + state["messages"]
    
    def execute_recipes(self, project_path: str, recipes: list) -> str:
        """Execute OpenRewrite recipes using ReAct reasoning"""
        print(f"\n⚙️ [EXECUTION AGENT] Starting execution for: {project_path}")
        print(f"📋 [EXECUTION AGENT] Recipes: {recipes}")
        start_time = datetime.now()
        
        try:
            recipes_str = ", ".join(recipes)
            context = f"""EXECUTE OPENREWRITE RECIPES FOR JAVA MIGRATION:

PROJECT PATH: {project_path}

RECIPES TO EXECUTE:
{recipes_str}

CRITICAL INSTRUCTIONS:
- OpenRewrite MUST be configured in pom.xml, NOT in YAML files
- YAML configuration files DO NOT WORK and should be ignored
- You must modify the pom.xml directly to configure recipes

EXECUTE THESE RECIPES SYSTEMATICALLY:
1. Check if OpenRewrite plugin exists in pom.xml
2. If not present, add plugin using add_openrewrite_plugin tool
3. Configure the plugin with recipes using configure_openrewrite_recipes tool (this adds both recipes AND dependencies)
4. Run mvn rewrite:run to execute the recipes
5. Validate compilation with mvn compile
6. Handle any errors that arise
7. Commit successful changes with descriptive message
8. PROVIDE FINAL STATUS REPORT

DO NOT create or use rewrite.yml files - they don't work. Only use pom.xml configuration.

Start by checking the current pom.xml state."""

            config = {
                "configurable": {
                    "thread_id": f"execution_{hash(project_path + recipes_str)}"
                },
                "recursion_limit": 35  # Keep higher limit as requested
            }
            
            print(f"🤖 [EXECUTION AGENT] Starting ReAct reasoning...")
            print(f"🔧 [EXECUTION AGENT] Streaming with detailed logging...")
            
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
            
            print(f"\n📊 [EXECUTION AGENT] Completed streaming, getting final result...")
            
            # Get final result
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": context}]},
                config
            )
            
            elapsed = datetime.now() - start_time
            print(f"⏱️ [EXECUTION AGENT] Completed in {elapsed.total_seconds():.2f} seconds")
            
            # Log the tool calls from the final result
            if result and "messages" in result:
                print(f"📊 [EXECUTION AGENT] Processing {len(result['messages'])} messages")
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
                
                print(f"📊 [EXECUTION AGENT] Total tool calls made: {tool_call_count}")
                
                # Extract final result
                last_message = result["messages"][-1]
                final_result = self._extract_message_content(last_message)
                return final_result
            else:
                return "Execution agent completed but no response found"
            
        except Exception as e:
            elapsed = datetime.now() - start_time
            print(f"❌ [EXECUTION AGENT] Failed after {elapsed.total_seconds():.2f} seconds: {str(e)}")
            logger.error(f"Execution agent error: {str(e)}", exc_info=True)
            return f"Execution agent failed: {str(e)}"
    
    def _extract_message_content(self, msg) -> str:
        """Extract content from a message object"""
        if hasattr(msg, 'content'):
            return str(msg.content)
        elif isinstance(msg, dict) and 'content' in msg:
            return str(msg['content'])
        else:
            return str(msg)