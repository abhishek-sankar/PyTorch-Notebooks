"""
Error Agent - LLM-driven error analysis and fixing using ReAct agent
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

class ErrorAgent:
    def __init__(self, **kwargs):
        self.model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        self.checkpointer = InMemorySaver()
        
        # Create ReAct agent with error fixing prompt
        self.agent = create_react_agent(
            model=self.model,
            tools=all_tools,
            prompt=self._create_error_fixing_prompt,
            checkpointer=self.checkpointer
        )
    
    def _create_error_fixing_prompt(self, state: AgentState, config: RunnableConfig) -> list[AnyMessage]:
        """Create system prompt for error fixing"""
        system_msg = """You are an expert Java migration error fixing agent. 

IMPORTANT: Focus on fixing the specific error efficiently. Provide a clear final status report.

When given an error message and project context, you need to:
1. ANALYZE the error to understand what went wrong
2. INVESTIGATE the project state using available tools
3. APPLY appropriate fixes using the tools
4. VERIFY the fix worked by running compile/test
5. PROVIDE FINAL RESULT

Available tools allow you to:
- Read and write files (read_file, write_file, find_replace)
- Search for patterns in code (search_files, list_java_files)
- Run Maven commands (mvn_compile, mvn_test, mvn_rewrite_run)
- Execute shell commands (run_command)
- Modify pom.xml (read_pom, update_java_version, add_openrewrite_plugin)
- Create OpenRewrite configurations (create_rewrite_config, get_available_recipes)

REASONING APPROACH:
- Think step by step about what the error means
- Use tools to investigate the current state
- Apply targeted fixes
- Test your fixes
- Report final status

If you cannot fix the error after multiple attempts, clearly state:
"ESCALATION NEEDED: [describe the issue and what human input is required]"

Take as many tool calls as needed to properly fix the error."""

        return [{"role": "system", "content": system_msg}] + state["messages"]
    
    def fix_error(self, error_message: str, project_path: str, last_output: str = "") -> str:
        """Fix any error using ReAct reasoning and available tools"""
        print(f"\n🔧 [ERROR AGENT] Starting error fix for: {project_path}")
        print(f"❌ [ERROR AGENT] Error: {error_message[:200]}...")
        start_time = datetime.now()
        
        try:
            # Create context message for the error
            context = f"""ERROR TO FIX:
{error_message}

PROJECT PATH: {project_path}

LAST OUTPUT CONTEXT:
{last_output[-2000] if last_output else "No previous output provided"}

Please analyze this error and fix it using the available tools. Start by understanding what went wrong, then investigate and apply fixes.

IMPORTANT: Provide a clear final status report when complete. Take as many tool calls as needed."""

            config = {
                "configurable": {
                    "thread_id": f"error_fix_{hash(error_message)}"
                },
                "recursion_limit": 30  # Keep higher limit as requested
            }
            
            print(f"🤖 [ERROR AGENT] Starting error fixing with detailed logging...")
            
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
            
            print(f"\n📊 [ERROR AGENT] Completed streaming, getting final result...")
            
            # Get final result
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": context}]},
                config
            )
            
            elapsed = datetime.now() - start_time
            print(f"⏱️ [ERROR AGENT] Completed in {elapsed.total_seconds():.2f} seconds")
            
            # Log detailed result information
            if result and "messages" in result:
                print(f"📊 [ERROR AGENT] Received {len(result['messages'])} messages")
                
                # Log key messages for debugging (summarize if too many)
                if len(result["messages"]) > 10:
                    print(f"  📝 [MSG 0-2] First few messages...")
                    for i in range(3):
                        if i < len(result["messages"]):
                            msg = result["messages"][i]
                            msg_content = self._extract_message_content(msg)[:150] + "..."
                            print(f"    [MSG {i}] {msg_content}")
                    
                    print(f"  📝 [MSG ...] Skipping {len(result['messages'])-6} middle messages for brevity")
                    
                    print(f"  📝 [MSG -3 to -1] Last few messages...")
                    for i in range(max(0, len(result["messages"])-3), len(result["messages"])):
                        msg = result["messages"][i]
                        msg_content = self._extract_message_content(msg)[:150] + "..."
                        print(f"    [MSG {i}] {msg_content}")
                else:
                    # Show all messages if not too many
                    for i, msg in enumerate(result["messages"]):
                        msg_content = self._extract_message_content(msg)[:200] + "..."
                        print(f"  📝 [MSG {i}] {msg_content}")
                
                last_message = result["messages"][-1]
                final_result = self._extract_message_content(last_message)
                
                print(f"✅ [ERROR AGENT] Final result: {final_result[:300]}...")
                return final_result
            else:
                print(f"❌ [ERROR AGENT] No messages in result")
                return "Error agent completed but no response found"
            
        except Exception as e:
            elapsed = datetime.now() - start_time
            print(f"❌ [ERROR AGENT] Failed after {elapsed.total_seconds():.2f} seconds: {str(e)}")
            logger.error(f"Error agent error: {str(e)}", exc_info=True)
            return f"Error agent failed: {str(e)}"
    
    def _extract_message_content(self, msg) -> str:
        """Extract content from a message object"""
        if hasattr(msg, 'content'):
            return str(msg.content)
        elif isinstance(msg, dict) and 'content' in msg:
            return str(msg['content'])
        else:
            return str(msg)