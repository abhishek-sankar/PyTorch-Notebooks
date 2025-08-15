"""
Supervisor-Based Migration Orchestrator
Uses LangGraph's create_supervisor for intelligent agent management
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

try:
    from migration.src.tools.command_executor import mvn_compile, mvn_test, run_command
    from migration.prompts.prompt_loader import (
        get_supervisor_prompt,
        get_migration_request,
        get_analysis_expert_prompt,
        get_execution_expert_prompt,
        get_error_expert_prompt
    )
except ImportError:
    # Fallback - create mock functions for now
    def mvn_compile(*args, **kwargs):
        return "Mock compilation result"
    
    def mvn_test(*args, **kwargs): 
        return "Mock test result"
        
    def run_command(*args, **kwargs):
        return "Mock command result"
    
    def get_supervisor_prompt():
        return "You are a helpful migration supervisor that coordinates Java migration tasks."
        
    def get_migration_request(project_path):
        return f"Please analyze and migrate the Java project at {project_path}"
        
    def get_analysis_expert_prompt():
        return "You are an analysis expert for Java migrations."
        
    def get_execution_expert_prompt():
        return "You are an execution expert for Java migrations."
        
    def get_error_expert_prompt():
        return "You are an error-fixing expert for Java migrations."


class SupervisorMigrationOrchestrator:
    """Supervisor that manages specialized migration agents"""
    
    def __init__(self):
        print("Initializing Supervisor Migration Orchestrator...")
        
        # Create specialized migration agents as workers
        self.migration_workers = self._create_migration_workers()
        
        # Create supervisor workflow
        self.supervisor_workflow = self._create_supervisor()
        
        # Compile the workflow
        self.app = self.supervisor_workflow.compile()
        
        print("Supervisor orchestrator initialized with workers:", [agent.name for agent in self.migration_workers])
    
    def _create_migration_workers(self):
        """Create specialized worker agents for migration tasks"""
        
        # Create validation tools for workers
        validation_tools = [mvn_compile, mvn_test, run_command]
        
        # Analysis Worker - analyzes projects and recommends recipes
        analysis_worker = create_react_agent(
            model=ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                temperature=0
            ),
            tools=self._get_analysis_tools(),
            prompt=get_analysis_expert_prompt(),
            name="analysis_expert"
        )
        
        # Execution Worker - executes OpenRewrite recipes
        execution_worker = create_react_agent(
            model=ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                temperature=0
            ),
            tools=self._get_execution_tools() + validation_tools,
            prompt=get_execution_expert_prompt(),
            name="execution_expert"
        )
        
        # Error Fixing Worker - fixes compilation and build errors
        error_worker = create_react_agent(
            model=ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                temperature=0
            ),
            tools=self._get_error_tools() + validation_tools,
            prompt=get_error_expert_prompt(),
            name="error_expert"
        )
        
        return [analysis_worker, execution_worker, error_worker]
    
    def _get_analysis_tools(self):
        """Get tools for analysis agent"""
        # Return basic tools for now - you can add real tools later
        return [mvn_compile, mvn_test, run_command]
    
    def _get_execution_tools(self):
        """Get tools for execution agent"""
        # Return basic tools for now - you can add real tools later
        return [mvn_compile, mvn_test, run_command]
    
    def _get_error_tools(self):
        """Get tools for error agent"""
        # Return basic tools for now - you can add real tools later
        return [mvn_compile, mvn_test, run_command]
    
    def _create_supervisor(self):
        """Create supervisor workflow that manages migration workers"""
        
        supervisor_model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0
        )
        
        # Create supervisor with migration context
        workflow = create_supervisor(
            agents=self.migration_workers,
            model=supervisor_model,
            prompt=get_supervisor_prompt()
        )
        
        return workflow
    
    def migrate_project_stream(self, project_path: str):
        """Start supervised migration with streaming progress updates"""
        print("="*80)
        print(f"STARTING SUPERVISED MIGRATION: {project_path}")
        print("="*80)
        
        if not os.path.exists(project_path):
            yield {"type": "complete", "success": False, "error": f"Project path does not exist: {project_path}"}
            return
        
        # Create migration request using external template
        migration_request = get_migration_request(project_path)
        
        try:
            start_time = datetime.now()
            
            print(f"\nSupervisor: Starting migration workflow...")
            print(f"Supervisor: Available workers: {[agent.name for agent in self.migration_workers]}")
            print("-" * 60)
            
            # Initial progress message
            yield {
                "type": "progress", 
                "step": 0, 
                "message": f"Starting migration for {project_path}",
                "agent": "supervisor"
            }
            
            # Stream the workflow execution with progress tracking
            step_count = 0
            last_agent = None
            
            for chunk in self.app.stream({
                "messages": [{"role": "user", "content": migration_request}]
            }):
                step_count += 1
                self._log_workflow_step(step_count, chunk)
                
                # Extract detailed progress information (can be multiple events per chunk)
                progress_events = self._extract_progress_info(chunk, step_count)
                
                # Yield each progress event
                for progress_event in progress_events:
                    current_agent = progress_event.get("agent", "supervisor")
                    
                    # Track agent transitions
                    if current_agent != last_agent:
                        yield {
                            "type": "progress",
                            "step": step_count,
                            "message": f"Calling {current_agent.replace('_', ' ').title()}",
                            "agent": current_agent,
                            "event_type": "agent_transition"
                        }
                        last_agent = current_agent
                    
                    # Yield the detailed progress event
                    yield progress_event
            
            duration = datetime.now() - start_time
            
            print("\n" + "="*80)
            print(f"SUPERVISED MIGRATION COMPLETED in {duration.total_seconds():.2f} seconds")
            print("="*80)
            
            # Get final result
            final_result = self.app.invoke({
                "messages": [{"role": "user", "content": migration_request}]
            })
            
            # Extract final result
            messages = final_result.get("messages", [])
            final_message = messages[-1] if messages else {}
            
            # Handle different message types
            if hasattr(final_message, 'content'):
                final_content = final_message.content
            elif isinstance(final_message, dict):
                final_content = final_message.get("content", str(final_message))
            else:
                final_content = str(final_message)
            
            yield {
                "type": "complete",
                "success": True,
                "result": final_content,
                "duration": duration.total_seconds(),
                "messages": len(messages),
                "steps": step_count,
                "project_path": project_path
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nSUPERVISOR ERROR: {str(e)}")
            yield {
                "type": "complete",
                "success": False,
                "error": str(e),
                "project_path": project_path
            }
    
    def _extract_progress_info(self, chunk: Dict[str, Any], step_count: int) -> List[Dict[str, Any]]:
        """Extract detailed progress information from workflow chunks"""
        if not chunk:
            return []
        
        progress_events = []
        
        # Check for active agent and extract detailed information
        for node_name, node_data in chunk.items():
            if node_name in ["analysis_expert", "execution_expert", "error_expert", "supervisor"]:
                
                # Look for messages in this node
                if isinstance(node_data, dict) and "messages" in node_data:
                    messages = node_data["messages"]
                    
                    # Process each message for detailed info
                    for i, msg in enumerate(messages):
                        # Extract LLM calls and responses
                        if self._is_ai_message(msg):
                            content = self._get_message_content(msg)
                            if content and content.strip():
                                progress_events.append({
                                    "type": "progress",
                                    "step": step_count,
                                    "agent": node_name,
                                    "event_type": "llm_response",
                                    "message": "LLM Response",
                                    "content": content,
                                    "details": {"message_index": i, "node": node_name}
                                })
                        
                        # Extract tool calls
                        tool_calls = self._get_tool_calls(msg)
                        if tool_calls:
                            for j, tool_call in enumerate(tool_calls):
                                tool_name = self._get_tool_name(tool_call)
                                tool_args = self._get_tool_args(tool_call)
                                
                                # Debug the actual tool_call structure
                                print(f"DEBUG TOOL CALL: {tool_call}")
                                print(f"DEBUG TOOL ARGS: {tool_args}")
                                
                                progress_events.append({
                                    "type": "progress",
                                    "step": step_count,
                                    "agent": node_name,
                                    "event_type": "tool_call",
                                    "message": f"Tool Call: {tool_name}",
                                    "content": f"**Tool:** {tool_name}\n**Parameters:** {json.dumps(tool_args, indent=2) if tool_args else 'No parameters'}",
                                    "details": {"tool": tool_name, "args": tool_args, "call_index": j}
                                })
                        
                        # Extract tool results
                        if self._is_tool_message(msg):
                            tool_name = self._get_message_name(msg)
                            tool_result = self._get_message_content(msg)
                            
                            progress_events.append({
                                "type": "progress", 
                                "step": step_count,
                                "agent": node_name,
                                "event_type": "tool_result",
                                "message": f"Tool Result: {tool_name}",
                                "content": f"**Tool:** {tool_name}\n**Result:** {tool_result[:500]}{'...' if len(str(tool_result)) > 500 else ''}",
                                "details": {"tool": tool_name, "result_preview": str(tool_result)[:100]}
                            })
                
                # If no detailed messages, show basic agent activity
                if not progress_events:
                    progress_events.append({
                        "type": "progress",
                        "step": step_count,
                        "agent": node_name,
                        "event_type": "agent_activity", 
                        "message": f"{node_name.replace('_', ' ').title()} activated",
                        "content": f"Agent {node_name} is now processing...",
                        "details": {"node": node_name}
                    })
        
        return progress_events
    
    def _is_ai_message(self, msg) -> bool:
        """Check if message is from AI/LLM"""
        if isinstance(msg, dict):
            return msg.get("type") in ["ai", "assistant"] or "ai" in str(msg.get("type", "")).lower()
        return hasattr(msg, "type") and (msg.type in ["ai", "assistant"] or "ai" in str(msg.type).lower())
    
    def _is_tool_message(self, msg) -> bool:
        """Check if message is a tool result"""
        if isinstance(msg, dict):
            return msg.get("type") in ["tool", "function"]
        return hasattr(msg, "type") and msg.type in ["tool", "function"]
    
    def _get_message_content(self, msg):
        """Extract content from message"""
        if isinstance(msg, dict):
            return msg.get("content", "")
        return getattr(msg, "content", "")
    
    def _get_message_name(self, msg):
        """Extract name/tool name from message"""
        if isinstance(msg, dict):
            return msg.get("name", "")
        return getattr(msg, "name", "")
    
    def _get_tool_calls(self, msg):
        """Extract tool calls from message"""
        if isinstance(msg, dict):
            return msg.get("tool_calls", [])
        return getattr(msg, "tool_calls", [])
    
    def _get_tool_name(self, tool_call):
        """Extract tool name from tool call"""
        if isinstance(tool_call, dict):
            return tool_call.get("name", "unknown_tool")
        return getattr(tool_call, "name", "unknown_tool")
    
    def _get_tool_args(self, tool_call):
        """Extract tool arguments from tool call"""
        if isinstance(tool_call, dict):
            # Try multiple possible keys for arguments
            args = tool_call.get("args")
            if args is None:
                args = tool_call.get("arguments")
            if args is None:
                args = tool_call.get("parameters")
            return args or {}
        else:
            # For object-style tool calls, try multiple attributes
            for attr in ["args", "arguments", "parameters"]:
                args = getattr(tool_call, attr, None)
                if args is not None:
                    return args
            return {}
    
    def _extract_tool_info(self, msg, progress_info: Dict[str, Any]) -> bool:
        """Extract tool usage information from messages"""
        try:
            # Handle different message types
            tool_calls = None
            if isinstance(msg, dict):
                tool_calls = msg.get("tool_calls", [])
            elif hasattr(msg, "tool_calls"):
                tool_calls = msg.tool_calls
            
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = None
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "")
                    elif hasattr(tool_call, "name"):
                        tool_name = tool_call.name
                    
                    if tool_name:
                        progress_info["message"] = f"{progress_info['agent'].replace('_', ' ').title()} using {tool_name}"
                        progress_info["tool"] = tool_name
                        return True
        
        except Exception as e:
            print(f"Error extracting tool info: {e}")
        
        return False

    def migrate_project(self, project_path: str) -> Dict[str, Any]:
        """Start supervised migration with detailed progress tracking"""
        print("="*80)
        print(f"STARTING SUPERVISED MIGRATION: {project_path}")
        print("="*80)
        
        if not os.path.exists(project_path):
            return {"success": False, "error": f"Project path does not exist: {project_path}"}
        
        # Create migration request using external template
        migration_request = get_migration_request(project_path)
        
        try:
            start_time = datetime.now()
            
            print(f"\nSupervisor: Starting migration workflow...")
            print(f"Supervisor: Available workers: {[agent.name for agent in self.migration_workers]}")
            print("-" * 60)
            
            # Stream the workflow execution with progress tracking
            step_count = 0
            for chunk in self.app.stream({
                "messages": [{"role": "user", "content": migration_request}]
            }):
                step_count += 1
                self._log_workflow_step(step_count, chunk)
            
            # Get final result
            final_result = self.app.invoke({
                "messages": [{"role": "user", "content": migration_request}]
            })
            
            duration = datetime.now() - start_time
            
            print("\n" + "="*80)
            print(f"SUPERVISED MIGRATION COMPLETED in {duration.total_seconds():.2f} seconds")
            print("="*80)
            
            # Extract final result
            messages = final_result.get("messages", [])
            final_message = messages[-1] if messages else {}
            final_content = final_message.get("content", "No final message") if isinstance(final_message, dict) else str(final_message)
            
            return {
                "success": True,
                "result": final_content,
                "duration": duration.total_seconds(),
                "messages": len(messages),
                "steps": step_count
            }
            
        except Exception as e:
            print(f"\nSUPERVISOR ERROR: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _log_workflow_step(self, step_count: int, chunk: Dict[str, Any]):
        """Log detailed information about each workflow step"""
        print(f"\n[STEP {step_count}] " + "="*50)
        
        if not chunk:
            print("Empty chunk received")
            return
        
        for node_name, node_data in chunk.items():
            print(f"NODE: {node_name}")
            
            # Check if this is a worker agent being called
            if node_name in ["analysis_expert", "execution_expert", "error_expert"]:
                print(f"  -> Calling {node_name.replace('_', ' ').title()}")
                
            # Show messages if available
            if isinstance(node_data, dict) and "messages" in node_data:
                messages = node_data["messages"]
                print(f"  Messages: {len(messages)}")
                
                # Show all messages to capture LLM responses
                for msg in messages:
                    self._display_detailed_message(msg)
            
            # Show other data if not messages
            elif node_data and str(node_data) != "{}":
                data_preview = str(node_data)[:150] + ("..." if len(str(node_data)) > 150 else "")
                print(f"  Data: {data_preview}")
        
        print("-" * 60)
    
    def _display_detailed_message(self, msg):
        """Display detailed information about a message including full LLM responses"""
        
        # Debug: print raw message structure
        print(f"    DEBUG: Message type: {type(msg)}")
        if isinstance(msg, dict):
            print(f"    DEBUG: Dict keys: {list(msg.keys())}")
        else:
            print(f"    DEBUG: Object attrs: {[attr for attr in dir(msg) if not attr.startswith('_')]}")
        
        # Extract message details
        if isinstance(msg, dict):
            msg_content = msg.get("content", "")
            msg_type = msg.get("type", "unknown")
            msg_name = msg.get("name", "")
            tool_calls = msg.get("tool_calls", [])
        else:
            msg_content = getattr(msg, "content", "")
            msg_type = getattr(msg, "type", "unknown") 
            msg_name = getattr(msg, "name", "")
            tool_calls = getattr(msg, "tool_calls", [])
        
        # Show message header
        if msg_name:
            print(f"    [{msg_type.upper()}] from {msg_name}:")
        else:
            print(f"    [{msg_type.upper()}]:")
        
        # Show full LLM response content - check for AI messages more broadly
        if msg_content and (msg_type in ["ai", "assistant"] or "ai" in str(msg_type).lower()):
            print("    " + "="*50)
            print(f"    LLM RESPONSE:")
            print("    " + "="*50)
            # Show full content for LLM responses
            content_lines = str(msg_content).split('\n')
            for line in content_lines:
                print(f"    {line}")
            print("    " + "="*50)
        elif msg_content and msg_type not in ["tool", "function"]:
            # Show full content for non-tool messages to catch LLM responses
            print("    " + "-"*30)
            print(f"    CONTENT ({msg_type}):")
            print("    " + "-"*30)
            content_lines = str(msg_content).split('\n')
            for line in content_lines[:20]:  # Show first 20 lines
                print(f"    {line}")
            if len(content_lines) > 20:
                print(f"    ... ({len(content_lines) - 20} more lines)")
            print("    " + "-"*30)
        elif msg_content:
            # Abbreviated for tool messages
            if len(str(msg_content)) > 200:
                print(f"    Content: {str(msg_content)[:200]}...")
            else:
                print(f"    Content: {str(msg_content)}")
        
        # Show detailed tool calls
        if tool_calls:
            print(f"    TOOL CALLS ({len(tool_calls)}):")
            for i, tool_call in enumerate(tool_calls, 1):
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "")
                else:
                    tool_name = getattr(tool_call, "name", "unknown")
                    tool_args = getattr(tool_call, "args", {})
                    tool_id = getattr(tool_call, "id", "")
                
                print(f"      [{i}] {tool_name}")
                if tool_id:
                    print(f"          ID: {tool_id}")
                if tool_args:
                    print(f"          Args: {tool_args}")
                print()


if __name__ == "__main__":
    project_path = "./flatworm"
    
    orchestrator = SupervisorMigrationOrchestrator()
    result = orchestrator.migrate_project(project_path)
    
    print(f"\nSupervised migration completed.")
    print(f"Success: {result.get('success')}")
    if result.get('success'):
        print(f"Duration: {result.get('duration', 0):.2f} seconds")
        print(f"Total messages: {result.get('messages', 0)}")
        print("\nFinal result:")
        print("-" * 60)
        print(result.get('result', 'No result'))
    else:
        print(f"Error: {result.get('error')}")