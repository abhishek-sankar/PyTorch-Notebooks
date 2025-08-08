"""
Supervisor-Based Migration Orchestrator
Uses LangGraph's create_supervisor for intelligent agent management
"""
import os
from datetime import datetime
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from src.tools.command_executor import mvn_compile, mvn_test, run_command
from prompts.prompt_loader import (
    get_supervisor_prompt,
    get_migration_request,
    get_analysis_expert_prompt,
    get_execution_expert_prompt,
    get_error_expert_prompt
)


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
        # Import tools from existing analysis agent
        from src.tools import all_tools
        # Filter to analysis-relevant tools
        analysis_tools = [tool for tool in all_tools if tool.name in [
            'read_pom', 'get_java_version', 'list_dependencies', 'read_file', 
            'list_java_files', 'search_files', 'mvn_rewrite_discover',
            'suggest_recipes_for_java_version', 'get_available_recipes'
        ]]
        return analysis_tools
    
    def _get_execution_tools(self):
        """Get tools for execution agent"""
        from src.tools import all_tools
        # Filter to execution-relevant tools
        execution_tools = [tool for tool in all_tools if tool.name in [
            'read_pom', 'update_java_version', 'add_openrewrite_plugin',
            'configure_openrewrite_recipes', 'mvn_rewrite_run', 
            'mvn_rewrite_run_recipe', 'run_command', 'read_file', 'write_file'
        ]]
        return execution_tools
    
    def _get_error_tools(self):
        """Get tools for error agent"""
        from src.tools import all_tools
        # Filter to error-fixing-relevant tools
        error_tools = [tool for tool in all_tools if tool.name in [
            'read_file', 'write_file', 'find_replace', 'search_files',
            'list_java_files', 'read_pom', 'update_java_version'
        ]]
        return error_tools
    
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