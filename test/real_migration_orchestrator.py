#!/usr/bin/env python3
"""
Real Migration Orchestrator - LangChain/LangGraph with Actual Tools

This is the working migration system that actually performs migrations
using real tools and LLM agents, not simulations.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Import our real tools
from real_migration_tools import (
    execute_openrewrite_recipe,
    compile_maven_project, 
    run_maven_tests,
    analyze_java_project
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()


@dataclass
class MigrationConfig:
    """Configuration for the migration process"""
    target_java_version: str = "21"
    dry_run: bool = False
    max_retry_attempts: int = 3
    enable_jakarta_migration: bool = True
    enable_junit5_migration: bool = True


class MigrationState(TypedDict):
    """State managed by the LangGraph orchestrator"""
    # Project info
    repository_path: str
    project_name: str
    
    # Current status
    current_phase: str
    completed_phases: List[str]
    
    # Analysis results
    java_version: Optional[str]
    needs_migration: bool
    recommended_recipes: List[str]
    
    # Migration results
    applied_recipes: List[str]
    compilation_errors: List[str]
    test_failures: List[str]
    changes_made: List[str]
    files_modified: List[str]
    
    # Agent messages
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Final status
    success: bool
    error_message: Optional[str]
    config: MigrationConfig


class RealMigrationOrchestrator:
    """
    Real migration orchestrator that actually performs migrations.
    
    This system uses LangChain agents with real tools to execute
    OpenRewrite recipes and transform Java code.
    """
    
    def __init__(self, config: Optional[MigrationConfig] = None):
        self.config = config or MigrationConfig()
        
        # Initialize LLM
        if os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=16384
            )
        else:
            raise ValueError("No LLM API key found. Set OPENAI_API_KEY")
        
        # Create tools list - these are the REAL tools the agents will use
        self.tools = [
            analyze_java_project,
            execute_openrewrite_recipe,
            compile_maven_project,
            run_maven_tests
        ]
        
        # Create the LLM with tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build the graph
        self.app = self._build_graph()
        
        logger.info("Real migration orchestrator initialized with actual tools")
    
    def _build_graph(self):
        """Build the LangGraph workflow with real tool integration"""
        
        workflow = StateGraph(MigrationState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Define the flow
        workflow.set_entry_point("agent")
        
        # The agent decides whether to use tools or finish
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # After tools, always go back to agent
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def _agent_node(self, state: MigrationState):
        """Main agent that coordinates the migration process"""
        
        # Get the current phase and determine what to do next
        current_phase = state.get("current_phase", "start")
        messages = state["messages"]
        
        # Create system message based on current phase
        if current_phase == "start":
            system_prompt = f"""You are a Java migration expert. Your task is to migrate the Java project at {state['repository_path']} to Java 21.

Available tools:
- analyze_java_project: Analyze the current state of the project
- execute_openrewrite_recipe: Execute OpenRewrite recipes to transform code  
- compile_maven_project: Compile the project to check for errors
- run_maven_tests: Run tests to verify migration success

Start by analyzing the project to understand its current state."""
            
        elif current_phase == "analysis_complete":
            system_prompt = f"""The project analysis is complete. Now plan and execute the migration recipes.

Project details:
- Java version: {state.get('java_version')}
- Needs migration: {state.get('needs_migration')}
- Recommended recipes: {state.get('recommended_recipes')}

Execute the recommended OpenRewrite recipes. Set dry_run=False to make actual changes."""
            
        elif current_phase == "recipes_applied":
            system_prompt = f"""Migration recipes have been applied. Now validate the changes.

Applied recipes: {state.get('applied_recipes')}
Changes made: {len(state.get('changes_made', []))} changes
Files modified: {len(state.get('files_modified', []))} files

Compile the project to check for compilation errors."""
            
        elif current_phase == "compilation_checked":
            if state.get('compilation_errors'):
                system_prompt = f"""Compilation found errors: {state.get('compilation_errors')}
Fix these errors and try compiling again."""
            else:
                system_prompt = "Compilation successful! Now run tests to verify everything works."
                
        elif current_phase == "tests_run":
            if state.get('test_failures'):
                system_prompt = f"""Tests failed: {state.get('test_failures')}
The migration may need additional fixes."""
            else:
                system_prompt = "All tests passed! Migration completed successfully."
                
        else:
            system_prompt = "Continue with the migration process based on the current state."
        
        # Add system message
        messages_with_system = [
            SystemMessage(content=system_prompt)
        ] + messages
        
        # Get response from LLM
        response = self.llm_with_tools.invoke(messages_with_system)
        
        # Update messages
        return {"messages": [response]}
    
    def _should_continue(self, state: MigrationState):
        """Determine whether to continue with tools or end"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the last message has tool calls, continue
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        else:
            return "end"
    
    def migrate_repository(self, repository_path: str) -> Dict[str, Any]:
        """Main entry point for repository migration"""
        logger.info(f"Starting REAL migration for repository: {repository_path}")
        
        # Initialize state
        initial_state: MigrationState = {
            "repository_path": repository_path,
            "project_name": Path(repository_path).name,
            "current_phase": "start",
            "completed_phases": [],
            "java_version": None,
            "needs_migration": False,
            "recommended_recipes": [],
            "applied_recipes": [],
            "compilation_errors": [],
            "test_failures": [],
            "changes_made": [],
            "files_modified": [],
            "messages": [HumanMessage(content=f"Begin real migration of {repository_path}")],
            "success": False,
            "error_message": None,
            "config": self.config
        }
        
        try:
            # Execute the workflow - this will actually perform the migration
            final_state = self.app.invoke(initial_state)
            
            # Determine success based on final state
            success = (
                len(final_state.get("applied_recipes", [])) > 0 and
                len(final_state.get("compilation_errors", [])) == 0 and
                len(final_state.get("test_failures", [])) == 0
            )
            
            # Calculate duration
            duration = "Completed"
            
            return {
                "success": success,
                "project_name": final_state["project_name"],
                "phases_completed": final_state.get("completed_phases", []),
                "applied_recipes": final_state.get("applied_recipes", []),
                "changes_made": final_state.get("changes_made", []),
                "files_modified": final_state.get("files_modified", []),
                "compilation_errors": final_state.get("compilation_errors", []),
                "test_failures": final_state.get("test_failures", []),
                "requires_human_intervention": len(final_state.get("compilation_errors", [])) > 0,
                "error_message": final_state.get("error_message"),
                "duration": duration,
                "final_state": final_state
            }
            
        except Exception as e:
            logger.error(f"Real migration failed with exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "project_name": Path(repository_path).name,
                "applied_recipes": [],
                "changes_made": [],
                "files_modified": []
            }


# Enhanced agent implementation with tool usage
class JavaMigrationAgent:
    """
    Specialized Java migration agent that uses tools systematically.
    """
    
    def __init__(self, llm: BaseChatModel, tools: List):
        self.llm = llm.bind_tools(tools)
        self.tools = tools
    
    def analyze_and_migrate(self, repository_path: str, dry_run: bool = False) -> Dict[str, Any]:
        """Perform complete analysis and migration"""
        
        messages = [
            SystemMessage(content=f"""You are an expert Java migration assistant. Your task is to migrate the Java project at {repository_path} to Java 21.

Follow this process:
1. First, use analyze_java_project to understand the current state
2. Based on the analysis, execute the recommended OpenRewrite recipes with dry_run={dry_run}
3. Compile the project to check for errors
4. Run tests to verify the migration
5. Report the results

Be thorough and use the tools systematically. Make sure to actually execute the recipes and transform the code."""),
            HumanMessage(content=f"Please migrate the Java project at {repository_path} to Java 21. Execute the actual migration recipes.")
        ]
        
        # Keep track of the conversation
        result_data = {
            "applied_recipes": [],
            "changes_made": [],
            "files_modified": [],
            "compilation_errors": [],
            "test_failures": []
        }
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent iteration {iteration}")
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            messages.append(response)
            
            # If there are tool calls, execute them
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    logger.info(f"Executing tool: {tool_call['name']}")
                    
                    # Execute the tool
                    tool_result = self._execute_tool_call(tool_call)
                    
                    # Track results
                    if tool_call['name'] == 'execute_openrewrite_recipe':
                        if tool_result.get('success'):
                            result_data["applied_recipes"].extend(tool_result.get('applied_recipes', []))
                            result_data["changes_made"].extend(tool_result.get('changes_made', []))
                            result_data["files_modified"].extend(tool_result.get('files_modified', []))
                    
                    elif tool_call['name'] == 'compile_maven_project':
                        if not tool_result.get('success'):
                            result_data["compilation_errors"].extend(tool_result.get('errors', []))
                    
                    elif tool_call['name'] == 'run_maven_tests':
                        if not tool_result.get('success'):
                            result_data["test_failures"].extend(tool_result.get('failures', []))
                    
                    # Add tool result to conversation
                    messages.append(AIMessage(content=f"Tool {tool_call['name']} result: {json.dumps(tool_result, indent=2)}"))
            
            else:
                # No more tool calls, agent is done
                break
        
        # Determine success
        success = (
            len(result_data["applied_recipes"]) > 0 and
            len(result_data["compilation_errors"]) == 0 and
            len(result_data["test_failures"]) == 0
        )
        
        return {
            "success": success,
            "applied_recipes": result_data["applied_recipes"],
            "changes_made": result_data["changes_made"],
            "files_modified": result_data["files_modified"],
            "compilation_errors": result_data["compilation_errors"],
            "test_failures": result_data["test_failures"],
            "conversation": [m.content for m in messages if hasattr(m, 'content')]
        }
    
    def _execute_tool_call(self, tool_call: Dict) -> Dict[str, Any]:
        """Execute a tool call and return the result"""
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        # Find the tool function
        tool_func = None
        for tool in self.tools:
            if tool.name == tool_name:
                tool_func = tool
                break
        
        if tool_func:
            try:
                return tool_func.invoke(tool_args)
            except Exception as e:
                return {"error": str(e)}
        else:
            return {"error": f"Tool {tool_name} not found"}


# Compatibility aliases
MigrationOrchestrator = RealMigrationOrchestrator


if __name__ == "__main__":
    # Test the real migration system
    print("🚀 Real Migration System Test")
    print("=" * 50)
    
    # Test with xsync project if available
    test_project = Path("./xsync")
    if test_project.exists():
        print(f"Testing REAL migration with project: {test_project}")
        
        # Create orchestrator
        config = MigrationConfig(dry_run=False)  # REAL MIGRATION
        orchestrator = RealMigrationOrchestrator(config=config)
        
        # Run migration
        result = orchestrator.migrate_repository(str(test_project))
        
        print(f"\n📊 REAL Migration Results:")
        print(f"   Success: {'✅' if result['success'] else '❌'}")
        print(f"   Project: {result['project_name']}")
        print(f"   Applied Recipes: {len(result['applied_recipes'])}")
        print(f"   Changes Made: {len(result['changes_made'])}")
        print(f"   Files Modified: {len(result['files_modified'])}")
        print(f"   Compilation Errors: {len(result['compilation_errors'])}")
        print(f"   Test Failures: {len(result['test_failures'])}")
        
        if result['applied_recipes']:
            print(f"\n✅ REAL RECIPES APPLIED:")
            for recipe in result['applied_recipes']:
                print(f"     - {recipe}")
        
        if result['changes_made']:
            print(f"\n🔄 REAL CHANGES MADE:")
            for change in result['changes_made'][:5]:  # Show first 5
                print(f"     - {change}")
        
        if result['files_modified']:
            print(f"\n📝 FILES ACTUALLY MODIFIED:")
            for file in result['files_modified'][:5]:  # Show first 5
                print(f"     - {file}")
        
    else:
        print("❌ xsync project not found for testing")
        print("Create a test project or modify the path to test real migrations")