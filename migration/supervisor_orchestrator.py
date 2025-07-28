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

from src.agents.analysis_agent import AnalysisAgent
from src.agents.execution_agent import ExecutionAgent
from src.agents.error_agent import ErrorAgent
from src.tools.command_executor import mvn_compile, mvn_test, run_command


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
            prompt="""You are a Java Migration Analysis Expert.

ROLE: Analyze Java projects and recommend OpenRewrite recipes for migration to Java 21.

RESPONSIBILITIES:
- Examine pom.xml for current Java version and dependencies
- Search source code for migration patterns (javax imports, JUnit versions, etc.)
- Discover available OpenRewrite recipes using Maven
- Recommend specific recipes in order of execution
- Assess migration complexity and risks

INSTRUCTIONS:
- Always start by reading pom.xml to understand current state
- Use mvn_rewrite_discover to find actually available recipes
- Provide specific recipe names that exist and work
- Focus on practical, executable recommendations
- End with a clear list of recommended recipes

Respond ONLY with your analysis results and recipe recommendations.""",
            name="analysis_expert"
        )
        
        # Execution Worker - executes OpenRewrite recipes
        execution_worker = create_react_agent(
            model=ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                temperature=0
            ),
            tools=self._get_execution_tools() + validation_tools,
            prompt="""You are a Java Migration Execution Expert.

ROLE: Execute OpenRewrite recipes and configure Maven projects for Java 21 migration.

RESPONSIBILITIES:
- Configure OpenRewrite plugin in pom.xml
- Execute specific OpenRewrite recipes
- Update Java versions in pom.xml
- Validate changes by compiling project
- Handle Maven and OpenRewrite configuration issues

INSTRUCTIONS:
- Configure OpenRewrite in pom.xml (NOT YAML files)
- Add required dependencies for recipes
- Execute recipes using mvn rewrite:run
- Validate with mvn compile after changes
- Report success/failure clearly

CRITICAL: OpenRewrite MUST be configured in pom.xml. YAML configurations don't work.

Respond ONLY with execution results and status.""",
            name="execution_expert"
        )
        
        # Error Fixing Worker - fixes compilation and build errors
        error_worker = create_react_agent(
            model=ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                temperature=0
            ),
            tools=self._get_error_tools() + validation_tools,
            prompt="""You are a Java Migration Error Fixing Expert.

ROLE: Fix compilation errors, build failures, and migration issues.

RESPONSIBILITIES:
- Analyze compilation and build errors
- Fix Java code issues after migration
- Resolve dependency conflicts
- Fix test failures
- Update deprecated API usage

INSTRUCTIONS:
- Read and understand the specific error messages
- Locate problematic files and fix issues
- Test fixes by compiling and running tests
- Make minimal, targeted changes
- Verify fixes work before concluding

Focus on practical solutions that resolve the specific errors presented.

Respond ONLY with your fix results and validation status.""",
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
            prompt="""You are a Java Migration Supervisor managing a team of specialized agents.

YOUR MISSION: Migrate Java projects from older versions to Java 21 using OpenRewrite.

TEAM MEMBERS:
- analysis_expert: Analyzes projects and recommends OpenRewrite recipes
- execution_expert: Executes OpenRewrite recipes and configures Maven
- error_expert: Fixes compilation errors and build issues

MIGRATION STRATEGY:
1. START with analysis_expert to understand current project state
2. Based on analysis, use execution_expert to apply recommended recipes
3. If errors occur, use error_expert to fix them
4. Validate progress and repeat until complete

SUCCESS CRITERIA:
- Java version upgraded to 21 in pom.xml
- Project compiles successfully (mvn compile passes)
- All tests pass (mvn test passes)

DECISION MAKING:
- Always analyze first if you don't know project state
- Execute recipes based on analysis recommendations
- Fix errors immediately when they occur
- Validate frequently to track progress
- Don't proceed if current step failed

Be systematic and intelligent about which agent to call next based on the current situation."""
        )
        
        return workflow
    
    def migrate_project(self, project_path: str) -> Dict[str, Any]:
        """Start supervised migration"""
        print(f"Starting supervised migration for: {project_path}")
        
        if not os.path.exists(project_path):
            return {"success": False, "error": f"Project path does not exist: {project_path}"}
        
        # Create migration request
        migration_request = f"""Please migrate this Java project to Java 21: {project_path}

INSTRUCTIONS:
1. First analyze the project to understand its current state
2. Execute appropriate OpenRewrite recipes based on analysis
3. Fix any errors that occur during migration
4. Validate that the final result meets success criteria

SUCCESS CRITERIA:
- Java version is 21 in pom.xml
- Project compiles successfully (mvn compile)
- All tests pass (mvn test)

Start by analyzing the project."""
        
        try:
            start_time = datetime.now()
            
            # Invoke supervisor workflow
            result = self.app.invoke({
                "messages": [{"role": "user", "content": migration_request}]
            })
            
            duration = datetime.now() - start_time
            
            print(f"Supervised migration completed in {duration.total_seconds():.2f} seconds")
            
            # Extract final result
            messages = result.get("messages", [])
            final_message = messages[-1] if messages else {}
            final_content = final_message.get("content", "No final message") if isinstance(final_message, dict) else str(final_message)
            
            return {
                "success": True,
                "result": final_content,
                "duration": duration.total_seconds(),
                "messages": len(messages)
            }
            
        except Exception as e:
            print(f"Supervised migration failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


if __name__ == "__main__":
    project_path = "/Users/abhisheksankar/Desktop/PyTorch-Notebooks/migration/xsync"
    
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