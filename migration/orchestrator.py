"""
Simple Action-Based Migration Orchestrator
"""
import os
from datetime import datetime
from src.agents.analysis_agent import AnalysisAgent
from src.agents.execution_agent import ExecutionAgent
from src.agents.error_agent import ErrorAgent


class MigrationOrchestrator:
    """Simple orchestrator that calls agents in sequence"""
    
    def __init__(self):
        print("Initializing Migration Orchestrator...")
        
        # Initialize agents
        self.analysis_agent = AnalysisAgent()
        self.execution_agent = ExecutionAgent()
        self.error_agent = ErrorAgent()
        
        print("All agents initialized")
    
    def analyze_project(self, project_path: str) -> str:
        """Analyze the project and return recommendations"""
        print(f"\nAnalyzing project: {project_path}")
        
        if not os.path.exists(project_path):
            return f"Error: Project path does not exist: {project_path}"
        
        # Check for pom.xml
        pom_path = os.path.join(project_path, "pom.xml")
        if not os.path.exists(pom_path):
            return f"Error: No pom.xml found in {project_path}"
        
        print("Project validation passed")
        print("Starting analysis agent...")
        
        # Call analysis agent
        start_time = datetime.now()
        analysis_result = self.analysis_agent.analyze_project(project_path)
        duration = datetime.now() - start_time
        
        print(f"Analysis completed in {duration.total_seconds():.2f} seconds")
        print("\nAnalysis Result:")
        print("-" * 60)
        print(analysis_result)
        print("-" * 60)
        
        return analysis_result
    
    def execute_migration(self, project_path: str, analysis_result: str) -> str:
        """Execute migration based on analysis"""
        print(f"\nExecuting migration: {project_path}")
        print("Starting execution agent...")
        
        start_time = datetime.now()
        # Pass analysis result to execution agent - let it extract recipes
        execution_result = self.execution_agent.execute_recipes(project_path, analysis_result)
        duration = datetime.now() - start_time
        
        print(f"Execution completed in {duration.total_seconds():.2f} seconds")
        print("\nExecution Result:")
        print("-" * 60)
        print(execution_result)
        print("-" * 60)
        
        return execution_result
    
    def fix_errors(self, project_path: str, error_message: str) -> str:
        """Fix errors in the project"""
        print(f"\nFixing errors in: {project_path}")
        print(f"Error: {error_message[:200]}...")
        print("Starting error agent...")
        
        start_time = datetime.now()
        fix_result = self.error_agent.fix_error(
            error_message=error_message,
            project_path=project_path,
            last_output=error_message
        )
        duration = datetime.now() - start_time
        
        print(f"Error fixing completed in {duration.total_seconds():.2f} seconds")
        print("\nFix Result:")
        print("-" * 60)
        print(fix_result)
        print("-" * 60)
        
        return fix_result
    
    def run_migration(self, project_path: str) -> str:
        """Run full migration: analyze then execute"""
        print(f"Starting migration for: {project_path}")
        
        # Step 1: Analyze
        analysis_result = self.analyze_project(project_path)
        
        # Step 2: Execute based on analysis
        print("\nPassing analysis to execution agent...")
        execution_result = self.execute_migration(project_path, analysis_result)
        
        return execution_result


if __name__ == "__main__":
    project_path = "/Users/abhisheksankar/Desktop/PyTorch-Notebooks/migration/xsync"
    
    orchestrator = MigrationOrchestrator()
    result = orchestrator.run_migration(project_path)
    
    print(f"\nMigration completed. Result length: {len(result)} characters")