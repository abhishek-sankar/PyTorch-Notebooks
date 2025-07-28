"""
Intelligent Java Migration System - Main Orchestrator

This module implements the LangGraph-based orchestration system for automating
Java 8/11/17 to Java 21 migrations with Spring Boot 3.x and Jakarta EE support.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

@dataclass
class MigrationConfig:
    """Configuration for the migration process"""
    target_java_version: str = "21"
    target_spring_version: str = "6.0"
    target_spring_boot_version: str = "3.2"
    enable_jakarta_migration: bool = True
    enable_junit5_migration: bool = True
    dry_run: bool = False
    backup_enabled: bool = True
    max_retry_attempts: int = 3
    openrewrite_recipes: List[str] = field(default_factory=lambda: [
        "org.openrewrite.java.migrate.Java8toJava11",
        "org.openrewrite.java.migrate.Java11toJava17", 
        "org.openrewrite.java.migrate.Java17toJava21",
        "org.openrewrite.java.spring.boot3.UpgradeSpringBoot_3_2",
        "org.openrewrite.java.migrate.jakarta.JavaxMigrationToJakarta"
    ])


class MigrationState(TypedDict):
    """State managed by the LangGraph orchestrator"""
    # Repository information
    repository_path: str
    project_name: str
    
    # Analysis results
    current_java_version: Optional[str]
    dependencies: List[Dict[str, Any]]
    migration_complexity: str  # "simple", "moderate", "complex"
    
    # Migration tracking
    current_phase: str  # "analysis", "planning", "execution", "validation", "complete"
    completed_phases: List[str]
    
    # Results and errors
    applied_recipes: List[str]
    compilation_errors: List[str]
    test_failures: List[str]
    fixes_applied: List[str]
    
    # Agent communication
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Status tracking
    success: bool
    requires_human_intervention: bool
    error_message: Optional[str]
    
    # Metadata
    start_time: str
    end_time: Optional[str]
    config: MigrationConfig


class MigrationOrchestrator:
    """
    Main orchestrator for the Java migration system using LangGraph.
    
    This class implements an agent-based architecture where different specialized
    agents handle analysis, execution, and error fixing with intelligent routing.
    """
    
    def __init__(self, llm: Optional[BaseChatModel] = None, config: Optional[MigrationConfig] = None):
        """
        Initialize the migration orchestrator.
        
        Args:
            llm: Language model for agents. Defaults to Claude Sonnet.
            config: Migration configuration. Uses defaults if not provided.
        """
        self.llm = llm or ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=16384,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        # self.llm = llm or ChatAnthropic(
        #     model="claude-3-sonnet-20240229",
        #     temperature=0.1,
        #     max_tokens=4000
        # )
        self.config = config or MigrationConfig()
        
        # Initialize checkpointer for state persistence
        self.checkpointer = SqliteSaver.from_conn_string(":memory:")
        
        # Build the state graph
        self.graph = self._build_graph()
        
        logger.info("Migration orchestrator initialized")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for migration orchestration"""
        
        # Create the state graph
        workflow = StateGraph(MigrationState)
        
        # Add nodes for each phase
        workflow.add_node("initialize", self._initialize_migration)
        workflow.add_node("analyze_repository", self._analyze_repository)
        workflow.add_node("plan_migration", self._plan_migration)
        workflow.add_node("execute_recipes", self._execute_recipes)
        workflow.add_node("fix_errors", self._fix_errors)
        workflow.add_node("validate_tests", self._validate_tests)
        workflow.add_node("generate_report", self._generate_report)
        workflow.add_node("human_escalation", self._human_escalation)
        
        # Define the workflow edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "analyze_repository")
        workflow.add_edge("analyze_repository", "plan_migration")
        workflow.add_edge("plan_migration", "execute_recipes")
        
        # Conditional edges for error handling
        workflow.add_conditional_edges(
            "execute_recipes",
            self._should_fix_errors,
            {
                "fix_errors": "fix_errors",
                "validate_tests": "validate_tests",
                "human_escalation": "human_escalation"
            }
        )
        
        workflow.add_conditional_edges(
            "fix_errors", 
            self._errors_fixed,
            {
                "execute_recipes": "execute_recipes",
                "validate_tests": "validate_tests", 
                "human_escalation": "human_escalation"
            }
        )
        
        workflow.add_conditional_edges(
            "validate_tests",
            self._tests_passing,
            {
                "fix_errors": "fix_errors",
                "generate_report": "generate_report",
                "human_escalation": "human_escalation"
            }
        )
        
        workflow.add_edge("generate_report", END)
        workflow.add_edge("human_escalation", END)
        
        # Compile the graph with checkpointing
        return workflow.compile(checkpointer=self.checkpointer)
    
    def migrate_repository(self, repository_path: str) -> Dict[str, Any]:
        """
        Main entry point for repository migration.
        
        Args:
            repository_path: Path to the Java repository to migrate
            
        Returns:
            Dictionary with migration results and status
        """
        logger.info(f"Starting migration for repository: {repository_path}")
        
        # Initialize state
        initial_state: MigrationState = {
            "repository_path": repository_path,
            "project_name": Path(repository_path).name,
            "current_java_version": None,
            "dependencies": [],
            "migration_complexity": "unknown",
            "current_phase": "initialization",
            "completed_phases": [],
            "applied_recipes": [],
            "compilation_errors": [],
            "test_failures": [],
            "fixes_applied": [],
            "messages": [HumanMessage(content=f"Begin migration of {repository_path}")],
            "success": False,
            "requires_human_intervention": False,
            "error_message": None,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "config": self.config
        }
        
        try:
            # Execute the migration workflow
            final_state = self.graph.invoke(
                initial_state, 
                config={"configurable": {"thread_id": f"migration_{datetime.now().timestamp()}"}}
            )
            
            return {
                "success": final_state["success"],
                "project_name": final_state["project_name"],
                "phases_completed": final_state["completed_phases"],
                "applied_recipes": final_state["applied_recipes"],
                "fixes_applied": final_state["fixes_applied"],
                "requires_human_intervention": final_state["requires_human_intervention"],
                "error_message": final_state.get("error_message"),
                "duration": self._calculate_duration(final_state["start_time"], final_state.get("end_time")),
                "final_state": final_state
            }
            
        except Exception as e:
            logger.error(f"Migration failed with exception: {e}")
            return {
                "success": False,
                "error": str(e),
                "project_name": Path(repository_path).name
            }
    
    # Node implementations
    
    def _initialize_migration(self, state: MigrationState) -> MigrationState:
        """Initialize the migration process"""
        logger.info("Initializing migration process")
        
        repo_path = Path(state["repository_path"])
        if not repo_path.exists():
            state["error_message"] = f"Repository path does not exist: {repo_path}"
            state["requires_human_intervention"] = True
            return state
        
        # Check for Maven project
        pom_xml = repo_path / "pom.xml"
        if not pom_xml.exists():
            state["error_message"] = "No pom.xml found - not a Maven project"
            state["requires_human_intervention"] = True
            return state
        
        state["current_phase"] = "initialization"
        state["completed_phases"].append("initialization")
        
        state["messages"].append(AIMessage(content="Migration initialized successfully"))
        
        return state
    
    def _analyze_repository(self, state: MigrationState) -> MigrationState:
        """Analyze the repository structure and dependencies"""
        logger.info("Analyzing repository")
        
        from agents.analysis_agent import AnalysisAgent
        
        analysis_agent = AnalysisAgent(self.llm)
        analysis_result = analysis_agent.analyze(state["repository_path"])
        
        # Update state with analysis results
        state["current_java_version"] = analysis_result.get("java_version")
        state["dependencies"] = analysis_result.get("dependencies", [])
        state["migration_complexity"] = analysis_result.get("complexity", "moderate")
        
        state["current_phase"] = "analysis"
        state["completed_phases"].append("analysis")
        
        state["messages"].append(AIMessage(
            content=f"Repository analysis complete. Java version: {state['current_java_version']}, "
                   f"Dependencies: {len(state['dependencies'])}, "
                   f"Complexity: {state['migration_complexity']}"
        ))
        
        return state
    
    def _plan_migration(self, state: MigrationState) -> MigrationState:
        """Plan the migration strategy based on analysis"""
        logger.info("Planning migration strategy")
        
        # Simple strategy selection based on current Java version
        recipes_to_apply = []
        current_version = state["current_java_version"]
        
        if current_version in ["8", "1.8"]:
            recipes_to_apply.extend([
                "org.openrewrite.java.migrate.Java8toJava11",
                "org.openrewrite.java.migrate.Java11toJava17",
                "org.openrewrite.java.migrate.Java17toJava21"
            ])
        elif current_version == "11":
            recipes_to_apply.extend([
                "org.openrewrite.java.migrate.Java11toJava17",
                "org.openrewrite.java.migrate.Java17toJava21"
            ])
        elif current_version == "17":
            recipes_to_apply.append("org.openrewrite.java.migrate.Java17toJava21")
        
        # Add framework-specific recipes
        if state["config"].enable_jakarta_migration:
            recipes_to_apply.append("org.openrewrite.java.migrate.jakarta.JavaxMigrationToJakarta")
        
        if state["config"].enable_junit5_migration:
            recipes_to_apply.append("org.openrewrite.java.testing.junit5.JUnit4to5Migration")
        
        # Store planned recipes
        state["planned_recipes"] = recipes_to_apply
        state["current_phase"] = "planning"
        state["completed_phases"].append("planning")
        
        state["messages"].append(AIMessage(
            content=f"Migration strategy planned. Will apply {len(recipes_to_apply)} recipes: {recipes_to_apply}"
        ))
        
        return state
    
    def _execute_recipes(self, state: MigrationState) -> MigrationState:
        """Execute OpenRewrite recipes"""
        logger.info("Executing OpenRewrite recipes")
        
        from agents.execution_agent import ExecutionAgent
        
        execution_agent = ExecutionAgent(self.llm)
        execution_result = execution_agent.execute_recipes(
            state["repository_path"], 
            state.get("planned_recipes", [])
        )
        
        state["applied_recipes"].extend(execution_result.get("applied_recipes", []))
        state["compilation_errors"].extend(execution_result.get("compilation_errors", []))
        
        state["current_phase"] = "execution"
        state["completed_phases"].append("execution")
        
        state["messages"].append(AIMessage(
            content=f"Recipe execution complete. Applied: {len(state['applied_recipes'])}, "
                   f"Errors: {len(state['compilation_errors'])}"
        ))
        
        return state
    
    def _fix_errors(self, state: MigrationState) -> MigrationState:
        """Fix compilation and other errors"""
        logger.info("Fixing compilation errors")
        
        from agents.error_agent import ErrorFixingAgent
        
        error_agent = ErrorFixingAgent(self.llm)
        fixing_result = error_agent.fix_errors(
            state["repository_path"],
            state["compilation_errors"]
        )
        
        state["fixes_applied"].extend(fixing_result.get("fixes_applied", []))
        state["compilation_errors"] = fixing_result.get("remaining_errors", [])
        
        state["current_phase"] = "error_fixing"
        if "error_fixing" not in state["completed_phases"]:
            state["completed_phases"].append("error_fixing")
        
        state["messages"].append(AIMessage(
            content=f"Error fixing complete. Applied {len(fixing_result.get('fixes_applied', []))} fixes. "
                   f"Remaining errors: {len(state['compilation_errors'])}"
        ))
        
        return state
    
    def _validate_tests(self, state: MigrationState) -> MigrationState:
        """Validate that tests pass after migration"""
        logger.info("Validating tests")
        
        from tools.command_executor import CommandExecutor
        
        executor = CommandExecutor()
        test_result = executor.run_tests(state["repository_path"])
        
        state["test_failures"] = test_result.get("failures", [])
        
        state["current_phase"] = "validation"
        state["completed_phases"].append("validation")
        
        if not state["test_failures"]:
            state["success"] = True
        
        state["messages"].append(AIMessage(
            content=f"Test validation complete. Failures: {len(state['test_failures'])}"
        ))
        
        return state
    
    def _generate_report(self, state: MigrationState) -> MigrationState:
        """Generate final migration report"""
        logger.info("Generating migration report")
        
        state["current_phase"] = "complete"
        state["end_time"] = datetime.now().isoformat()
        state["completed_phases"].append("complete")
        
        # Generate comprehensive report (implementation in separate module)
        from utils.report_generator import ReportGenerator
        report_generator = ReportGenerator()
        report_path = report_generator.generate_report(state)
        
        state["messages"].append(AIMessage(
            content=f"Migration complete! Report generated at: {report_path}"
        ))
        
        return state
    
    def _human_escalation(self, state: MigrationState) -> MigrationState:
        """Handle cases requiring human intervention"""
        logger.info("Escalating to human intervention")
        
        state["requires_human_intervention"] = True
        state["current_phase"] = "human_escalation"
        state["end_time"] = datetime.now().isoformat()
        
        escalation_message = self._create_escalation_message(state)
        state["messages"].append(AIMessage(content=escalation_message))
        
        return state
    
    # Conditional edge functions
    
    def _should_fix_errors(self, state: MigrationState) -> str:
        """Determine if errors need fixing after recipe execution"""
        if state["compilation_errors"]:
            if len(state["compilation_errors"]) > 10:  # Too many errors
                return "human_escalation"
            return "fix_errors"
        return "validate_tests"
    
    def _errors_fixed(self, state: MigrationState) -> str:
        """Determine next step after error fixing"""
        if state["compilation_errors"]:
            # Check retry count to avoid infinite loops
            error_fixing_attempts = state["completed_phases"].count("error_fixing")
            if error_fixing_attempts >= self.config.max_retry_attempts:
                return "human_escalation"
            return "execute_recipes"  # Try re-executing recipes
        return "validate_tests"
    
    def _tests_passing(self, state: MigrationState) -> str:
        """Determine next step after test validation"""
        if state["test_failures"]:
            if len(state["test_failures"]) > 5:  # Too many test failures
                return "human_escalation"
            return "fix_errors"
        return "generate_report"
    
    # Utility methods
    
    def _calculate_duration(self, start_time: str, end_time: Optional[str]) -> str:
        """Calculate migration duration"""
        if not end_time:
            return "In progress"
        
        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)
        duration = end - start
        
        return str(duration)
    
    def _create_escalation_message(self, state: MigrationState) -> str:
        """Create human escalation message"""
        return f"""
Migration requires human intervention for project: {state['project_name']}

Current Phase: {state['current_phase']}
Completed Phases: {', '.join(state['completed_phases'])}

Issues:
- Compilation Errors: {len(state['compilation_errors'])}
- Test Failures: {len(state['test_failures'])}

Error Details:
{chr(10).join(state['compilation_errors'][:5])}  # Show first 5 errors

Please review the project and provide manual fixes where needed.
"""


if __name__ == "__main__":
    # Example usage
    orchestrator = MigrationOrchestrator()
    result = orchestrator.migrate_repository("./xsync")
    print(json.dumps(result, indent=2))