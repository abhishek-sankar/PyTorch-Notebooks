# Copy the content from the first artifact here
# Due to size, you'll need to copy this from the Claude interface
"""
Java 11 to 21 Migration Pipeline
Hybrid approach using OpenRewrite + AI Agents
"""

import json
import subprocess
import re
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== Data Models =====

class ErrorType(Enum):
    CANNOT_FIND_SYMBOL = "CANNOT_FIND_SYMBOL"
    INCOMPATIBLE_TYPES = "INCOMPATIBLE_TYPES"
    METHOD_DOES_NOT_OVERRIDE = "METHOD_DOES_NOT_OVERRIDE"
    PACKAGE_DOES_NOT_EXIST = "PACKAGE_DOES_NOT_EXIST"
    CANNOT_ACCESS = "CANNOT_ACCESS"
    UNKNOWN = "UNKNOWN"

@dataclass
class CompilationError:
    file_path: str
    line_number: int
    column_number: int
    error_type: ErrorType
    error_code: str
    details: Dict[str, str]
    raw_message: str

@dataclass
class TestFailure:
    test_class: str
    test_method: str
    failure_type: str
    stack_trace: str
    assertion_message: Optional[str] = None

# ===== Phase 1: OpenRewrite Orchestration =====

class OpenRewriteOrchestrator:
    """Manages the sequential application of OpenRewrite recipes"""
    
    def __init__(self, project_path: str, build_tool: str = "maven"):
        self.project_path = Path(project_path)
        self.build_tool = build_tool
        self.recipes = self._get_migration_recipes()
        
    def _get_migration_recipes(self) -> List[Dict[str, str]]:
        """Returns the ordered list of recipes to apply"""
        return [
            {
                "phase": "1_test_framework",
                "recipe": "org.openrewrite.java.spring.boot2.SpringBoot2JUnit4to5Migration",
                "description": "Migrate JUnit 4 to JUnit 5"
            },
            {
                "phase": "2_jakarta_namespace",
                "recipe": "org.openrewrite.java.migrate.jakarta.JavaxMigrationToJakarta",
                "description": "Migrate javax.* to jakarta.*"
            },
            {
                "phase": "3_java_platform",
                "recipe": "org.openrewrite.java.migrate.UpgradeToJava21",
                "description": "Upgrade Java 11 to Java 21"
            },
            {
                "phase": "4_spring_boot",
                "recipe": "org.openrewrite.java.spring.boot3.UpgradeSpringBoot_3_3",
                "description": "Upgrade to Spring Boot 3.x"
            }
        ]
    
    def run_phase(self, phase_index: int, dry_run: bool = True) -> Dict[str, any]:
        """Execute a single OpenRewrite phase"""
        recipe = self.recipes[phase_index]
        logger.info(f"Running Phase {phase_index + 1}: {recipe['description']}")
        
        if self.build_tool == "maven":
            cmd = [
                "mvn", "rewrite:run",
                f"-Drewrite.activeRecipes={recipe['recipe']}",
                f"-Drewrite.exportDatatables=true"
            ]
            if dry_run:
                cmd.append("-Drewrite.dryRun=true")
        else:  # gradle
            cmd = [
                "./gradlew", "rewriteRun",
                f"--activeRecipe={recipe['recipe']}",
                "--exportDatatables"
            ]
            if dry_run:
                cmd.append("-Drewrite.dryRun=true")
        
        result = subprocess.run(cmd, cwd=self.project_path, capture_output=True, text=True)
        
        return {
            "phase": recipe['phase'],
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "patch_file": self.project_path / "rewrite.patch" if dry_run else None
        }
    
    def analyze_results(self, dry_run_results: Dict) -> Dict[str, List[str]]:
        """Parse OpenRewrite data tables to identify problem files"""
        analysis = {
            "high_churn_files": [],
            "failed_files": [],
            "dependency_issues": []
        }
        
        # Parse data tables (simplified - real implementation would parse CSV/JSON)
        data_tables_path = self.project_path / ".rewrite" / "data-tables"
        
        if (data_tables_path / "org.openrewrite.table.SourcesFileErrors.csv").exists():
            # Parse files that failed processing
            with open(data_tables_path / "org.openrewrite.table.SourcesFileErrors.csv") as f:
                for line in f.readlines()[1:]:  # Skip header
                    file_path = line.split(",")[0].strip()
                    analysis["failed_files"].append(file_path)
        
        return analysis

# ===== Phase 2: AI Repair Agent =====

class AIRepairAgent:
    """Autonomous agent for fixing compilation and test errors"""
    
    def __init__(self, llm_provider: str = "claude", model: str = "claude-3-5-sonnet-20241022"):
        self.llm_provider = llm_provider
        self.model = model
        self.tools = self._initialize_tools()
        self.memory = []  # Agent's short-term memory
        
    def _initialize_tools(self) -> Dict[str, callable]:
        """Initialize the agent's toolset"""
        return {
            "read_range": self.tool_read_range,
            "get_class_methods": self.tool_get_class_methods,
            "search_code_base": self.tool_search_code_base,
            "find_similar_api_calls": self.tool_find_similar_api_calls,
            "write_fix": self.tool_write_fix,
            "run_single_test": self.tool_run_single_test,
            "analyze_stack_trace": self.tool_analyze_stack_trace,
            "express_hypothesis": self.tool_express_hypothesis,
            "goal_accomplished": self.tool_goal_accomplished
        }
    
    def repair_compilation_error(self, error: CompilationError, project_path: Path) -> Optional[Dict]:
        """Main entry point for repairing a single compilation error"""
        self.memory = []  # Reset memory for new repair task
        self.project_path = project_path
        
        context = self._build_initial_context(error)
        
        max_iterations = 10
        for i in range(max_iterations):
            # Get next action from LLM
            action = self._get_next_action(context, error)
            
            if action["command"] == "goal_accomplished":
                return {
                    "success": True,
                    "patch": self.memory[-1].get("patch") if self.memory else None,
                    "reasoning": action.get("thoughts", "")
                }
            
            # Execute the chosen tool
            tool_result = self._execute_tool(action)
            
            # Update memory and context
            self.memory.append({
                "step": i + 1,
                "action": action,
                "result": tool_result
            })
            
            context = self._update_context(context, tool_result)
        
        return {"success": False, "reason": "Max iterations reached"}
    
    def _build_initial_context(self, error: CompilationError) -> Dict:
        """Build the initial context for the agent"""
        # Read code around the error
        code_snippet = self.tool_read_range(
            error.file_path,
            max(1, error.line_number - 10),
            error.line_number + 10
        )
        
        return {
            "error": error,
            "code_snippet": code_snippet,
            "file_path": error.file_path,
            "project_type": self._detect_project_type()
        }
    
    def _get_next_action(self, context: Dict, error: CompilationError) -> Dict:
        """Query LLM for next action"""
        prompt = self._build_prompt(context, error)
        
        # This is where you'd integrate with your LLM of choice
        # For now, returning a mock response
        if error.error_type == ErrorType.CANNOT_FIND_SYMBOL:
            return {
                "thoughts": "The symbol is not found. I should search for similar symbols in the codebase.",
                "command": "search_code_base",
                "args": {"query": error.details.get("symbol_name", "")}
            }
        
        return {"command": "express_hypothesis", "args": {"hypothesis": "Need more context"}}
    
    def _build_prompt(self, context: Dict, error: CompilationError) -> str:
        """Build the prompt for the LLM"""
        memory_str = "\n".join([
            f"Step {m['step']}: {m['action']['command']} -> {m['result'].get('summary', 'Done')}"
            for m in self.memory[-5:]  # Last 5 steps
        ])
        
        return f"""You are an expert Java software engineer specializing in migrations from Java 11 to Java 21 with Spring Boot 3.

Current Goal: Fix the compilation error

Error Details:
- Type: {error.error_type.value}
- File: {error.file_path}
- Line: {error.line_number}
- Message: {error.raw_message}

Code Context:
```java
{context.get('code_snippet', 'N/A')}
```

Previous Actions:
{memory_str if memory_str else "None"}

Available Tools:
- read_range: Read specific lines from a file
- search_code_base: Search for code patterns
- find_similar_api_calls: Find usage examples
- write_fix: Apply a code fix
- express_hypothesis: State your reasoning
- goal_accomplished: Mark task complete

Respond with JSON:
{{
    "thoughts": "Your reasoning about the error and next step",
    "command": "tool_name",
    "args": {{...}}
}}"""
    
    # Tool implementations
    def tool_read_range(self, file_path: str, start_line: int, end_line: int) -> str:
        """Read a range of lines from a file"""
        try:
            with open(self.project_path / file_path, 'r') as f:
                lines = f.readlines()
                return ''.join(lines[start_line-1:end_line])
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def tool_search_code_base(self, query: str) -> List[Dict]:
        """Search for code patterns in the project"""
        # Simplified grep-based search
        cmd = ["grep", "-r", "-n", query, "--include=*.java", str(self.project_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        matches = []
        for line in result.stdout.splitlines()[:10]:  # Limit results
            parts = line.split(":", 2)
            if len(parts) >= 3:
                matches.append({
                    "file": parts[0].replace(str(self.project_path) + "/", ""),
                    "line": int(parts[1]),
                    "snippet": parts[2].strip()
                })
        
        return matches
    
    def tool_write_fix(self, file_path: str, patch_instructions: Dict) -> Dict:
        """Apply a fix to the code"""
        # This would implement the actual patching logic
        # For now, returning mock success
        return {"success": True, "build_log": "Mock build successful"}
    
    def tool_get_class_methods(self, file_path: str, class_name: str) -> List[str]:
        """Extract method signatures from a class"""
        # Simplified implementation
        return ["public void method1()", "private String method2(int param)"]
    
    def tool_find_similar_api_calls(self, method_signature: str) -> List[Dict]:
        """Find similar API usage patterns"""
        return self.tool_search_code_base(method_signature)
    
    def tool_run_single_test(self, test_class: str, test_method: str) -> Dict:
        """Run a single test"""
        cmd = ["mvn", "test", f"-Dtest={test_class}#{test_method}"]
        result = subprocess.run(cmd, cwd=self.project_path, capture_output=True, text=True)
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr
        }
    
    def tool_analyze_stack_trace(self, stack_trace: str) -> Dict:
        """Analyze a stack trace"""
        lines = stack_trace.splitlines()
        if not lines:
            return {}
        
        # Extract key information
        exception_line = lines[0]
        exception_type = exception_line.split(":")[0].strip()
        
        # Find the first line in our code (not library code)
        our_code_line = None
        for line in lines[1:]:
            if "at com.example" in line:  # Adjust based on your package
                our_code_line = line.strip()
                break
        
        return {
            "exception_type": exception_type,
            "first_application_frame": our_code_line,
            "full_trace": stack_trace
        }
    
    def tool_express_hypothesis(self, hypothesis: str) -> Dict:
        """Express a hypothesis about the problem"""
        logger.info(f"Agent hypothesis: {hypothesis}")
        return {"status": "OK"}
    
    def tool_goal_accomplished(self) -> Dict:
        """Mark the current repair task as complete"""
        return {"status": "Terminated"}
    
    def _detect_project_type(self) -> str:
        """Detect if this is a Maven or Gradle project"""
        if (self.project_path / "pom.xml").exists():
            return "maven"
        elif (self.project_path / "build.gradle").exists():
            return "gradle"
        return "unknown"

# ===== Error Parsing =====

class ErrorParser:
    """Parses compiler and test outputs into structured formats"""
    
    @staticmethod
    def parse_compilation_errors(build_output: str) -> List[CompilationError]:
        """Parse Java compilation errors from build output"""
        errors = []
        
        # Pattern for Maven compiler errors
        error_pattern = re.compile(
            r'\[ERROR\] (.+?):\[(\d+),(\d+)\] (.+)'
        )
        
        # Pattern for error details
        symbol_pattern = re.compile(
            r'symbol:\s+(\w+)\s+(\w+)'
        )
        
        location_pattern = re.compile(
            r'location:\s+(\w+)\s+(.+)'
        )
        
        current_file = None
        current_error = None
        
        lines = build_output.splitlines()
        for i, line in enumerate(lines):
            error_match = error_pattern.match(line)
            if error_match:
                # Save previous error if exists
                if current_error:
                    errors.append(current_error)
                
                file_path = error_match.group(1)
                line_num = int(error_match.group(2))
                col_num = int(error_match.group(3))
                message = error_match.group(4)
                
                # Determine error type
                error_type = ErrorType.UNKNOWN
                if "cannot find symbol" in message:
                    error_type = ErrorType.CANNOT_FIND_SYMBOL
                elif "incompatible types" in message:
                    error_type = ErrorType.INCOMPATIBLE_TYPES
                elif "does not override" in message:
                    error_type = ErrorType.METHOD_DOES_NOT_OVERRIDE
                elif "package" in message and "does not exist" in message:
                    error_type = ErrorType.PACKAGE_DOES_NOT_EXIST
                
                current_error = CompilationError(
                    file_path=file_path,
                    line_number=line_num,
                    column_number=col_num,
                    error_type=error_type,
                    error_code="compiler.err." + error_type.value.lower(),
                    details={},
                    raw_message=message
                )
            
            # Parse additional error details
            elif current_error:
                symbol_match = symbol_pattern.search(line)
                if symbol_match:
                    current_error.details["symbol_type"] = symbol_match.group(1)
                    current_error.details["symbol_name"] = symbol_match.group(2)
                
                location_match = location_pattern.search(line)
                if location_match:
                    current_error.details["location_type"] = location_match.group(1)
                    current_error.details["location_name"] = location_match.group(2)
        
        # Don't forget the last error
        if current_error:
            errors.append(current_error)
        
        return errors
    
    @staticmethod
    def parse_test_failures(test_output: str) -> List[TestFailure]:
        """Parse test failures from Maven Surefire or Gradle output"""
        failures = []
        
        # For Maven Surefire XML reports
        surefire_dir = Path("target/surefire-reports")
        if surefire_dir.exists():
            for xml_file in surefire_dir.glob("TEST-*.xml"):
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                for testcase in root.findall(".//testcase"):
                    failure = testcase.find("failure")
                    if failure is not None:
                        test_failure = TestFailure(
                            test_class=testcase.get("classname"),
                            test_method=testcase.get("name"),
                            failure_type=failure.get("type", "Unknown"),
                            stack_trace=failure.text or "",
                            assertion_message=failure.get("message")
                        )
                        failures.append(test_failure)
        
        return failures

# ===== Main Pipeline Orchestrator =====

class MigrationPipeline:
    """Main orchestrator for the entire migration pipeline"""
    
    def __init__(self, project_path: str, max_repair_attempts: int = 100):
        self.project_path = Path(project_path)
        self.max_repair_attempts = max_repair_attempts
        self.openrewrite = OpenRewriteOrchestrator(project_path)
        self.ai_agent = AIRepairAgent()
        self.error_parser = ErrorParser()
        
    def run_full_migration(self) -> Dict[str, any]:
        """Execute the complete migration pipeline"""
        results = {
            "openrewrite_phases": [],
            "compilation_fixes": [],
            "test_fixes": [],
            "success": False
        }
        
        # Phase 1: Run OpenRewrite recipes
        logger.info("Starting Phase 1: OpenRewrite transformations")
        for i in range(len(self.openrewrite.recipes)):
            phase_result = self.openrewrite.run_phase(i, dry_run=False)
            results["openrewrite_phases"].append(phase_result)
            
            if not phase_result["success"]:
                logger.error(f"OpenRewrite phase {i+1} failed")
                return results
        
        # Commit OpenRewrite changes
        self._commit_changes("OpenRewrite: Bulk migration complete")
        
        # Phase 2: AI-assisted compilation fixes
        logger.info("Starting Phase 2: AI-assisted compilation fixes")
        compilation_success = self._run_compilation_repair_loop(results)
        
        if not compilation_success:
            logger.error("Failed to achieve compilation")
            return results
        
        # Phase 3: AI-assisted test fixes
        logger.info("Starting Phase 3: AI-assisted test fixes")
        test_success = self._run_test_repair_loop(results)
        
        results["success"] = compilation_success and test_success
        return results
    
    def _run_compilation_repair_loop(self, results: Dict) -> bool:
        """Run the compilation repair loop"""
        for attempt in range(self.max_repair_attempts):
            # Try to build
            build_result = self._build_project(skip_tests=True)
            
            if build_result["success"]:
                logger.info("Compilation successful!")
                return True
            
            # Parse errors
            errors = self.error_parser.parse_compilation_errors(build_result["output"])
            
            if not errors:
                logger.error("Build failed but no compilation errors found")
                return False
            
            # Fix first error
            error = errors[0]
            logger.info(f"Attempting to fix: {error.error_type.value} in {error.file_path}:{error.line_number}")
            
            repair_result = self.ai_agent.repair_compilation_error(error, self.project_path)
            
            if repair_result and repair_result["success"]:
                results["compilation_fixes"].append({
                    "error": error,
                    "fix": repair_result,
                    "attempt": attempt + 1
                })
                
                # Commit the fix
                self._commit_changes(f"AI-FIX: {error.error_type.value} in {error.file_path}:{error.line_number}")
            else:
                logger.warning(f"Failed to fix error: {error.raw_message}")
        
        return False
    
    def _run_test_repair_loop(self, results: Dict) -> bool:
        """Run the test repair loop"""
        for attempt in range(self.max_repair_attempts):
            # Run tests
            test_result = self._build_project(skip_tests=False)
            
            if test_result["success"]:
                logger.info("All tests passing!")
                return True
            
            # Parse test failures
            failures = self.error_parser.parse_test_failures(test_result["output"])
            
            if not failures:
                logger.error("Tests failed but no failures parsed")
                return False
            
            # Fix first failure
            failure = failures[0]
            logger.info(f"Attempting to fix test: {failure.test_class}::{failure.test_method}")
            
            # Convert test failure to a "compilation error" format for the agent
            # In a real implementation, you'd have a separate test repair method
            pseudo_error = CompilationError(
                file_path=f"src/test/java/{failure.test_class.replace('.', '/')}.java",
                line_number=0,  # Would need to parse from stack trace
                column_number=0,
                error_type=ErrorType.UNKNOWN,
                error_code="test.failure",
                details={"test_method": failure.test_method},
                raw_message=failure.stack_trace
            )
            
            repair_result = self.ai_agent.repair_compilation_error(pseudo_error, self.project_path)
            
            if repair_result and repair_result["success"]:
                results["test_fixes"].append({
                    "failure": failure,
                    "fix": repair_result,
                    "attempt": attempt + 1
                })
                
                self._commit_changes(f"AI-FIX: Test {failure.test_class}::{failure.test_method}")
        
        return False
    
    def _build_project(self, skip_tests: bool = True) -> Dict[str, any]:
        """Build the project and capture output"""
        if (self.project_path / "pom.xml").exists():
            cmd = ["mvn", "clean", "install"]
            if skip_tests:
                cmd.append("-DskipTests")
        else:
            cmd = ["./gradlew", "build"]
            if skip_tests:
                cmd.extend(["-x", "test"])
        
        result = subprocess.run(cmd, cwd=self.project_path, capture_output=True, text=True)
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout + "\n" + result.stderr,
            "return_code": result.returncode
        }
    
    def _commit_changes(self, message: str):
        """Commit changes to git"""
        subprocess.run(["git", "add", "-A"], cwd=self.project_path)
        subprocess.run(["git", "commit", "-m", message], cwd=self.project_path)


# ===== Usage Example =====

if __name__ == "__main__":
    # Example usage
    project_path = "/path/to/your/java/project"
    
    # Create and run the pipeline
    pipeline = MigrationPipeline(project_path)
    results = pipeline.run_full_migration()
    
    # Print summary
    print(f"Migration {'succeeded' if results['success'] else 'failed'}")
    print(f"OpenRewrite phases completed: {len(results['openrewrite_phases'])}")
    print(f"Compilation fixes applied: {len(results['compilation_fixes'])}")
    print(f"Test fixes applied: {len(results['test_fixes'])}")