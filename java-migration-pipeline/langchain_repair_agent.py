# Copy the content from the second artifact here
# Due to size, you'll need to copy this from the Claude interface
"""
LangChain-based AI Repair Agent for Java Migration
Implements the AI agent component using LangChain with tool support
"""

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import Tool, StructuredTool
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import subprocess
import os
import re
from pathlib import Path
import json
import ast

# ===== Tool Definitions =====

class CodeRange(BaseModel):
    """Input model for reading code ranges"""
    file_path: str = Field(description="Path to the file relative to project root")
    start_line: int = Field(description="Starting line number (1-indexed)")
    end_line: int = Field(description="Ending line number (inclusive)")

class SearchQuery(BaseModel):
    """Input model for code search"""
    query: str = Field(description="Search query or pattern")
    file_pattern: str = Field(default="*.java", description="File pattern to search")
    max_results: int = Field(default=10, description="Maximum results to return")

class CodePatch(BaseModel):
    """Input model for applying code fixes"""
    file_path: str = Field(description="Path to the file to patch")
    old_text: str = Field(description="Exact text to replace")
    new_text: str = Field(description="Replacement text")
    
class TestExecution(BaseModel):
    """Input model for running tests"""
    test_class: str = Field(description="Fully qualified test class name")
    test_method: Optional[str] = Field(default=None, description="Specific test method")

class JavaMigrationTools:
    """Collection of tools for Java migration tasks"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.build_tool = self._detect_build_tool()
        
    def _detect_build_tool(self) -> str:
        """Detect if this is a Maven or Gradle project"""
        if (self.project_path / "pom.xml").exists():
            return "maven"
        elif (self.project_path / "build.gradle").exists():
            return "gradle"
        return "unknown"
    
    def read_code_range(self, file_path: str, start_line: int, end_line: int) -> str:
        """Read a specific range of lines from a Java file"""
        try:
            full_path = self.project_path / file_path
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Adjust for 0-indexing
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            
            result_lines = []
            for i in range(start_idx, end_idx):
                result_lines.append(f"{i+1}: {lines[i].rstrip()}")
            
            return "\n".join(result_lines)
        except Exception as e:
            return f"Error searching codebase: {str(e)}"
    
    def find_method_usages(self, method_name: str) -> str:
        """Find all usages of a specific method"""
        # Search for method calls (simplified pattern)
        pattern = f"\\.{method_name}\\s*\\("
        return self.search_codebase(pattern, max_results=20)
    
    def get_class_structure(self, file_path: str, class_name: str) -> str:
        """Extract class structure including methods and fields"""
        try:
            full_path = self.project_path / file_path
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use AST parsing for Java (simplified version)
            # In production, use a proper Java parser like javalang
            class_pattern = rf"class\s+{class_name}\s*(?:extends\s+\w+)?\s*(?:implements\s+[\w\s,]+)?\s*\{{"
            class_match = re.search(class_pattern, content)
            
            if not class_match:
                return f"Class {class_name} not found in {file_path}"
            
            # Extract methods (simplified)
            method_pattern = r"(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*(?:synchronized)?\s*[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)"
            methods = re.findall(method_pattern, content)
            
            # Extract fields (simplified)
            field_pattern = r"(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*[\w<>\[\]]+\s+(\w+)\s*[;=]"
            fields = re.findall(field_pattern, content)
            
            result = f"Class: {class_name}\n"
            result += f"Fields: {', '.join(set(fields))}\n"
            result += f"Methods: {', '.join(set(methods))}"
            
            return result
            
        except Exception as e:
            return f"Error analyzing class structure: {str(e)}"
    
    def apply_code_patch(self, file_path: str, old_text: str, new_text: str) -> str:
        """Apply a patch to a file by replacing exact text"""
        try:
            full_path = self.project_path / file_path
            
            # Read current content
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if old_text exists
            if old_text not in content:
                return "Error: The specified text to replace was not found in the file"
            
            # Apply the patch
            new_content = content.replace(old_text, new_text, 1)  # Replace only first occurrence
            
            # Write back
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # Try to compile the single file to get immediate feedback
            compile_result = self._compile_single_file(file_path)
            
            return f"Patch applied successfully. Compilation result: {compile_result}"
            
        except Exception as e:
            return f"Error applying patch: {str(e)}"
    
    def _compile_single_file(self, file_path: str) -> str:
        """Attempt to compile a single Java file"""
        if self.build_tool == "maven":
            cmd = ["mvn", "compile", "-pl", ".", "-am"]
        else:
            cmd = ["./gradlew", "compileJava"]
        
        result = subprocess.run(cmd, cwd=self.project_path, capture_output=True, text=True)
        
        if result.returncode == 0:
            return "Success"
        else:
            # Extract relevant error messages
            error_lines = [line for line in result.stdout.splitlines() + result.stderr.splitlines() 
                          if "[ERROR]" in line or "error:" in line]
            return "\n".join(error_lines[:5])  # Return first 5 error lines
    
    def run_single_test(self, test_class: str, test_method: Optional[str] = None) -> str:
        """Run a single test class or method"""
        try:
            if self.build_tool == "maven":
                if test_method:
                    cmd = ["mvn", "test", f"-Dtest={test_class}#{test_method}"]
                else:
                    cmd = ["mvn", "test", f"-Dtest={test_class}"]
            else:
                if test_method:
                    cmd = ["./gradlew", "test", f"--tests", f"{test_class}.{test_method}"]
                else:
                    cmd = ["./gradlew", "test", f"--tests", f"{test_class}"]
            
            result = subprocess.run(cmd, cwd=self.project_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                return "Test passed successfully"
            else:
                # Extract failure information
                output = result.stdout + "\n" + result.stderr
                failure_pattern = r"(.*Exception.*|.*Error.*|.*assert.*)"
                failures = re.findall(failure_pattern, output)
                
                return f"Test failed:\n" + "\n".join(failures[:10])
                
        except Exception as e:
            return f"Error running test: {str(e)}"
    
    def analyze_stack_trace(self, stack_trace: str) -> str:
        """Analyze a stack trace to identify the root cause"""
        lines = stack_trace.strip().splitlines()
        
        if not lines:
            return "Empty stack trace"
        
        analysis = []
        
        # Extract exception type and message
        first_line = lines[0]
        exception_match = re.match(r"([\w\.]+Exception|[\w\.]+Error):\s*(.*)", first_line)
        if exception_match:
            exception_type = exception_match.group(1)
            message = exception_match.group(2)
            analysis.append(f"Exception Type: {exception_type}")
            analysis.append(f"Message: {message}")
        
        # Find first occurrence in project code
        project_frame = None
        for line in lines[1:]:
            if "at " in line and ".java:" in line:
                # Check if it's from the project (heuristic: not from common libraries)
                if not any(pkg in line for pkg in ["java.", "javax.", "org.springframework", "org.junit"]):
                    frame_match = re.search(r"at\s+([\w\.]+)\(([\w]+\.java):(\d+)\)", line)
                    if frame_match:
                        project_frame = {
                            "method": frame_match.group(1),
                            "file": frame_match.group(2),
                            "line": frame_match.group(3)
                        }
                        break
        
        if project_frame:
            analysis.append(f"\nFirst project occurrence:")
            analysis.append(f"  Method: {project_frame['method']}")
            analysis.append(f"  File: {project_frame['file']}")
            analysis.append(f"  Line: {project_frame['line']}")
        
        return "\n".join(analysis)

# ===== LangChain Agent Setup =====

class JavaMigrationAgent:
    """LangChain-based agent for Java migration tasks"""
    
    def __init__(self, project_path: str, llm_provider: str = "anthropic", model: str = "claude-3-5-sonnet-20241022"):
        self.project_path = Path(project_path)
        self.tools_impl = JavaMigrationTools(project_path)
        self.llm = self._create_llm(llm_provider, model)
        self.tools = self._create_tools()
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=2000,
            return_messages=True
        )
        self.agent_executor = self._create_agent()
    
    def _create_llm(self, provider: str, model: str):
        """Create the LLM instance"""
        if provider == "anthropic":
            return ChatAnthropic(
                model=model,
                temperature=0,
                max_tokens=4096
            )
        elif provider == "openai":
            return ChatOpenAI(
                model=model,
                temperature=0,
                max_tokens=4096
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools from our implementation"""
        return [
            StructuredTool(
                name="read_code_range",
                description="Read specific lines from a Java source file. Use this to examine code around errors.",
                func=lambda file_path, start_line, end_line: self.tools_impl.read_code_range(
                    file_path, start_line, end_line
                ),
                args_schema=CodeRange
            ),
            StructuredTool(
                name="search_codebase",
                description="Search for code patterns or text across the entire Java project.",
                func=lambda query, file_pattern="*.java", max_results=10: self.tools_impl.search_codebase(
                    query, file_pattern, max_results
                ),
                args_schema=SearchQuery
            ),
            Tool(
                name="find_method_usages",
                description="Find all places where a specific method is called in the codebase.",
                func=self.tools_impl.find_method_usages
            ),
            Tool(
                name="get_class_structure",
                description="Get the structure of a Java class including its methods and fields.",
                func=lambda args: self.tools_impl.get_class_structure(
                    args.split(",")[0].strip(), 
                    args.split(",")[1].strip()
                )
            ),
            StructuredTool(
                name="apply_code_patch",
                description="Apply a fix by replacing exact text in a file. Returns compilation result.",
                func=lambda file_path, old_text, new_text: self.tools_impl.apply_code_patch(
                    file_path, old_text, new_text
                ),
                args_schema=CodePatch
            ),
            StructuredTool(
                name="run_single_test",
                description="Run a specific test class or method to validate fixes.",
                func=lambda test_class, test_method=None: self.tools_impl.run_single_test(
                    test_class, test_method
                ),
                args_schema=TestExecution
            ),
            Tool(
                name="analyze_stack_trace",
                description="Analyze a Java stack trace to identify the root cause and location of an error.",
                func=self.tools_impl.analyze_stack_trace
            )
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent"""
        system_message = """You are an expert Java software engineer specializing in large-scale migrations from Java 11 to Java 21, Spring Boot 2 to 3, and Jakarta EE migrations.

Your task is to fix compilation and test errors that arise during the migration process. You have access to tools that let you:
1. Read and search through the codebase
2. Apply fixes to the code
3. Run tests to validate your changes

Key migration patterns to remember:
- javax.* packages are now jakarta.*
- Spring Security's WebSecurityConfigurerAdapter is removed - use SecurityFilterChain beans
- JUnit 4 @Before/@After are now @BeforeEach/@AfterEach in JUnit 5
- Many Spring Boot properties have been renamed
- Java 21 has new APIs like SequencedCollection

Always:
1. First understand the error by reading the relevant code
2. Search for similar patterns in the codebase to understand the correct usage
3. Apply minimal, targeted fixes
4. Validate your fixes when possible

Be systematic and thorough. Think step by step."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
        
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=15,
            return_intermediate_steps=True
        )
    
    def fix_compilation_error(self, error: Dict[str, Any]) -> Dict[str, Any]:
        """Fix a single compilation error"""
        # Format the error information for the agent
        error_context = f"""
Fix the following compilation error:

File: {error['file_path']}
Line: {error['line_number']}
Error Type: {error['error_type']}
Message: {error['raw_message']}

Additional Details:
{json.dumps(error.get('details', {}), indent=2)}

Start by reading the code around line {error['line_number']} in {error['file_path']}, then analyze the error and apply a fix.
"""
        
        try:
            result = self.agent_executor.invoke({"input": error_context})
            
            # Extract the final answer and intermediate steps
            return {
                "success": True,
                "output": result["output"],
                "intermediate_steps": result.get("intermediate_steps", []),
                "memory": self.memory.chat_memory.messages
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "memory": self.memory.chat_memory.messages
            }
    
    def fix_test_failure(self, failure: Dict[str, Any]) -> Dict[str, Any]:
        """Fix a failing test"""
        failure_context = f"""
Fix the following test failure:

Test Class: {failure['test_class']}
Test Method: {failure['test_method']}
Failure Type: {failure['failure_type']}

Stack Trace:
{failure['stack_trace']}

Start by analyzing the stack trace, then read the relevant code and fix the issue.
"""
        
        try:
            result = self.agent_executor.invoke({"input": failure_context})
            
            return {
                "success": True,
                "output": result["output"],
                "intermediate_steps": result.get("intermediate_steps", []),
                "memory": self.memory.chat_memory.messages
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "memory": self.memory.chat_memory.messages
            }

# ===== Integration with Main Pipeline =====

class LangChainMigrationPipeline:
    """Enhanced pipeline using LangChain agents"""
    
    def __init__(self, project_path: str, llm_provider: str = "anthropic"):
        self.project_path = Path(project_path)
        self.agent = JavaMigrationAgent(project_path, llm_provider)
        
    def process_compilation_errors(self, errors: List[Dict]) -> List[Dict]:
        """Process a list of compilation errors using the agent"""
        results = []
        
        for i, error in enumerate(errors):
            print(f"\nProcessing error {i+1}/{len(errors)}")
            print(f"Error: {error['error_type']} in {error['file_path']}:{error['line_number']}")
            
            # Let the agent fix it
            fix_result = self.agent.fix_compilation_error(error)
            
            results.append({
                "error": error,
                "fix_result": fix_result,
                "success": fix_result["success"]
            })
            
            if fix_result["success"]:
                print(f"✓ Fixed successfully")
                # Commit the change
                self._commit_fix(error, fix_result)
            else:
                print(f"✗ Failed to fix: {fix_result.get('error', 'Unknown error')}")
        
        return results
    
    def _commit_fix(self, error: Dict, fix_result: Dict):
        """Commit a successful fix to version control"""
        commit_message = f"AI-FIX: {error['error_type']} in {error['file_path']}:{error['line_number']}"
        
        subprocess.run(["git", "add", "-A"], cwd=self.project_path)
        subprocess.run(["git", "commit", "-m", commit_message], cwd=self.project_path)

# ===== Example Usage =====

if __name__ == "__main__":
    # Example of using the LangChain-based agent
    project_path = "/path/to/your/java/project"
    
    # Create the agent
    pipeline = LangChainMigrationPipeline(project_path)
    
    # Example compilation error
    sample_error = {
        "file_path": "src/main/java/com/example/UserService.java",
        "line_number": 42,
        "column_number": 15,
        "error_type": "CANNOT_FIND_SYMBOL",
        "error_code": "compiler.err.cant.resolve.location",
        "details": {
            "symbol_name": "ServletException",
            "symbol_type": "class",
            "location_type": "package",
            "location_name": "javax.servlet"
        },
        "raw_message": "cannot find symbol\n  symbol:   class ServletException\n  location: package javax.servlet"
    }
    
    # Fix the error
    result = pipeline.agent.fix_compilation_error(sample_error)
    
    print("\nFix Result:")
    print(f"Success: {result['success']}")
    print(f"Output: {result['output']}")
    
    # The agent would have:
    # 1. Read the code around line 42
    # 2. Identified that javax.servlet.ServletException needs to be jakarta.servlet.ServletException
    # 3. Applied the fix
    # 4. Validated the compilation"Error reading file: {str(e)}"
    
    def search_codebase(self, query: str, file_pattern: str = "*.java", max_results: int = 10) -> str:
        """Search for code patterns across the project"""
        try:
            # Use ripgrep if available, otherwise fall back to grep
            if subprocess.run(["which", "rg"], capture_output=True).returncode == 0:
                cmd = ["rg", "-n", "--type", "java", query, str(self.project_path)]
            else:
                cmd = ["grep", "-r", "-n", query, f"--include={file_pattern}", str(self.project_path)]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return "No matches found"
            
            matches = []
            for i, line in enumerate(result.stdout.splitlines()):
                if i >= max_results:
                    break
                    
                # Parse the output
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    file_path = parts[0].replace(str(self.project_path) + "/", "")
                    line_num = parts[1]
                    code = parts[2].strip()
                    matches.append(f"{file_path}:{line_num} - {code}")
            
            return "\n".join(matches) if matches else "No matches found"
            
        except Exception as e:
            return f