"""
LangChain tools for command execution in migration agents
"""
import subprocess
from langchain_core.tools import tool

@tool
def run_command(command: str, cwd: str = ".", timeout: int = 300) -> str:
    """Run a shell command and return the result."""
    try:
        # Use shell=True for commands with complex quoting
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True
        )
        
        output = f"Return code: {result.returncode}\n"
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        
        return output
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def mvn_compile(project_path: str) -> str:
    """Run Maven compile in the specified project directory."""
    return run_command.invoke({"command": "mvn compile", "cwd": project_path})

@tool
def mvn_test(project_path: str) -> str:
    """Run Maven test in the specified project directory."""
    return run_command.invoke({"command": "mvn test", "cwd": project_path})

@tool
def mvn_rewrite_run(project_path: str) -> str:
    """Run OpenRewrite recipes using Maven in the specified project directory."""
    return run_command.invoke({"command": "mvn rewrite:run", "cwd": project_path})

@tool
def git_status(project_path: str) -> str:
    """Get git status for the specified project directory."""
    return run_command.invoke({"command": "git status --porcelain", "cwd": project_path})

@tool
def git_add_all(project_path: str) -> str:
    """Git add all changes in the specified project directory."""
    return run_command.invoke({"command": "git add .", "cwd": project_path})

@tool
def git_commit(project_path: str, message: str) -> str:
    """Git commit with message in the specified project directory."""
    return run_command.invoke({"command": f'git commit -m "{message}"', "cwd": project_path})

@tool
def mvn_rewrite_discover(project_path: str) -> str:
    """Discover available OpenRewrite recipes using Maven."""
    return run_command.invoke({"command": "mvn rewrite:discover", "cwd": project_path})

@tool
def mvn_rewrite_run_recipe(project_path: str, recipe_name: str) -> str:
    """Run a specific OpenRewrite recipe using Maven command line."""
    return run_command.invoke({"command": f"mvn rewrite:run -Drewrite.activeRecipes={recipe_name}", "cwd": project_path})

@tool
def mvn_rewrite_dry_run(project_path: str) -> str:
    """Run OpenRewrite in dry-run mode to see what changes would be made."""
    return run_command.invoke({"command": "mvn rewrite:dryRun", "cwd": project_path})

# Collect all command tools
command_tools = [run_command, mvn_compile, mvn_test, mvn_rewrite_run, mvn_rewrite_discover, mvn_rewrite_run_recipe, mvn_rewrite_dry_run, git_status, git_add_all, git_commit]