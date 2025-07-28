"""
Command Executor Tool for Java Migration System

Provides safe command execution with timeout, monitoring, and result parsing.
Used by agents for running Maven builds, tests, and other system commands.
"""

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import signal
import psutil

logger = logging.getLogger(__name__)


class CommandExecutor:
    """
    Safe command execution with monitoring and timeout capabilities.
    
    Provides agents with the ability to execute system commands
    while monitoring resource usage, handling timeouts, and parsing results.
    """
    
    def __init__(self, default_timeout: int = 300):
        self.default_timeout = default_timeout
        self.execution_log: List[Dict[str, Any]] = []
        self.active_processes: Dict[int, subprocess.Popen] = {}
        
    def execute_command(
        self, 
        command: List[str], 
        working_dir: Union[str, Path] = None,
        timeout: int = None,
        capture_output: bool = True,
        env_vars: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Execute a system command with monitoring.
        
        Args:
            command: Command and arguments as a list
            working_dir: Working directory for the command
            timeout: Timeout in seconds (uses default if None)
            capture_output: Whether to capture stdout/stderr
            env_vars: Additional environment variables
            
        Returns:
            Dictionary with execution results
        """
        timeout = timeout or self.default_timeout
        working_dir = Path(working_dir) if working_dir else Path.cwd()
        
        logger.info(f"Executing command: {' '.join(command)} in {working_dir}")
        
        start_time = time.time()
        result = {
            "command": command,
            "working_dir": str(working_dir),
            "start_time": datetime.now().isoformat(),
            "timeout": timeout,
            "success": False
        }
        
        try:
            # Prepare environment
            env = os.environ.copy()
            if env_vars:
                env.update(env_vars)
            
            # Start the process
            process = subprocess.Popen(
                command,
                cwd=working_dir,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                text=True,
                env=env,
                preexec_fn=os.setsid if os.name != 'nt' else None  # Create process group on Unix
            )
            
            # Register active process
            self.active_processes[process.pid] = process
            
            try:
                # Monitor process with timeout
                stdout, stderr = process.communicate(timeout=timeout)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Update result
                result.update({
                    "success": process.returncode == 0,
                    "exit_code": process.returncode,
                    "stdout": stdout or "",
                    "stderr": stderr or "",
                    "execution_time": execution_time,
                    "end_time": datetime.now().isoformat(),
                    "pid": process.pid
                })
                
                if process.returncode == 0:
                    logger.info(f"Command completed successfully in {execution_time:.2f}s")
                else:
                    logger.warning(f"Command failed with exit code {process.returncode}")
                
            except subprocess.TimeoutExpired:
                logger.error(f"Command timed out after {timeout} seconds")
                
                # Kill the process and its children
                self._kill_process_tree(process.pid)
                
                # Try to get partial output
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    stdout, stderr = "", ""
                
                result.update({
                    "success": False,
                    "exit_code": -1,
                    "stdout": stdout or "",
                    "stderr": stderr or "",
                    "execution_time": timeout,
                    "end_time": datetime.now().isoformat(),
                    "error": "Command timed out",
                    "timeout_occurred": True
                })
                
            finally:
                # Clean up process registry
                if process.pid in self.active_processes:
                    del self.active_processes[process.pid]
                    
        except FileNotFoundError:
            logger.error(f"Command not found: {command[0]}")
            result.update({
                "success": False,
                "error": f"Command not found: {command[0]}",
                "exit_code": -1
            })
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            result.update({
                "success": False,
                "error": str(e),
                "exit_code": -1,
                "execution_time": time.time() - start_time
            })
        
        # Log the execution
        self.execution_log.append(result)
        
        return result
    
    def run_maven_compile(self, project_path: Union[str, Path], clean: bool = False) -> Dict[str, Any]:
        """
        Run Maven compile command.
        
        Args:
            project_path: Path to the Maven project
            clean: Whether to run clean before compile
            
        Returns:
            Dictionary with compilation results
        """
        logger.info("Running Maven compile")
        
        command = ["mvn"]
        if clean:
            command.append("clean")
        command.append("compile")
        
        result = self.execute_command(command, working_dir=project_path, timeout=600)
        
        # Parse compilation-specific information
        if result.get("stderr"):
            compilation_errors = self._parse_maven_errors(result["stderr"])
            result["compilation_errors"] = compilation_errors
            result["error_count"] = len(compilation_errors)
        
        return result
    
    def run_maven_test(self, project_path: Union[str, Path], test_class: str = None) -> Dict[str, Any]:
        """
        Run Maven test command.
        
        Args:
            project_path: Path to the Maven project
            test_class: Specific test class to run (optional)
            
        Returns:
            Dictionary with test results
        """
        logger.info("Running Maven tests")
        
        command = ["mvn", "test"]
        if test_class:
            command.extend(["-Dtest=" + test_class])
        
        result = self.execute_command(command, working_dir=project_path, timeout=900)
        
        # Parse test-specific information
        if result.get("stdout"):
            test_results = self._parse_maven_test_results(result["stdout"])
            result.update(test_results)
        
        return result
    
    def run_tests(self, project_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Run project tests and return detailed results.
        
        Args:
            project_path: Path to the project
            
        Returns:
            Dictionary with test execution results and failures
        """
        logger.info("Running project tests")
        
        # First try Maven test
        maven_result = self.run_maven_test(project_path)
        
        result = {
            "test_framework": "maven",
            "success": maven_result["success"],
            "execution_time": maven_result.get("execution_time", 0),
            "stdout": maven_result.get("stdout", ""),
            "stderr": maven_result.get("stderr", "")
        }
        
        # Parse test failures
        if not maven_result["success"]:
            failures = self._parse_test_failures(maven_result.get("stdout", "") + maven_result.get("stderr", ""))
            result["failures"] = failures
            result["failure_count"] = len(failures)
        else:
            result["failures"] = []
            result["failure_count"] = 0
        
        return result
    
    def run_maven_package(self, project_path: Union[str, Path], skip_tests: bool = True) -> Dict[str, Any]:
        """
        Run Maven package command.
        
        Args:
            project_path: Path to the Maven project
            skip_tests: Whether to skip tests during packaging
            
        Returns:
            Dictionary with packaging results
        """
        logger.info("Running Maven package")
        
        command = ["mvn", "package"]
        if skip_tests:
            command.append("-DskipTests")
        
        return self.execute_command(command, working_dir=project_path, timeout=900)
    
    def run_openrewrite_command(self, project_path: Union[str, Path], recipe: str) -> Dict[str, Any]:
        """
        Run OpenRewrite command for a specific recipe.
        
        Args:
            project_path: Path to the Maven project
            recipe: OpenRewrite recipe to execute
            
        Returns:
            Dictionary with OpenRewrite execution results
        """
        logger.info(f"Running OpenRewrite recipe: {recipe}")
        
        command = [
            "mvn",
            "org.openrewrite.maven:rewrite-maven-plugin:run",
            f"-Drewrite.activeRecipes={recipe}"
        ]
        
        result = self.execute_command(command, working_dir=project_path, timeout=1200)
        
        # Parse OpenRewrite-specific information
        if result.get("stdout"):
            rewrite_info = self._parse_openrewrite_output(result["stdout"])
            result.update(rewrite_info)
        
        return result
    
    def kill_all_active_processes(self) -> Dict[str, Any]:
        """
        Kill all active processes started by this executor.
        
        Returns:
            Dictionary with cleanup results
        """
        logger.info("Killing all active processes")
        
        killed_count = 0
        errors = []
        
        for pid, process in list(self.active_processes.items()):
            try:
                if process.poll() is None:  # Process still running
                    self._kill_process_tree(pid)
                    killed_count += 1
                    logger.info(f"Killed process {pid}")
                
                del self.active_processes[pid]
                
            except Exception as e:
                logger.error(f"Failed to kill process {pid}: {e}")
                errors.append(f"PID {pid}: {e}")
        
        return {
            "killed_count": killed_count,
            "errors": errors,
            "remaining_processes": len(self.active_processes)
        }
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get the log of all command executions"""
        return self.execution_log.copy()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for debugging"""
        try:
            return {
                "platform": os.name,
                "working_directory": str(Path.cwd()),
                "java_home": os.environ.get("JAVA_HOME"),
                "maven_home": os.environ.get("MAVEN_HOME"),
                "path": os.environ.get("PATH"),
                "active_processes": len(self.active_processes),
                "cpu_count": os.cpu_count(),
                "memory_available": psutil.virtual_memory().available if psutil else "unknown"
            }
        except Exception as e:
            return {"error": str(e)}
    
    # Private helper methods
    
    def _kill_process_tree(self, pid: int):
        """Kill a process and all its children"""
        try:
            if os.name == 'nt':
                # Windows
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], 
                             capture_output=True, check=False)
            else:
                # Unix-like systems
                try:
                    # Kill the process group
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    time.sleep(2)  # Give it time to terminate gracefully
                    
                    # Force kill if still running
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Process already terminated
                        
                except ProcessLookupError:
                    pass  # Process already terminated
                    
        except Exception as e:
            logger.warning(f"Failed to kill process tree {pid}: {e}")
    
    def _parse_maven_errors(self, stderr: str) -> List[str]:
        """Parse Maven compilation errors from stderr"""
        errors = []
        
        if not stderr:
            return errors
        
        lines = stderr.split('\n')
        for line in lines:
            line = line.strip()
            if '[ERROR]' in line and 'compilation failure' not in line.lower():
                # Clean up the error message
                error = line.replace('[ERROR]', '').strip()
                if error and len(error) > 10:  # Filter out very short messages
                    errors.append(error)
        
        return errors
    
    def _parse_maven_test_results(self, stdout: str) -> Dict[str, Any]:
        """Parse Maven test results from stdout"""
        results = {
            "tests_run": 0,
            "tests_failed": 0,
            "tests_errors": 0,
            "tests_skipped": 0
        }
        
        if not stdout:
            return results
        
        # Look for test summary line
        import re
        
        # Pattern for "Tests run: X, Failures: Y, Errors: Z, Skipped: W"
        pattern = r"Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)"
        matches = re.findall(pattern, stdout)
        
        if matches:
            # Take the last match (final summary)
            last_match = matches[-1]
            results.update({
                "tests_run": int(last_match[0]),
                "tests_failed": int(last_match[1]),
                "tests_errors": int(last_match[2]),
                "tests_skipped": int(last_match[3])
            })
        
        return results
    
    def _parse_test_failures(self, output: str) -> List[str]:
        """Parse test failures from command output"""
        failures = []
        
        if not output:
            return failures
        
        lines = output.split('\n')
        in_failure = False
        current_failure = []
        
        for line in lines:
            line = line.strip()
            
            # Start of a test failure
            if 'FAILURE' in line or 'ERROR' in line or 'Failed tests:' in line:
                in_failure = True
                current_failure = [line]
                continue
            
            # End of failure section
            if in_failure:
                if line.startswith('Tests run:') or line.startswith('Results:') or not line:
                    if current_failure:
                        failures.append('\n'.join(current_failure))
                        current_failure = []
                    in_failure = False
                else:
                    current_failure.append(line)
        
        # Add any remaining failure
        if current_failure:
            failures.append('\n'.join(current_failure))
        
        return failures[:10]  # Limit to first 10 failures
    
    def _parse_openrewrite_output(self, stdout: str) -> Dict[str, Any]:
        """Parse OpenRewrite execution output"""
        info = {
            "files_changed": 0,
            "changes_made": 0,
            "recipes_applied": []
        }
        
        if not stdout:
            return info
        
        lines = stdout.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for OpenRewrite-specific output patterns
            if 'files changed' in line.lower():
                # Try to extract number of files changed
                import re
                match = re.search(r'(\d+)\s+files?\s+changed', line.lower())
                if match:
                    info["files_changed"] = int(match.group(1))
            
            if 'recipe' in line.lower() and 'applied' in line.lower():
                info["recipes_applied"].append(line)
        
        return info