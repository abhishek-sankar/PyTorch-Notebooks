"""
Execution Agent for Java Migration System

This agent is responsible for:
- OpenRewrite recipe execution and monitoring
- Command execution with timeout and error handling
- Build system integration (Maven/Gradle)
- Recipe composition and sequencing
- Progress tracking and validation
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import yaml

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from tools.command_executor import CommandExecutor
from tools.openrewrite_client import OpenRewriteClient
from tools.file_operations import FileOperations

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """
    Intelligent agent for executing migration recipes and managing the build process.
    
    Uses LLM reasoning to make decisions about recipe execution order, error handling,
    and adaptive strategy adjustments based on execution results.
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.command_executor = CommandExecutor()
        self.openrewrite_client = OpenRewriteClient()
        self.file_ops = FileOperations()
        
        # System prompt for execution decisions
        self.system_prompt = """You are an expert Java migration execution specialist. Your job is to orchestrate the execution of migration recipes and handle any issues that arise.

You have access to:
1. OpenRewrite recipe execution capabilities
2. Maven/Gradle build system integration
3. Command execution with monitoring
4. File system operations for backup and modification

Your responsibilities:
1. Determine optimal recipe execution order
2. Handle recipe execution errors intelligently
3. Validate recipe application results
4. Make adaptive decisions based on execution outcomes
5. Coordinate with build system validation

Be methodical and cautious. Always validate changes before proceeding to the next step."""
    
    def execute_recipes(self, repository_path: str, recipes: List[str]) -> Dict[str, Any]:
        """
        Execute a list of OpenRewrite recipes on the repository.
        
        Args:
            repository_path: Path to the Java repository
            recipes: List of OpenRewrite recipe identifiers
            
        Returns:
            Dictionary containing execution results and any errors
        """
        logger.info(f"Starting recipe execution for {len(recipes)} recipes")
        
        repo_path = Path(repository_path)
        results = {
            "applied_recipes": [],
            "failed_recipes": [],
            "compilation_errors": [],
            "execution_log": [],
            "success": True
        }
        
        try:
            # Create backup before starting
            backup_path = self._create_backup(repo_path)
            results["backup_path"] = str(backup_path)
            
            # Get LLM-recommended execution order
            execution_plan = self._plan_execution_order(recipes, repository_path)
            results["execution_plan"] = execution_plan
            
            # Execute recipes in planned order
            for recipe in execution_plan["ordered_recipes"]:
                logger.info(f"Executing recipe: {recipe}")
                
                recipe_result = self._execute_single_recipe(repo_path, recipe)
                results["execution_log"].append({
                    "recipe": recipe,
                    "result": recipe_result,
                    "timestamp": self._get_timestamp()
                })
                
                if recipe_result["success"]:
                    results["applied_recipes"].append(recipe)
                    logger.info(f"Successfully applied recipe: {recipe}")
                    
                    # Validate compilation after each recipe
                    compile_result = self._validate_compilation(repo_path)
                    if not compile_result["success"]:
                        logger.warning(f"Compilation issues after recipe {recipe}")
                        results["compilation_errors"].extend(compile_result["errors"])
                        
                        # Decide whether to continue or rollback
                        continue_decision = self._should_continue_after_error(
                            recipe, compile_result["errors"], results
                        )
                        
                        if not continue_decision["continue"]:
                            logger.error(f"Stopping execution due to: {continue_decision['reason']}")
                            results["success"] = False
                            break
                            
                else:
                    results["failed_recipes"].append({
                        "recipe": recipe,
                        "error": recipe_result["error"]
                    })
                    logger.error(f"Failed to apply recipe {recipe}: {recipe_result['error']}")
                    
                    # Decide whether to continue with other recipes
                    continue_decision = self._should_continue_after_failure(recipe, recipe_result["error"])
                    if not continue_decision["continue"]:
                        results["success"] = False
                        break
            
            # Final compilation check
            final_compile = self._validate_compilation(repo_path)
            results["final_compilation"] = final_compile
            
            if not final_compile["success"]:
                results["compilation_errors"].extend(final_compile["errors"])
                results["success"] = False
            
            logger.info(f"Recipe execution completed. Applied: {len(results['applied_recipes'])}, Failed: {len(results['failed_recipes'])}")
            return results
            
        except Exception as e:
            logger.error(f"Recipe execution failed with exception: {e}")
            results["success"] = False
            results["error"] = str(e)
            return results
    
    def _plan_execution_order(self, recipes: List[str], repository_path: str) -> Dict[str, Any]:
        """Use LLM to determine optimal recipe execution order"""
        logger.info("Planning recipe execution order")
        
        # Analyze repository to inform ordering decisions
        repo_context = self._get_repository_context(repository_path)
        
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Plan the optimal execution order for these OpenRewrite recipes:

Recipes to execute: {recipes}

Repository context: {context}

Consider:
1. Dependencies between recipes (Java version upgrades should come first)
2. Framework migrations (Spring Boot, Jakarta EE)
3. Test framework migrations (JUnit 4->5)
4. Potential conflicts or interactions

Provide the recipes in optimal execution order with reasoning for each decision.
Format as a numbered list with explanations.""")
        ])
        
        try:
            response = self.llm.invoke(
                planning_prompt.format_messages(
                    recipes=recipes,
                    context=repo_context
                )
            )
            
            # Parse LLM response to extract ordered recipes
            ordered_recipes = self._parse_recipe_order(response.content, recipes)
            
            return {
                "ordered_recipes": ordered_recipes,
                "reasoning": response.content,
                "original_order": recipes
            }
            
        except Exception as e:
            logger.warning(f"LLM planning failed, using default order: {e}")
            # Fallback to sensible default ordering
            ordered_recipes = self._get_default_recipe_order(recipes)
            return {
                "ordered_recipes": ordered_recipes,
                "reasoning": "Used default ordering due to LLM failure",
                "original_order": recipes
            }
    
    def _execute_single_recipe(self, repo_path: Path, recipe: str) -> Dict[str, Any]:
        """Execute a single OpenRewrite recipe"""
        logger.info(f"Executing OpenRewrite recipe: {recipe}")
        
        try:
            # Use OpenRewrite client to execute recipe
            result = self.openrewrite_client.run_recipe(str(repo_path), recipe)
            
            if result["success"]:
                return {
                    "success": True,
                    "files_modified": result.get("files_modified", []),
                    "changes_made": result.get("changes_made", 0),
                    "execution_time": result.get("execution_time", "unknown")
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "stderr": result.get("stderr", "")
                }
                
        except Exception as e:
            logger.error(f"Recipe execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _validate_compilation(self, repo_path: Path) -> Dict[str, Any]:
        """Validate that the project compiles after recipe application"""
        logger.info("Validating compilation")
        
        try:
            # Run Maven compile
            compile_result = self.command_executor.run_maven_compile(str(repo_path))
            
            return {
                "success": compile_result["exit_code"] == 0,
                "exit_code": compile_result["exit_code"],
                "errors": self._parse_compilation_errors(compile_result.get("stderr", "")),
                "stdout": compile_result.get("stdout", ""),
                "stderr": compile_result.get("stderr", "")
            }
            
        except Exception as e:
            logger.error(f"Compilation validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "errors": [str(e)]
            }
    
    def _should_continue_after_error(self, recipe: str, errors: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to decide whether to continue after compilation errors"""
        
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """A recipe execution resulted in compilation errors. Should we continue?

Recipe that caused errors: {recipe}
Compilation errors: {errors}
Execution context: {context}

Analyze:
1. Are these errors recoverable?
2. Will continuing cause more problems?
3. Should we rollback this recipe?
4. What's the best course of action?

Provide a clear continue/stop decision with reasoning.""")
        ])
        
        try:
            response = self.llm.invoke(
                decision_prompt.format_messages(
                    recipe=recipe,
                    errors=errors[:5],  # Limit to first 5 errors
                    context=str(context)
                )
            )
            
            # Parse decision from response
            content = response.content.lower()
            should_continue = "continue" in content and "stop" not in content
            
            return {
                "continue": should_continue,
                "reasoning": response.content,
                "reason": "LLM decision based on error analysis"
            }
            
        except Exception as e:
            logger.warning(f"LLM decision failed, using conservative approach: {e}")
            # Conservative default: stop if more than 3 errors
            return {
                "continue": len(errors) <= 3,
                "reasoning": "Conservative fallback decision",
                "reason": f"Too many errors ({len(errors)}) to continue safely"
            }
    
    def _should_continue_after_failure(self, recipe: str, error: str) -> Dict[str, Any]:
        """Decide whether to continue after a recipe fails completely"""
        
        # Some recipes are critical, others are optional
        critical_recipes = [
            "Java8toJava11",
            "Java11toJava17", 
            "Java17toJava21"
        ]
        
        is_critical = any(critical in recipe for critical in critical_recipes)
        
        if is_critical:
            return {
                "continue": False,
                "reason": f"Critical recipe {recipe} failed: {error}"
            }
        else:
            return {
                "continue": True,
                "reason": f"Optional recipe {recipe} failed, but continuing with others"
            }
    
    def _create_backup(self, repo_path: Path) -> Path:
        """Create a backup of the repository before making changes"""
        logger.info("Creating repository backup")
        
        backup_dir = repo_path.parent / f"{repo_path.name}_backup_{self._get_timestamp()}"
        
        try:
            # Use rsync or shutil.copytree for backup
            import shutil
            shutil.copytree(repo_path, backup_dir, ignore=shutil.ignore_patterns(
                '.git', 'target', 'build', '*.class', '*.jar'
            ))
            
            logger.info(f"Backup created at: {backup_dir}")
            return backup_dir
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            raise
    
    def _get_repository_context(self, repository_path: str) -> Dict[str, Any]:
        """Get context about the repository for planning decisions"""
        repo_path = Path(repository_path)
        
        context = {
            "has_spring_boot": self._contains_pattern(repo_path, "@SpringBootApplication"),
            "has_javax_imports": self._contains_pattern(repo_path, "import javax."),
            "has_junit4": self._contains_pattern(repo_path, "import org.junit.Test"),
            "java_file_count": len(list(repo_path.rglob("*.java"))),
        }
        
        # Read pom.xml for additional context
        pom_path = repo_path / "pom.xml"
        if pom_path.exists():
            try:
                pom_content = pom_path.read_text()
                context["spring_boot_version"] = self._extract_version(pom_content, "spring-boot")
                context["java_version"] = self._extract_java_version(pom_content)
            except Exception as e:
                logger.warning(f"Could not read pom.xml: {e}")
        
        return context
    
    def _contains_pattern(self, repo_path: Path, pattern: str) -> bool:
        """Check if any Java file contains the specified pattern"""
        java_files = list(repo_path.rglob("*.java"))[:10]  # Sample first 10 files
        
        for java_file in java_files:
            try:
                content = java_file.read_text()
                if pattern in content:
                    return True
            except Exception:
                continue
                
        return False
    
    def _extract_version(self, content: str, artifact: str) -> Optional[str]:
        """Extract version from POM content"""
        import re
        pattern = rf"<{artifact}\.version>([^<]+)</{artifact}\.version>"
        match = re.search(pattern, content)
        return match.group(1) if match else None
    
    def _extract_java_version(self, content: str) -> Optional[str]:
        """Extract Java version from POM content"""
        import re
        patterns = [
            r"<java\.version>([^<]+)</java\.version>",
            r"<maven\.compiler\.source>([^<]+)</maven\.compiler\.source>",
            r"<maven\.compiler\.target>([^<]+)</maven\.compiler\.target>"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
                
        return None
    
    def _parse_recipe_order(self, llm_response: str, original_recipes: List[str]) -> List[str]:
        """Parse LLM response to extract recipe execution order"""
        
        # Simple parsing - look for recipes mentioned in order
        ordered_recipes = []
        
        for recipe in original_recipes:
            if recipe in llm_response:
                ordered_recipes.append(recipe)
        
        # Add any missing recipes at the end
        for recipe in original_recipes:
            if recipe not in ordered_recipes:
                ordered_recipes.append(recipe)
        
        return ordered_recipes
    
    def _get_default_recipe_order(self, recipes: List[str]) -> List[str]:
        """Provide sensible default ordering for recipes"""
        
        # Define priority order based on recipe types
        priority_order = [
            "Java8toJava11",
            "Java11toJava17", 
            "Java17toJava21",
            "SpringBoot", 
            "JavaxMigrationToJakarta",
            "JUnit4to5"
        ]
        
        ordered = []
        
        # Add recipes in priority order
        for priority in priority_order:
            for recipe in recipes:
                if priority in recipe and recipe not in ordered:
                    ordered.append(recipe)
        
        # Add remaining recipes
        for recipe in recipes:
            if recipe not in ordered:
                ordered.append(recipe)
        
        return ordered
    
    def _parse_compilation_errors(self, stderr: str) -> List[str]:
        """Parse Maven compilation errors from stderr"""
        if not stderr:
            return []
        
        errors = []
        lines = stderr.split('\n')
        
        for line in lines:
            line = line.strip()
            if '[ERROR]' in line and 'compilation failure' not in line.lower():
                # Clean up the error message
                error = line.replace('[ERROR]', '').strip()
                if error and len(error) > 10:  # Filter out very short messages
                    errors.append(error)
        
        return errors[:10]  # Limit to first 10 errors
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for logging"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")