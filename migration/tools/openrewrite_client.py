"""
OpenRewrite Client for Java Migration System

Provides integration with OpenRewrite for automated code migrations.
Handles recipe execution, configuration management, and result parsing.
"""

import os
import yaml
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime

from .command_executor import CommandExecutor

logger = logging.getLogger(__name__)


class OpenRewriteClient:
    """
    Client for executing OpenRewrite recipes and managing configurations.
    
    Provides functionality to:
    - Execute individual and batch recipes
    - Manage rewrite.yml configurations
    - Parse execution results
    - Handle recipe dependencies and ordering
    """
    
    def __init__(self, maven_plugin_version: str = "5.3.0"):
        self.maven_plugin_version = maven_plugin_version
        self.command_executor = CommandExecutor()
        self.execution_log: List[Dict[str, Any]] = []
        
        # Common recipe mappings
        self.recipe_mappings = {
            "java8to11": "org.openrewrite.java.migrate.Java8toJava11",
            "java11to17": "org.openrewrite.java.migrate.Java11toJava17", 
            "java17to21": "org.openrewrite.java.migrate.Java17toJava21",
            "springboot3": "org.openrewrite.java.spring.boot3.UpgradeSpringBoot_3_2",
            "jakarta": "org.openrewrite.java.migrate.jakarta.JavaxMigrationToJakarta",
            "junit5": "org.openrewrite.java.testing.junit5.JUnit4to5Migration"
        }
    
    def run_recipe(self, project_path: Union[str, Path], recipe: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute a single OpenRewrite recipe.
        
        Args:
            project_path: Path to the Maven project
            recipe: Recipe name or identifier
            dry_run: Whether to run in dry-run mode
            
        Returns:
            Dictionary with execution results
        """
        project_path = Path(project_path)
        recipe_name = self._resolve_recipe_name(recipe)
        
        logger.info(f"Executing OpenRewrite recipe: {recipe_name}")
        
        start_time = datetime.now()
        result = {
            "recipe": recipe_name,
            "project_path": str(project_path),
            "dry_run": dry_run,
            "start_time": start_time.isoformat(),
            "success": False
        }
        
        try:
            # Ensure Maven project exists
            pom_path = project_path / "pom.xml"
            if not pom_path.exists():
                result["error"] = f"No pom.xml found in {project_path}"
                return result
            
            # Prepare Maven command
            command = [
                "mvn", 
                f"org.openrewrite.maven:rewrite-maven-plugin:{self.maven_plugin_version}:run",
                f"-Drewrite.activeRecipes={recipe_name}"
            ]
            
            if dry_run:
                command.append("-Drewrite.dryRun=true")
            
            # Execute the command
            execution_result = self.command_executor.execute_command(
                command, 
                working_dir=project_path,
                timeout=1800  # 30 minutes timeout
            )
            
            # Process results
            result.update({
                "success": execution_result["success"],
                "exit_code": execution_result.get("exit_code", -1),
                "execution_time": execution_result.get("execution_time", 0),
                "stdout": execution_result.get("stdout", ""),
                "stderr": execution_result.get("stderr", ""),
                "end_time": datetime.now().isoformat()
            })
            
            if execution_result["success"]:
                # Parse OpenRewrite output for details
                rewrite_details = self._parse_openrewrite_output(execution_result.get("stdout", ""))
                result.update(rewrite_details)
                
                logger.info(f"Recipe {recipe_name} completed successfully")
            else:
                result["error"] = f"Recipe execution failed with exit code {execution_result.get('exit_code')}"
                logger.error(f"Recipe {recipe_name} failed: {result['error']}")
            
        except Exception as e:
            logger.error(f"Recipe execution failed with exception: {e}")
            result["error"] = str(e)
            result["end_time"] = datetime.now().isoformat()
        
        # Log the execution
        self.execution_log.append(result)
        
        return result
    
    def run_recipe_batch(self, project_path: Union[str, Path], recipes: List[str], continue_on_failure: bool = True) -> Dict[str, Any]:
        """
        Execute multiple OpenRewrite recipes in sequence.
        
        Args:
            project_path: Path to the Maven project
            recipes: List of recipe names or identifiers
            continue_on_failure: Whether to continue if a recipe fails
            
        Returns:
            Dictionary with batch execution results
        """
        project_path = Path(project_path)
        
        logger.info(f"Executing batch of {len(recipes)} OpenRewrite recipes")
        
        batch_result = {
            "project_path": str(project_path),
            "total_recipes": len(recipes),
            "successful_recipes": [],
            "failed_recipes": [],
            "recipe_results": [],
            "start_time": datetime.now().isoformat(),
            "success": True
        }
        
        for i, recipe in enumerate(recipes):
            logger.info(f"Executing recipe {i+1}/{len(recipes)}: {recipe}")
            
            recipe_result = self.run_recipe(project_path, recipe)
            batch_result["recipe_results"].append(recipe_result)
            
            if recipe_result["success"]:
                batch_result["successful_recipes"].append(recipe)
                logger.info(f"Recipe {recipe} completed successfully")
            else:
                batch_result["failed_recipes"].append({
                    "recipe": recipe,
                    "error": recipe_result.get("error", "Unknown error")
                })
                logger.error(f"Recipe {recipe} failed: {recipe_result.get('error')}")
                
                if not continue_on_failure:
                    logger.info("Stopping batch execution due to failure")
                    batch_result["success"] = False
                    break
        
        # Calculate summary statistics
        batch_result["end_time"] = datetime.now().isoformat()
        batch_result["success"] = len(batch_result["failed_recipes"]) == 0
        
        logger.info(f"Batch execution completed. Success: {len(batch_result['successful_recipes'])}, Failed: {len(batch_result['failed_recipes'])}")
        
        return batch_result
    
    def create_rewrite_config(self, project_path: Union[str, Path], recipes: List[str], config_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a rewrite.yml configuration file.
        
        Args:
            project_path: Path to the Maven project
            recipes: List of recipes to include
            config_options: Additional configuration options
            
        Returns:
            Dictionary with configuration creation result
        """
        project_path = Path(project_path)
        config_path = project_path / "rewrite.yml"
        
        logger.info(f"Creating rewrite.yml configuration with {len(recipes)} recipes")
        
        try:
            # Resolve recipe names
            resolved_recipes = [self._resolve_recipe_name(recipe) for recipe in recipes]
            
            # Build configuration
            config = {
                "type": "specs.openrewrite.org/v1beta/recipe",
                "name": "com.migration.JavaMigrationRecipes",
                "displayName": "Java Migration Recipes",
                "description": "Automated Java migration recipes generated by migration system",
                "recipeList": resolved_recipes
            }
            
            # Add custom options if provided
            if config_options:
                config.update(config_options)
            
            # Write configuration file
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Created rewrite.yml configuration at {config_path}")
            
            return {
                "success": True,
                "config_path": str(config_path),
                "recipes_count": len(resolved_recipes),
                "recipes": resolved_recipes
            }
            
        except Exception as e:
            logger.error(f"Failed to create rewrite.yml: {e}")
            return {
                "success": False,
                "error": str(e),
                "config_path": str(config_path)
            }
    
    def discover_applicable_recipes(self, project_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Discover which recipes are applicable to a project.
        
        Args:
            project_path: Path to the Maven project
            
        Returns:
            Dictionary with discovered recipes
        """
        project_path = Path(project_path)
        
        logger.info("Discovering applicable OpenRewrite recipes")
        
        try:
            # Run OpenRewrite discovery command
            command = [
                "mvn",
                f"org.openrewrite.maven:rewrite-maven-plugin:{self.maven_plugin_version}:discover"
            ]
            
            execution_result = self.command_executor.execute_command(
                command,
                working_dir=project_path,
                timeout=300
            )
            
            if execution_result["success"]:
                # Parse discovery output
                discovered_recipes = self._parse_discovery_output(execution_result.get("stdout", ""))
                
                return {
                    "success": True,
                    "discovered_recipes": discovered_recipes,
                    "recipe_count": len(discovered_recipes)
                }
            else:
                return {
                    "success": False,
                    "error": f"Discovery failed with exit code {execution_result.get('exit_code')}",
                    "stderr": execution_result.get("stderr", "")
                }
                
        except Exception as e:
            logger.error(f"Recipe discovery failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_project_compatibility(self, project_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate that a project is compatible with OpenRewrite.
        
        Args:
            project_path: Path to the Maven project
            
        Returns:
            Dictionary with compatibility information
        """
        project_path = Path(project_path)
        
        logger.info("Validating OpenRewrite compatibility")
        
        compatibility = {
            "compatible": True,
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Check for Maven project
            pom_path = project_path / "pom.xml"
            if not pom_path.exists():
                compatibility["compatible"] = False
                compatibility["issues"].append("No pom.xml found - not a Maven project")
                return compatibility
            
            # Check Maven version compatibility
            try:
                mvn_version_result = self.command_executor.execute_command(
                    ["mvn", "--version"], 
                    working_dir=project_path
                )
                
                if mvn_version_result["success"]:
                    maven_version = self._extract_maven_version(mvn_version_result.get("stdout", ""))
                    if maven_version:
                        compatibility["maven_version"] = maven_version
                        
                        # Check if Maven version is compatible
                        if self._is_maven_version_compatible(maven_version):
                            compatibility["recommendations"].append(f"Maven {maven_version} is compatible")
                        else:
                            compatibility["issues"].append(f"Maven {maven_version} may have compatibility issues")
                            compatibility["recommendations"].append("Consider upgrading to Maven 3.6+")
                
            except Exception as e:
                compatibility["issues"].append(f"Could not determine Maven version: {e}")
            
            # Check Java version
            try:
                java_version_result = self.command_executor.execute_command(
                    ["java", "-version"],
                    working_dir=project_path
                )
                
                if java_version_result["success"]:
                    java_version = self._extract_java_version(java_version_result.get("stderr", ""))
                    if java_version:
                        compatibility["java_version"] = java_version
                        
                        if int(java_version.split('.')[0]) >= 8:
                            compatibility["recommendations"].append(f"Java {java_version} is compatible")
                        else:
                            compatibility["compatible"] = False
                            compatibility["issues"].append(f"Java {java_version} is not supported - requires Java 8+")
                
            except Exception as e:
                compatibility["issues"].append(f"Could not determine Java version: {e}")
            
            # Check for potential conflicts in pom.xml
            try:
                pom_content = pom_path.read_text()
                
                # Check for existing OpenRewrite plugin
                if "rewrite-maven-plugin" in pom_content:
                    compatibility["recommendations"].append("OpenRewrite plugin already configured in pom.xml")
                else:
                    compatibility["recommendations"].append("Will need to configure OpenRewrite plugin in pom.xml")
                
                # Check for potential problematic plugins
                problematic_patterns = [
                    "maven-compiler-plugin.*<version>2",  # Very old compiler plugin
                    "maven-surefire-plugin.*<version>2"   # Very old surefire plugin
                ]
                
                for pattern in problematic_patterns:
                    import re
                    if re.search(pattern, pom_content, re.DOTALL):
                        compatibility["issues"].append(f"Found potentially problematic plugin configuration: {pattern}")
                
            except Exception as e:
                compatibility["issues"].append(f"Could not analyze pom.xml: {e}")
            
            # Final compatibility assessment
            if compatibility["issues"]:
                if any("not supported" in issue for issue in compatibility["issues"]):
                    compatibility["compatible"] = False
                else:
                    compatibility["recommendations"].append("Issues found but project may still be compatible")
            
            return compatibility
            
        except Exception as e:
            logger.error(f"Compatibility validation failed: {e}")
            return {
                "compatible": False,
                "error": str(e),
                "issues": [str(e)],
                "recommendations": []
            }
    
    def get_execution_log(self) -> List[Dict[str, Any]]:
        """Get the log of all OpenRewrite executions"""
        return self.execution_log.copy()
    
    def clear_execution_log(self):
        """Clear the execution log"""
        self.execution_log.clear()
        logger.info("OpenRewrite execution log cleared")
    
    # Private helper methods
    
    def _resolve_recipe_name(self, recipe: str) -> str:
        """Resolve recipe shorthand to full name"""
        if recipe in self.recipe_mappings:
            return self.recipe_mappings[recipe]
        return recipe
    
    def _parse_openrewrite_output(self, output: str) -> Dict[str, Any]:
        """Parse OpenRewrite execution output for details"""
        details = {
            "files_modified": 0,
            "changes_made": 0,
            "recipes_applied": [],
            "warnings": [],
            "errors": []
        }
        
        if not output:
            return details
        
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for files modified
            if 'files changed' in line.lower() or 'file changed' in line.lower():
                import re
                match = re.search(r'(\d+)\s+files?\s+changed', line.lower())
                if match:
                    details["files_modified"] = int(match.group(1))
            
            # Look for recipe applications
            if 'applied recipe' in line.lower() or 'running recipe' in line.lower():
                details["recipes_applied"].append(line)
            
            # Look for warnings
            if '[WARNING]' in line:
                details["warnings"].append(line.replace('[WARNING]', '').strip())
            
            # Look for errors
            if '[ERROR]' in line and 'build failure' not in line.lower():
                details["errors"].append(line.replace('[ERROR]', '').strip())
        
        return details
    
    def _parse_discovery_output(self, output: str) -> List[str]:
        """Parse recipe discovery output"""
        recipes = []
        
        if not output:
            return recipes
        
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for recipe names (this is a simplified parser)
            if line.startswith('org.openrewrite.') or 'recipe' in line.lower():
                recipes.append(line)
        
        return recipes
    
    def _extract_maven_version(self, version_output: str) -> Optional[str]:
        """Extract Maven version from version output"""
        import re
        
        match = re.search(r'Apache Maven (\d+\.\d+\.\d+)', version_output)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_java_version(self, version_output: str) -> Optional[str]:
        """Extract Java version from version output"""
        import re
        
        # Try different Java version patterns
        patterns = [
            r'version "(\d+\.\d+\.\d+)',  # Java 8 style
            r'version "(\d+)',           # Java 9+ style
            r'openjdk version "(\d+\.\d+\.\d+)',
            r'openjdk version "(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, version_output)
            if match:
                return match.group(1)
        
        return None
    
    def _is_maven_version_compatible(self, version: str) -> bool:
        """Check if Maven version is compatible with OpenRewrite"""
        try:
            major, minor, patch = map(int, version.split('.'))
            
            # OpenRewrite requires Maven 3.6+
            if major > 3:
                return True
            elif major == 3 and minor >= 6:
                return True
            else:
                return False
                
        except ValueError:
            return False  # Couldn't parse version, assume incompatible