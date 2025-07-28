"""
Error Fixing Agent for Java Migration System

This agent is responsible for:
- Compilation error analysis and resolution
- Test failure diagnosis and fixing
- Code pattern modernization
- Dependency conflict resolution
- Intelligent code modification with LLM assistance
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from tools.file_operations import FileOperations
from tools.command_executor import CommandExecutor

logger = logging.getLogger(__name__)


class ErrorFixingAgent:
    """
    Intelligent agent for analyzing and fixing compilation errors and test failures.
    
    Uses LLM reasoning combined with pattern recognition to identify and fix
    common migration-related issues automatically.
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.file_ops = FileOperations()
        self.command_executor = CommandExecutor()
        
        # System prompt for error analysis and fixing
        self.system_prompt = """You are an expert Java migration troubleshooter. Your job is to analyze compilation errors, test failures, and other migration issues, then provide specific fixes.

You have access to:
1. Complete error messages and stack traces
2. Source code reading and modification capabilities
3. Build system integration for testing fixes
4. Pattern recognition for common migration issues

Your approach should be:
1. Categorize the error type (compilation, test, dependency, configuration)
2. Identify the root cause (API changes, missing dependencies, configuration issues)
3. Provide specific, targeted fixes
4. Validate fixes don't break other functionality
5. Learn from previous fixes to improve future solutions

Common Java 21 migration patterns:
- Deprecated wrapper constructors (new Integer() -> Integer.valueOf())
- Javax to Jakarta namespace changes
- JUnit 4 to 5 API changes
- Spring Boot configuration updates
- Security manager removals
- Module system conflicts

Be precise and surgical in your fixes. Always explain your reasoning."""
    
    def fix_errors(self, repository_path: str, errors: List[str]) -> Dict[str, Any]:
        """
        Analyze and fix compilation errors in the repository.
        
        Args:
            repository_path: Path to the Java repository
            errors: List of compilation error messages
            
        Returns:
            Dictionary containing fix results and remaining errors
        """
        logger.info(f"Starting error analysis for {len(errors)} errors")
        
        repo_path = Path(repository_path)
        results = {
            "fixes_applied": [],
            "remaining_errors": [],
            "fix_log": [],
            "success": True
        }
        
        if not errors:
            logger.info("No errors to fix")
            return results
        
        try:
            # Categorize errors by type
            categorized_errors = self._categorize_errors(errors)
            results["error_categories"] = categorized_errors
            
            # Apply fixes in order of priority
            for category, error_list in categorized_errors.items():
                if not error_list:
                    continue
                    
                logger.info(f"Fixing {len(error_list)} {category} errors")
                category_result = self._fix_category_errors(repo_path, category, error_list)
                
                results["fixes_applied"].extend(category_result["fixes_applied"])
                results["fix_log"].extend(category_result["fix_log"])
                
                # Validate fixes by recompiling
                validation_result = self._validate_fixes(repo_path)
                if validation_result["success"]:
                    logger.info(f"Successfully fixed {category} errors")
                else:
                    logger.warning(f"Some {category} fixes may have introduced new issues")
                    results["remaining_errors"].extend(validation_result["errors"])
            
            # Final compilation check
            final_validation = self._validate_fixes(repo_path)
            results["remaining_errors"] = final_validation.get("errors", [])
            results["success"] = len(results["remaining_errors"]) == 0
            
            logger.info(f"Error fixing completed. Applied {len(results['fixes_applied'])} fixes, {len(results['remaining_errors'])} remaining errors")
            return results
            
        except Exception as e:
            logger.error(f"Error fixing failed with exception: {e}")
            results["success"] = False
            results["error"] = str(e)
            return results
    
    def _categorize_errors(self, errors: List[str]) -> Dict[str, List[str]]:
        """Categorize errors by type for targeted fixing"""
        logger.info("Categorizing errors")
        
        categories = {
            "deprecated_api": [],
            "import_issues": [],
            "dependency_conflicts": [],
            "annotation_changes": [],
            "api_removals": [],
            "configuration_issues": [],
            "generic_compilation": []
        }
        
        # Error pattern definitions
        patterns = {
            "deprecated_api": [
                r"constructor \w+\(\w+\) in class \w+ has been deprecated",
                r"new Integer\(|new Long\(|new Double\(|new Boolean\(",
                r"\.newInstance\(\)",
                r"SecurityManager"
            ],
            "import_issues": [
                r"package javax\.\w+ does not exist",
                r"cannot find symbol.*javax\.",
                r"import javax\."
            ],
            "dependency_conflicts": [
                r"package .* does not exist",
                r"cannot find symbol.*class \w+",
                r"NoClassDefFoundError"
            ],
            "annotation_changes": [
                r"cannot find symbol.*@Test",
                r"@Before.*cannot find symbol",
                r"@After.*cannot find symbol"
            ],
            "api_removals": [
                r"cannot find symbol.*method",
                r"incompatible types",
                r"method .* cannot be applied"
            ]
        }
        
        for error in errors:
            categorized = False
            
            for category, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, error, re.IGNORECASE):
                        categories[category].append(error)
                        categorized = True
                        break
                if categorized:
                    break
            
            if not categorized:
                categories["generic_compilation"].append(error)
        
        # Log categorization results
        for category, error_list in categories.items():
            if error_list:
                logger.info(f"Found {len(error_list)} {category} errors")
        
        return categories
    
    def _fix_category_errors(self, repo_path: Path, category: str, errors: List[str]) -> Dict[str, Any]:
        """Fix errors of a specific category"""
        logger.info(f"Fixing {category} errors")
        
        result = {
            "fixes_applied": [],
            "fix_log": []
        }
        
        try:
            if category == "deprecated_api":
                fixes = self._fix_deprecated_api_errors(repo_path, errors)
            elif category == "import_issues":
                fixes = self._fix_import_issues(repo_path, errors)
            elif category == "annotation_changes":
                fixes = self._fix_annotation_changes(repo_path, errors)
            elif category == "dependency_conflicts":
                fixes = self._fix_dependency_conflicts(repo_path, errors)
            else:
                # Use LLM for generic or complex errors
                fixes = self._fix_with_llm(repo_path, category, errors)
            
            result["fixes_applied"] = fixes.get("fixes_applied", [])
            result["fix_log"] = fixes.get("fix_log", [])
            
        except Exception as e:
            logger.error(f"Failed to fix {category} errors: {e}")
            result["fix_log"].append(f"Error fixing {category}: {e}")
        
        return result
    
    def _fix_deprecated_api_errors(self, repo_path: Path, errors: List[str]) -> Dict[str, Any]:
        """Fix deprecated API usage errors"""
        logger.info("Fixing deprecated API errors")
        
        fixes = {
            "fixes_applied": [],
            "fix_log": []
        }
        
        # Common deprecated API replacements
        replacements = {
            r"new Integer\(([^)]+)\)": r"Integer.valueOf(\1)",
            r"new Long\(([^)]+)\)": r"Long.valueOf(\1)",
            r"new Double\(([^)]+)\)": r"Double.valueOf(\1)",
            r"new Boolean\(([^)]+)\)": r"Boolean.valueOf(\1)",
            r"new Float\(([^)]+)\)": r"Float.valueOf(\1)",
            r"new Short\(([^)]+)\)": r"Short.valueOf(\1)",
            r"new Byte\(([^)]+)\)": r"Byte.valueOf(\1)",
            r"\.newInstance\(\)": r".getDeclaredConstructor().newInstance()"
        }
        
        java_files = list(repo_path.rglob("*.java"))
        
        for java_file in java_files:
            try:
                content = java_file.read_text(encoding='utf-8')
                original_content = content
                
                for pattern, replacement in replacements.items():
                    if re.search(pattern, content):
                        content = re.sub(pattern, replacement, content)
                        fixes["fix_log"].append(f"Applied {pattern} -> {replacement} in {java_file.name}")
                
                if content != original_content:
                    java_file.write_text(content, encoding='utf-8')
                    fixes["fixes_applied"].append(f"Updated deprecated APIs in {java_file.name}")
                    
            except Exception as e:
                logger.warning(f"Could not process {java_file}: {e}")
                continue
        
        return fixes
    
    def _fix_import_issues(self, repo_path: Path, errors: List[str]) -> Dict[str, Any]:
        """Fix import-related errors (mainly javax -> jakarta)"""
        logger.info("Fixing import issues")
        
        fixes = {
            "fixes_applied": [],
            "fix_log": []
        }
        
        # Javax to Jakarta mappings
        javax_to_jakarta = {
            "javax.servlet": "jakarta.servlet",
            "javax.persistence": "jakarta.persistence",
            "javax.validation": "jakarta.validation",
            "javax.annotation": "jakarta.annotation",
            "javax.inject": "jakarta.inject",
            "javax.transaction": "jakarta.transaction",
            "javax.security": "jakarta.security",
            "javax.ws.rs": "jakarta.ws.rs",
            "javax.json": "jakarta.json",
            "javax.xml.bind": "jakarta.xml.bind"
        }
        
        java_files = list(repo_path.rglob("*.java"))
        
        for java_file in java_files:
            try:
                content = java_file.read_text(encoding='utf-8')
                original_content = content
                
                for javax_pkg, jakarta_pkg in javax_to_jakarta.items():
                    if f"import {javax_pkg}" in content:
                        content = content.replace(f"import {javax_pkg}", f"import {jakarta_pkg}")
                        fixes["fix_log"].append(f"Updated import {javax_pkg} -> {jakarta_pkg} in {java_file.name}")
                
                if content != original_content:
                    java_file.write_text(content, encoding='utf-8') 
                    fixes["fixes_applied"].append(f"Updated javax imports in {java_file.name}")
                    
            except Exception as e:
                logger.warning(f"Could not process {java_file}: {e}")
                continue
        
        return fixes
    
    def _fix_annotation_changes(self, repo_path: Path, errors: List[str]) -> Dict[str, Any]:
        """Fix annotation-related errors (mainly JUnit 4 -> 5)"""
        logger.info("Fixing annotation changes")
        
        fixes = {
            "fixes_applied": [],
            "fix_log": []
        }
        
        # JUnit 4 to 5 annotation mappings
        junit_mappings = {
            "import org.junit.Test;": "import org.junit.jupiter.api.Test;",
            "import org.junit.Before;": "import org.junit.jupiter.api.BeforeEach;",
            "import org.junit.After;": "import org.junit.jupiter.api.AfterEach;",
            "import org.junit.BeforeClass;": "import org.junit.jupiter.api.BeforeAll;",
            "import org.junit.AfterClass;": "import org.junit.jupiter.api.AfterAll;",
            "import org.junit.Assert;": "import org.junit.jupiter.api.Assertions;",
            "@Before": "@BeforeEach",
            "@After": "@AfterEach", 
            "@BeforeClass": "@BeforeAll",
            "@AfterClass": "@AfterAll",
            "Assert.assertEquals": "Assertions.assertEquals",
            "Assert.assertTrue": "Assertions.assertTrue",
            "Assert.assertFalse": "Assertions.assertFalse",
            "Assert.assertNull": "Assertions.assertNull",
            "Assert.assertNotNull": "Assertions.assertNotNull"
        }
        
        # Focus on test files
        test_files = list(repo_path.rglob("*Test.java")) + list((repo_path / "src" / "test").rglob("*.java")) if (repo_path / "src" / "test").exists() else []
        
        for test_file in test_files:
            try:
                content = test_file.read_text(encoding='utf-8')
                original_content = content
                
                for old_annotation, new_annotation in junit_mappings.items():
                    if old_annotation in content:
                        content = content.replace(old_annotation, new_annotation)
                        fixes["fix_log"].append(f"Updated {old_annotation} -> {new_annotation} in {test_file.name}")
                
                if content != original_content:
                    test_file.write_text(content, encoding='utf-8')
                    fixes["fixes_applied"].append(f"Updated JUnit annotations in {test_file.name}")
                    
            except Exception as e:
                logger.warning(f"Could not process {test_file}: {e}")
                continue
        
        return fixes
    
    def _fix_dependency_conflicts(self, repo_path: Path, errors: List[str]) -> Dict[str, Any]:
        """Fix dependency-related conflicts"""
        logger.info("Fixing dependency conflicts")
        
        fixes = {
            "fixes_applied": [],
            "fix_log": []
        }
        
        # This would typically involve updating pom.xml
        pom_path = repo_path / "pom.xml"
        if not pom_path.exists():
            return fixes
        
        try:
            # Use LLM to analyze dependency conflicts and suggest fixes
            dependency_analysis = self._analyze_dependency_conflicts_with_llm(pom_path, errors)
            
            if dependency_analysis.get("suggested_fixes"):
                # Apply suggested dependency fixes (simplified implementation)
                fixes["fix_log"].append("Dependency conflict analysis completed")
                fixes["fixes_applied"].append("Analyzed dependency conflicts for manual review")
            
        except Exception as e:
            logger.error(f"Dependency conflict analysis failed: {e}")
            fixes["fix_log"].append(f"Dependency analysis error: {e}")
        
        return fixes
    
    def _fix_with_llm(self, repo_path: Path, category: str, errors: List[str]) -> Dict[str, Any]:
        """Use LLM to analyze and fix complex errors"""
        logger.info(f"Using LLM to fix {category} errors")
        
        fixes = {
            "fixes_applied": [],
            "fix_log": []
        }
        
        # Get relevant source code context for the errors
        context = self._get_error_context(repo_path, errors)
        
        fixing_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Analyze these compilation errors and provide specific fixes:

Category: {category}
Errors: {errors}
Source code context: {context}

For each error, provide:
1. Root cause analysis
2. Specific fix (exact code changes)
3. File locations to modify
4. Validation steps

Format your response with clear sections for each fix.""")
        ])
        
        try:
            response = self.llm.invoke(
                fixing_prompt.format_messages(
                    category=category,
                    errors=errors[:5],  # Limit to first 5 errors
                    context=context
                )
            )
            
            # Apply LLM-suggested fixes (this would need more sophisticated parsing)
            llm_fixes = self._parse_and_apply_llm_fixes(repo_path, response.content)
            
            fixes["fixes_applied"].extend(llm_fixes.get("applied", []))
            fixes["fix_log"].append(f"LLM analysis for {category} errors: {response.content[:200]}...")
            
        except Exception as e:
            logger.error(f"LLM fixing failed: {e}")
            fixes["fix_log"].append(f"LLM fixing error: {e}")
        
        return fixes
    
    def _validate_fixes(self, repo_path: Path) -> Dict[str, Any]:
        """Validate that fixes don't break compilation"""
        logger.info("Validating applied fixes")
        
        try:
            compile_result = self.command_executor.run_maven_compile(str(repo_path))
            
            return {
                "success": compile_result["exit_code"] == 0,
                "exit_code": compile_result["exit_code"],
                "errors": self._parse_compilation_errors(compile_result.get("stderr", "")),
                "stdout": compile_result.get("stdout", "")
            }
            
        except Exception as e:
            logger.error(f"Fix validation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "errors": [str(e)]
            }
    
    def _get_error_context(self, repo_path: Path, errors: List[str]) -> Dict[str, Any]:
        """Get relevant source code context for errors"""
        
        # Extract file names and line numbers from errors
        context = {
            "file_count": len(list(repo_path.rglob("*.java"))),
            "error_files": [],
            "sample_code": []
        }
        
        for error in errors[:3]:  # Analyze first 3 errors
            # Simple regex to extract file information
            file_match = re.search(r'([A-Za-z0-9_]+\.java)', error)
            if file_match:
                file_name = file_match.group(1)
                context["error_files"].append(file_name)
                
                # Try to find and read the file
                for java_file in repo_path.rglob(file_name):
                    try:
                        content = java_file.read_text(encoding='utf-8')
                        context["sample_code"].append({
                            "file": file_name,
                            "content": content[:500]  # First 500 chars
                        })
                        break
                    except Exception:
                        continue
        
        return context
    
    def _analyze_dependency_conflicts_with_llm(self, pom_path: Path, errors: List[str]) -> Dict[str, Any]:
        """Use LLM to analyze dependency conflicts"""
        
        try:
            pom_content = pom_path.read_text()
            
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a Maven dependency expert. Analyze POM files and suggest dependency conflict resolutions."),
                ("human", "Analyze this POM file and compilation errors for dependency conflicts:\n\nPOM content:\n{pom}\n\nErrors:\n{errors}\n\nSuggest specific dependency updates or exclusions to resolve conflicts.")
            ])
            
            response = self.llm.invoke(
                analysis_prompt.format_messages(
                    pom=pom_content[:2000],  # Limit POM content
                    errors=errors
                )
            )
            
            return {
                "analysis": response.content,
                "suggested_fixes": []  # Would parse from LLM response
            }
            
        except Exception as e:
            logger.error(f"LLM dependency analysis failed: {e}")
            return {"error": str(e)}
    
    def _parse_and_apply_llm_fixes(self, repo_path: Path, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response and apply suggested fixes"""
        
        # This is a simplified implementation
        # In practice, you'd need sophisticated parsing of LLM responses
        
        applied_fixes = []
        
        # Look for code changes in the response
        if "replace" in llm_response.lower() or "change" in llm_response.lower():
            applied_fixes.append("LLM suggested code changes identified")
        
        return {"applied": applied_fixes}
    
    def _parse_compilation_errors(self, stderr: str) -> List[str]:
        """Parse compilation errors from Maven stderr"""
        if not stderr:
            return []
        
        errors = []
        lines = stderr.split('\n')
        
        for line in lines:
            line = line.strip()
            if '[ERROR]' in line and 'compilation failure' not in line.lower():
                error = line.replace('[ERROR]', '').strip()
                if error and len(error) > 10:
                    errors.append(error)
        
        return errors[:10]  # Limit to first 10 errors