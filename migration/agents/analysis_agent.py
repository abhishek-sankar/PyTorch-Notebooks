"""
Analysis Agent for Java Migration System

This agent is responsible for:
- Repository structure analysis
- Dependency discovery and version checking
- Java version detection
- Migration complexity assessment
- Maven Central API integration for dependency analysis
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from tools.maven_api import MavenCentralAPI
from tools.file_operations import FileOperations

logger = logging.getLogger(__name__)


class AnalysisAgent:
    """
    Intelligent agent for analyzing Java repositories and planning migrations.
    
    Uses LLM reasoning combined with structured analysis to understand
    repository characteristics and recommend migration strategies.
    """
    
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.maven_api = MavenCentralAPI()
        self.file_ops = FileOperations()
        
        # System prompt for analysis
        self.system_prompt = """You are an expert Java migration analyst. Your job is to analyze Java projects and provide detailed migration assessments.

You have access to:
1. Project structure and file contents
2. Maven POM analysis results
3. Maven Central API for dependency version checking
4. Source code pattern analysis

For each analysis, provide:
1. Current Java version detection
2. Framework and library inventory
3. Migration complexity assessment (simple/moderate/complex)
4. Specific migration recommendations
5. Risk assessment and potential issues

Be thorough but concise. Focus on actionable insights."""
    
    def analyze(self, repository_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive repository analysis.
        
        Args:
            repository_path: Path to the Java repository
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Starting analysis of repository: {repository_path}")
        
        repo_path = Path(repository_path)
        
        try:
            # Collect all analysis data
            structure_analysis = self._analyze_project_structure(repo_path)
            pom_analysis = self._analyze_pom_xml(repo_path)
            source_analysis = self._analyze_source_code(repo_path)
            dependency_analysis = self._analyze_dependencies(pom_analysis.get("dependencies", []))
            
            # Prepare context for LLM analysis
            analysis_context = {
                "project_path": str(repo_path),
                "structure": structure_analysis,
                "pom_analysis": pom_analysis,
                "source_analysis": source_analysis,
                "dependency_analysis": dependency_analysis
            }
            
            # Get LLM-powered analysis and recommendations
            llm_analysis = self._get_llm_analysis(analysis_context)
            
            # Combine structured and LLM analysis
            final_analysis = {
                **structure_analysis,
                **pom_analysis,
                **source_analysis,
                "dependency_analysis": dependency_analysis,
                "llm_insights": llm_analysis,
                "analysis_timestamp": self._get_timestamp()
            }
            
            logger.info("Repository analysis completed successfully")
            return final_analysis
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "error": str(e),
                "java_version": "unknown",
                "complexity": "unknown",
                "dependencies": []
            }
    
    def _analyze_project_structure(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze the project directory structure"""
        logger.info("Analyzing project structure")
        
        structure = {
            "is_maven_project": (repo_path / "pom.xml").exists(),
            "is_gradle_project": (repo_path / "build.gradle").exists() or (repo_path / "build.gradle.kts").exists(),
            "has_src_main_java": (repo_path / "src" / "main" / "java").exists(),
            "has_src_test_java": (repo_path / "src" / "test" / "java").exists(),
            "java_file_count": len(list(repo_path.rglob("*.java"))),
            "test_file_count": len(list((repo_path / "src" / "test").rglob("*.java"))) if (repo_path / "src" / "test").exists() else 0,
            "resource_files": len(list(repo_path.rglob("*.properties"))) + len(list(repo_path.rglob("*.xml"))) + len(list(repo_path.rglob("*.yml"))),
        }
        
        # Determine project type
        if structure["is_maven_project"]:
            structure["build_system"] = "maven"
        elif structure["is_gradle_project"]:
            structure["build_system"] = "gradle"
        else:
            structure["build_system"] = "unknown"
        
        return structure
    
    def _analyze_pom_xml(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze Maven POM file for dependencies and configuration"""
        logger.info("Analyzing pom.xml")
        
        pom_path = repo_path / "pom.xml"
        if not pom_path.exists():
            return {"error": "No pom.xml found"}
        
        try:
            tree = ET.parse(pom_path)
            root = tree.getroot()
            
            # Handle namespace
            namespace = ""
            if root.tag.startswith("{"):
                namespace = root.tag.split("}")[0] + "}"
            
            analysis = {
                "group_id": self._get_xml_text(root, f"{namespace}groupId"),
                "artifact_id": self._get_xml_text(root, f"{namespace}artifactId"),
                "version": self._get_xml_text(root, f"{namespace}version"),
                "packaging": self._get_xml_text(root, f"{namespace}packaging", "jar"),
            }
            
            # Extract properties
            properties = {}
            props_elem = root.find(f"{namespace}properties")
            if props_elem is not None:
                for prop in props_elem:
                    prop_name = prop.tag.replace(namespace, "")
                    properties[prop_name] = prop.text or ""
            
            analysis["properties"] = properties
            
            # Detect Java version
            java_version = self._detect_java_version(properties, root, namespace)
            analysis["java_version"] = java_version
            
            # Extract dependencies
            dependencies = []
            deps_elem = root.find(f"{namespace}dependencies")
            if deps_elem is not None:
                for dep in deps_elem.findall(f"{namespace}dependency"):
                    dep_info = {
                        "group_id": self._get_xml_text(dep, f"{namespace}groupId"),
                        "artifact_id": self._get_xml_text(dep, f"{namespace}artifactId"),
                        "version": self._get_xml_text(dep, f"{namespace}version"),
                        "scope": self._get_xml_text(dep, f"{namespace}scope", "compile")
                    }
                    if dep_info["group_id"] and dep_info["artifact_id"]:
                        dependencies.append(dep_info)
            
            analysis["dependencies"] = dependencies
            analysis["dependency_count"] = len(dependencies)
            
            # Extract plugins
            plugins = []
            build_elem = root.find(f"{namespace}build")
            if build_elem is not None:
                plugins_elem = build_elem.find(f"{namespace}plugins")
                if plugins_elem is not None:
                    for plugin in plugins_elem.findall(f"{namespace}plugin"):
                        plugin_info = {
                            "group_id": self._get_xml_text(plugin, f"{namespace}groupId"),
                            "artifact_id": self._get_xml_text(plugin, f"{namespace}artifactId"),
                            "version": self._get_xml_text(plugin, f"{namespace}version")
                        }
                        if plugin_info["artifact_id"]:
                            plugins.append(plugin_info)
            
            analysis["plugins"] = plugins
            analysis["plugin_count"] = len(plugins)
            
            return analysis
            
        except ET.ParseError as e:
            logger.error(f"Failed to parse pom.xml: {e}")
            return {"error": f"Invalid XML: {e}"}
    
    def _analyze_source_code(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze Java source code for patterns and frameworks"""
        logger.info("Analyzing source code patterns")
        
        java_files = list(repo_path.rglob("*.java"))
        if not java_files:
            return {"java_files": 0, "patterns": []}
        
        patterns = {
            "spring_boot": 0,
            "spring_framework": 0,
            "javax_imports": 0,
            "jakarta_imports": 0,
            "junit4_tests": 0,
            "junit5_tests": 0,
            "deprecated_apis": 0,
            "lambda_expressions": 0,
            "stream_api": 0
        }
        
        # Pattern matching regexes
        pattern_regexes = {
            "spring_boot": re.compile(r"@SpringBootApplication|@EnableAutoConfiguration"),
            "spring_framework": re.compile(r"@Component|@Service|@Repository|@Controller"),
            "javax_imports": re.compile(r"import javax\."),
            "jakarta_imports": re.compile(r"import jakarta\."),
            "junit4_tests": re.compile(r"import org\.junit\.Test|@Test.*org\.junit"),
            "junit5_tests": re.compile(r"import org\.junit\.jupiter|@Test.*jupiter"),
            "deprecated_apis": re.compile(r"@Deprecated|\.newInstance\(\)|new Integer\(|new Long\("),
            "lambda_expressions": re.compile(r"->"),
            "stream_api": re.compile(r"\.stream\(\)|\.collect\(|\.filter\(|\.map\(")
        }
        
        # Sample first 20 files for performance
        sample_files = java_files[:20] if len(java_files) > 20 else java_files
        
        for java_file in sample_files:
            try:
                content = java_file.read_text(encoding='utf-8')
                
                for pattern_name, regex in pattern_regexes.items():
                    patterns[pattern_name] += len(regex.findall(content))
                    
            except Exception as e:
                logger.warning(f"Could not read {java_file}: {e}")
                continue
        
        # Determine framework usage
        frameworks = []
        if patterns["spring_boot"] > 0:
            frameworks.append("spring_boot")
        elif patterns["spring_framework"] > 0:
            frameworks.append("spring_framework")
            
        if patterns["junit4_tests"] > 0:
            frameworks.append("junit4")
        elif patterns["junit5_tests"] > 0:
            frameworks.append("junit5")
        
        return {
            "java_files": len(java_files),
            "analyzed_files": len(sample_files),
            "patterns": patterns,
            "frameworks": frameworks,
            "migration_indicators": {
                "needs_javax_to_jakarta": patterns["javax_imports"] > 0,
                "needs_junit_migration": patterns["junit4_tests"] > 0,
                "has_deprecated_apis": patterns["deprecated_apis"] > 0,
                "modern_java_features": patterns["lambda_expressions"] > 0 or patterns["stream_api"] > 0
            }
        }
    
    def _analyze_dependencies(self, dependencies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dependencies using Maven Central API"""
        logger.info("Analyzing dependencies with Maven Central API")
        
        if not dependencies:
            return {"analyzed_dependencies": 0, "updates_available": []}
        
        analysis_results = []
        updates_available = []
        
        # Analyze a subset for performance
        deps_to_analyze = dependencies[:10] if len(dependencies) > 10 else dependencies
        
        for dep in deps_to_analyze:
            if not dep.get("group_id") or not dep.get("artifact_id"):
                continue
                
            try:
                # Get latest version from Maven Central
                latest_version = self.maven_api.get_latest_version(
                    dep["group_id"], 
                    dep["artifact_id"]
                )
                
                current_version = dep.get("version", "unknown")
                
                dep_analysis = {
                    "group_id": dep["group_id"],
                    "artifact_id": dep["artifact_id"],
                    "current_version": current_version,
                    "latest_version": latest_version,
                    "update_available": latest_version != current_version and latest_version != "unknown"
                }
                
                analysis_results.append(dep_analysis)
                
                if dep_analysis["update_available"]:
                    updates_available.append(dep_analysis)
                    
            except Exception as e:
                logger.warning(f"Could not analyze dependency {dep['group_id']}:{dep['artifact_id']}: {e}")
                continue
        
        return {
            "analyzed_dependencies": len(analysis_results),
            "total_dependencies": len(dependencies),
            "dependency_details": analysis_results,
            "updates_available": updates_available,
            "update_count": len(updates_available)
        }
    
    def _get_llm_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get LLM-powered analysis and recommendations"""
        logger.info("Getting LLM analysis")
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Analyze this Java project and provide migration recommendations:

Project Analysis Context:
{context}

Please provide:
1. Migration complexity assessment (simple/moderate/complex) with reasoning
2. Recommended migration strategy and order of operations  
3. Potential risks and challenges
4. Framework-specific considerations
5. Estimated effort and timeline

Format your response as structured recommendations.""")
        ])
        
        try:
            response = self.llm.invoke(
                analysis_prompt.format_messages(context=str(context))
            )
            
            # Parse LLM response (simplified - could be enhanced with structured output)
            content = response.content
            
            # Extract complexity assessment
            complexity = "moderate"  # default
            if "simple" in content.lower():
                complexity = "simple"
            elif "complex" in content.lower():
                complexity = "complex"
            
            return {
                "complexity": complexity,
                "recommendations": content,
                "llm_model": getattr(self.llm, 'model_name', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {
                "complexity": "moderate",
                "recommendations": f"LLM analysis failed: {e}",
                "error": str(e)
            }
    
    # Utility methods
    
    def _get_xml_text(self, element: ET.Element, tag: str, default: str = "") -> str:
        """Get text content from XML element"""
        elem = element.find(tag)
        return elem.text if elem is not None and elem.text else default
    
    def _detect_java_version(self, properties: Dict[str, str], root: ET.Element, namespace: str) -> str:
        """Detect Java version from various POM locations"""
        
        # Check properties
        for prop_name in ["java.version", "maven.compiler.source", "maven.compiler.target"]:
            if prop_name in properties:
                version = properties[prop_name]
                if version:
                    # Normalize version format
                    if version.startswith("1."):
                        return version.split(".")[-1]  # 1.8 -> 8
                    return version
        
        # Check compiler plugin configuration
        build_elem = root.find(f"{namespace}build")
        if build_elem is not None:
            plugins_elem = build_elem.find(f"{namespace}plugins")
            if plugins_elem is not None:
                for plugin in plugins_elem.findall(f"{namespace}plugin"):
                    artifact_id = self._get_xml_text(plugin, f"{namespace}artifactId")
                    if artifact_id == "maven-compiler-plugin":
                        config_elem = plugin.find(f"{namespace}configuration")
                        if config_elem is not None:
                            source = self._get_xml_text(config_elem, f"{namespace}source")
                            target = self._get_xml_text(config_elem, f"{namespace}target")
                            if source:
                                return source.replace("1.", "")
                            elif target:
                                return target.replace("1.", "")
        
        # Default assumption
        return "8"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()