"""
LangChain tools for Maven operations in migration agents
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List
import requests
import json

@tool
def read_pom(project_path: str) -> str:
    """Read and parse pom.xml to extract basic project information."""
    try:
        pom_path = Path(project_path) / "pom.xml"
        if not pom_path.exists():
            return "Error: pom.xml not found"
        
        tree = ET.parse(pom_path)
        root = tree.getroot()
        
        # Handle namespace
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"
        
        # Extract basic info
        groupId = _get_text(root, f"{namespace}groupId")
        artifactId = _get_text(root, f"{namespace}artifactId")
        version = _get_text(root, f"{namespace}version")
        
        # Extract Java version
        java_version = "Unknown"
        props = root.find(f"{namespace}properties")
        if props is not None:
            java_version = (_get_text(props, f"{namespace}java.version") or 
                          _get_text(props, f"{namespace}maven.compiler.source") or 
                          _get_text(props, f"{namespace}maven.compiler.target") or 
                          "8")
        
        # Count dependencies
        deps = root.find(f"{namespace}dependencies")
        dep_count = len(deps.findall(f"{namespace}dependency")) if deps is not None else 0
        
        return f"""Project Info:
- GroupId: {groupId}
- ArtifactId: {artifactId}
- Version: {version}
- Java Version: {java_version}
- Dependencies: {dep_count}"""
        
    except Exception as e:
        return f"Error reading pom.xml: {str(e)}"

@tool
def get_java_version(project_path: str) -> str:
    """Get the current Java version from pom.xml."""
    try:
        pom_path = Path(project_path) / "pom.xml"
        tree = ET.parse(pom_path)
        root = tree.getroot()
        
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"
        
        props = root.find(f"{namespace}properties")
        if props is not None:
            java_version = (_get_text(props, f"{namespace}java.version") or 
                          _get_text(props, f"{namespace}maven.compiler.source") or 
                          _get_text(props, f"{namespace}maven.compiler.target"))
            if java_version:
                return java_version if java_version != "1.8" else "8"
        
        return "8"  # Default
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def update_java_version(project_path: str, java_version: str) -> str:
    """Update Java version in pom.xml properties."""
    try:
        pom_path = Path(project_path) / "pom.xml"
        content = pom_path.read_text(encoding='utf-8')
        
        # Simple replacements for common Java version properties
        replacements = [
            (f'<java.version>1.8</java.version>', f'<java.version>{java_version}</java.version>'),
            (f'<maven.compiler.source>1.8</maven.compiler.source>', f'<maven.compiler.source>{java_version}</maven.compiler.source>'),
            (f'<maven.compiler.target>1.8</maven.compiler.target>', f'<maven.compiler.target>{java_version}</maven.compiler.target>'),
            (f'<java.version>8</java.version>', f'<java.version>{java_version}</java.version>'),
            (f'<maven.compiler.source>8</maven.compiler.source>', f'<maven.compiler.source>{java_version}</maven.compiler.source>'),
            (f'<maven.compiler.target>8</maven.compiler.target>', f'<maven.compiler.target>{java_version}</maven.compiler.target>')
        ]
        
        changes_made = 0
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                changes_made += 1
        
        if changes_made > 0:
            pom_path.write_text(content, encoding='utf-8')
            return f"Updated Java version to {java_version} in pom.xml ({changes_made} changes made)"
        else:
            return f"No Java version properties found to update in pom.xml"
            
    except Exception as e:
        return f"Error updating Java version: {str(e)}"

@tool
def list_dependencies(project_path: str) -> str:
    """List all dependencies from pom.xml."""
    try:
        pom_path = Path(project_path) / "pom.xml"
        tree = ET.parse(pom_path)
        root = tree.getroot()
        
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"
        
        deps = root.find(f"{namespace}dependencies")
        if deps is None:
            return "No dependencies found"
        
        dependencies = []
        for dep in deps.findall(f"{namespace}dependency"):
            groupId = _get_text(dep, f"{namespace}groupId")
            artifactId = _get_text(dep, f"{namespace}artifactId")
            version = _get_text(dep, f"{namespace}version", "managed")
            scope = _get_text(dep, f"{namespace}scope", "compile")
            dependencies.append(f"- {groupId}:{artifactId}:{version} ({scope})")
        
        return f"Dependencies ({len(dependencies)}):\n" + "\n".join(dependencies)
        
    except Exception as e:
        return f"Error listing dependencies: {str(e)}"

@tool
def add_openrewrite_plugin(project_path: str) -> str:
    """Add basic OpenRewrite Maven plugin to pom.xml if not present."""
    try:
        pom_path = Path(project_path) / "pom.xml"
        content = pom_path.read_text(encoding='utf-8')
        
        if "rewrite-maven-plugin" in content:
            return "OpenRewrite plugin already present in pom.xml"
        
        plugin_xml = '''            <plugin>
                <groupId>org.openrewrite.maven</groupId>
                <artifactId>rewrite-maven-plugin</artifactId>
                <version>5.3.0</version>
            </plugin>'''
        
        # Insert plugin before closing </plugins> tag
        if "</plugins>" in content:
            content = content.replace("</plugins>", f"{plugin_xml}\n        </plugins>")
        elif "<build>" in content and "</build>" in content:
            # Add plugins section to build
            plugins_section = f'''        <plugins>
{plugin_xml}
        </plugins>'''
            content = content.replace("</build>", f"{plugins_section}\n    </build>")
        else:
            # Add entire build section
            build_section = f'''    <build>
        <plugins>
{plugin_xml}
        </plugins>
    </build>'''
            content = content.replace("</project>", f"{build_section}\n</project>")
        
        pom_path.write_text(content, encoding='utf-8')
        return "Successfully added OpenRewrite plugin to pom.xml"
        
    except Exception as e:
        return f"Error adding OpenRewrite plugin: {str(e)}"

class ConfigureOpenRewriteRecipesInput(BaseModel):
    """Input for configure_openrewrite_recipes tool."""
    project_path: str = Field(description="Path to the project directory containing pom.xml")
    recipes: List[str] = Field(description="List of OpenRewrite recipe names to configure")

@tool(args_schema=ConfigureOpenRewriteRecipesInput)
def configure_openrewrite_recipes(project_path: str, recipes: List[str]) -> str:
    """Configure OpenRewrite plugin with active recipes and dependencies in pom.xml."""
    try:
        pom_path = Path(project_path) / "pom.xml"
        content = pom_path.read_text(encoding='utf-8')
        
        if "rewrite-maven-plugin" not in content:
            return "Error: OpenRewrite plugin not found. Add plugin first using add_openrewrite_plugin."
        
        # Build active recipes configuration
        active_recipes = ""
        for recipe in recipes:
            active_recipes += f"                        <recipe>{recipe}</recipe>\n"
        
        # Complete plugin configuration with recipes and dependencies
        new_plugin_config = f'''            <plugin>
                <groupId>org.openrewrite.maven</groupId>
                <artifactId>rewrite-maven-plugin</artifactId>
                <version>5.3.0</version>
                <configuration>
                    <activeRecipes>
{active_recipes.rstrip()}
                    </activeRecipes>
                </configuration>
                <dependencies>
                    <dependency>
                        <groupId>org.openrewrite.recipe</groupId>
                        <artifactId>rewrite-migrate-java</artifactId>
                        <version>2.0.7</version>
                    </dependency>
                </dependencies>
            </plugin>'''
        
        # Replace the existing plugin configuration
        import re
        plugin_pattern = r'<plugin>\s*<groupId>org\.openrewrite\.maven</groupId>.*?</plugin>'
        if re.search(plugin_pattern, content, re.DOTALL):
            content = re.sub(plugin_pattern, new_plugin_config, content, flags=re.DOTALL)
        else:
            return "Error: Could not find OpenRewrite plugin to replace"
        
        pom_path.write_text(content, encoding='utf-8')
        return f"Successfully configured OpenRewrite plugin with {len(recipes)} active recipes"
        
    except Exception as e:
        return f"Error configuring OpenRewrite recipes: {str(e)}"

@tool 
def add_rewrite_dependency(project_path: str, dependency_artifact: str, version: str = "2.0.7") -> str:
    """Add a specific OpenRewrite recipe dependency to the plugin."""
    try:
        pom_path = Path(project_path) / "pom.xml"
        content = pom_path.read_text(encoding='utf-8')
        
        if "rewrite-maven-plugin" not in content:
            return "Error: OpenRewrite plugin not found in pom.xml"
        
        new_dependency = f'''                    <dependency>
                        <groupId>org.openrewrite.recipe</groupId>
                        <artifactId>{dependency_artifact}</artifactId>
                        <version>{version}</version>
                    </dependency>'''
        
        # Check if dependencies section exists in the plugin
        if "<dependencies>" in content and "rewrite-maven-plugin" in content:
            # Add to existing dependencies
            content = content.replace("</dependencies>", f"{new_dependency}\n                </dependencies>")
        else:
            # Add dependencies section to plugin
            dependencies_section = f'''                <dependencies>
{new_dependency}
                </dependencies>'''
            # Insert before closing plugin tag
            content = content.replace("</plugin>", f"{dependencies_section}\n            </plugin>")
        
        pom_path.write_text(content, encoding='utf-8')
        return f"Successfully added dependency {dependency_artifact}:{version} to OpenRewrite plugin"
        
    except Exception as e:
        return f"Error adding OpenRewrite dependency: {str(e)}"

@tool
def get_latest_version_from_maven_central(group_id: str, artifact_id: str) -> str:
    """Query Maven Central to find the latest version of a dependency or plugin."""
    try:
        # Maven Central search API endpoint
        url = f"https://search.maven.org/solrsearch/select"
        params = {
            "q": f"g:{group_id} AND a:{artifact_id}",
            "core": "gav",
            "rows": 1,
            "wt": "json"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("response", {}).get("numFound", 0) == 0:
            return f"No artifact found for {group_id}:{artifact_id}"
        
        docs = data.get("response", {}).get("docs", [])
        if not docs:
            return f"No version information found for {group_id}:{artifact_id}"
        
        latest_version = docs[0].get("v", "unknown")
        timestamp = docs[0].get("timestamp", 0)
        
        # Convert timestamp to readable date
        import datetime
        date_str = datetime.datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d") if timestamp else "unknown"
        
        return f"Latest version of {group_id}:{artifact_id} is {latest_version} (published: {date_str})"
        
    except requests.RequestException as e:
        return f"Error querying Maven Central: {str(e)}"
    except Exception as e:
        return f"Error processing Maven Central response: {str(e)}"

@tool  
def get_spring_boot_latest_version() -> str:
    """Get the latest Spring Boot 3.x version from Maven Central."""
    try:
        # Query for Spring Boot starter parent
        url = "https://search.maven.org/solrsearch/select"
        params = {
            "q": "g:org.springframework.boot AND a:spring-boot-starter-parent AND v:3.*",
            "core": "gav", 
            "rows": 10,
            "wt": "json"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        docs = data.get("response", {}).get("docs", [])
        
        if not docs:
            return "No Spring Boot 3.x versions found"
        
        # Get the latest version (first in results)
        latest = docs[0]
        version = latest.get("v", "unknown")
        timestamp = latest.get("timestamp", 0)
        
        import datetime
        date_str = datetime.datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d") if timestamp else "unknown"
        
        return f"Latest Spring Boot 3.x version: {version} (published: {date_str})"
        
    except Exception as e:
        return f"Error getting Spring Boot version: {str(e)}"

@tool
def get_spring_framework_latest_version() -> str:
    """Get the latest Spring Framework 6.x version from Maven Central."""
    try:
        url = "https://search.maven.org/solrsearch/select"
        params = {
            "q": "g:org.springframework AND a:spring-core AND v:6.*",
            "core": "gav",
            "rows": 10, 
            "wt": "json"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        docs = data.get("response", {}).get("docs", [])
        
        if not docs:
            return "No Spring Framework 6.x versions found"
        
        latest = docs[0]
        version = latest.get("v", "unknown")
        timestamp = latest.get("timestamp", 0)
        
        import datetime
        date_str = datetime.datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d") if timestamp else "unknown"
        
        return f"Latest Spring Framework 6.x version: {version} (published: {date_str})"
        
    except Exception as e:
        return f"Error getting Spring Framework version: {str(e)}"

def _get_text(element, tag: str, default: str = "") -> str:
    """Get text from XML element safely."""
    child = element.find(tag)
    return child.text if child is not None and child.text else default

# Collect all Maven tools
maven_tools = [
    read_pom, get_java_version, update_java_version, list_dependencies, 
    add_openrewrite_plugin, configure_openrewrite_recipes, add_rewrite_dependency,
    get_latest_version_from_maven_central, get_spring_boot_latest_version, get_spring_framework_latest_version
]