# Copy the content from the test artifact here
# Due to size, you'll need to copy this from the Claude interface
"""
Test Suite for Java Migration Pipeline
Tests the core functionality of the migration system
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

from java_migration_pipeline import (
    CompilationError,
    ErrorType,
    ErrorParser,
    OpenRewriteOrchestrator,
    AIRepairAgent
)
from langchain_repair_agent import JavaMigrationTools, JavaMigrationAgent
from migration_orchestrator import MigrationConfig, MigrationMetrics, MigrationOrchestrator


class TestErrorParser:
    """Test the error parsing functionality"""
    
    def test_parse_maven_compilation_error(self):
        """Test parsing Maven compilation errors"""
        maven_output = """
[INFO] -------------------------------------------------------------
[ERROR] COMPILATION ERROR : 
[INFO] -------------------------------------------------------------
[ERROR] /home/user/project/src/main/java/com/example/UserService.java:[42,15] cannot find symbol
  symbol:   class ServletException
  location: package javax.servlet
[ERROR] /home/user/project/src/main/java/com/example/UserController.java:[10,25] package javax.validation does not exist
[INFO] 2 errors
"""
        
        parser = ErrorParser()
        errors = parser.parse_compilation_errors(maven_output)
        
        assert len(errors) == 2
        
        # Check first error
        error1 = errors[0]
        assert error1.file_path == "/home/user/project/src/main/java/com/example/UserService.java"
        assert error1.line_number == 42
        assert error1.column_number == 15
        assert error1.error_type == ErrorType.CANNOT_FIND_SYMBOL
        assert error1.details["symbol_name"] == "ServletException"
        assert error1.details["location_name"] == "javax.servlet"
        
        # Check second error
        error2 = errors[1]
        assert error2.error_type == ErrorType.PACKAGE_DOES_NOT_EXIST
    
    def test_parse_gradle_compilation_error(self):
        """Test parsing Gradle compilation errors"""
        gradle_output = """
> Task :compileJava FAILED
/src/main/java/com/example/Config.java:15: error: incompatible types: String cannot be converted to Integer
        Integer port = environment.getProperty("server.port");
                                              ^
"""
        
        parser = ErrorParser()
        # Note: This is a simplified test - real implementation would need Gradle-specific parsing
        # For now, testing the concept
        
    def test_parse_test_failures(self):
        """Test parsing test failure reports"""
        # Create a mock Surefire XML report
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="com.example.UserServiceTest" tests="3" failures="1">
    <testcase name="testGetUser" classname="com.example.UserServiceTest" time="0.123"/>
    <testcase name="testCreateUser" classname="com.example.UserServiceTest" time="0.456">
        <failure type="java.lang.AssertionError" message="Expected 201 but was 500">
java.lang.AssertionError: Expected 201 but was 500
    at org.junit.Assert.assertEquals(Assert.java:115)
    at com.example.UserServiceTest.testCreateUser(UserServiceTest.java:45)
        </failure>
    </testcase>
    <testcase name="testDeleteUser" classname="com.example.UserServiceTest" time="0.789"/>
</testsuite>"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_file = f.name
        
        # Mock the surefire directory structure
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.glob', return_value=[Path(temp_file)]):
            
            parser = ErrorParser()
            failures = parser.parse_test_failures("")
            
            assert len(failures) == 1
            failure = failures[0]
            assert failure.test_class == "com.example.UserServiceTest"
            assert failure.test_method == "testCreateUser"
            assert failure.failure_type == "java.lang.AssertionError"
            assert "Expected 201 but was 500" in failure.assertion_message


class TestJavaMigrationTools:
    """Test the Java migration tools"""
    
    @pytest.fixture
    def setup_test_project(self):
        """Create a temporary test project"""
        temp_dir = tempfile.mkdtemp()
        project_path = Path(temp_dir)
        
        # Create Maven project structure
        (project_path / "src/main/java/com/example").mkdir(parents=True)
        (project_path / "src/test/java/com/example").mkdir(parents=True)
        
        # Create pom.xml
        pom_content = """<?xml version="1.0" encoding="UTF-8"?>
<project>
    <groupId>com.example</groupId>
    <artifactId>test-project</artifactId>
    <version>1.0.0</version>
</project>"""
        (project_path / "pom.xml").write_text(pom_content)
        
        # Create a test Java file
        java_content = """package com.example;

import javax.servlet.ServletException;
import javax.validation.Valid;

public class UserService {
    public void createUser(@Valid User user) throws ServletException {
        // Implementation
    }
}"""
        (project_path / "src/main/java/com/example/UserService.java").write_text(java_content)
        
        yield str(project_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_read_code_range(self, setup_test_project):
        """Test reading code ranges from files"""
        tools = JavaMigrationTools(setup_test_project)
        
        result = tools.read_code_range(
            "src/main/java/com/example/UserService.java",
            3,  # import javax.servlet.ServletException;
            5   # import javax.validation.Valid;
        )
        
        assert "javax.servlet.ServletException" in result
        assert "javax.validation.Valid" in result
        assert "3:" in result  # Line numbers included
    
    def test_search_codebase(self, setup_test_project):
        """Test searching the codebase"""
        tools = JavaMigrationTools(setup_test_project)
        
        # Search for javax imports
        result = tools.search_codebase("javax.")
        
        assert "UserService.java" in result
        assert "javax.servlet.ServletException" in result or "javax.validation.Valid" in result
    
    def test_apply_code_patch(self, setup_test_project):
        """Test applying code patches"""
        tools = JavaMigrationTools(setup_test_project)
        
        # Read original content
        file_path = "src/main/java/com/example/UserService.java"
        original = tools.read_code_range(file_path, 1, 10)
        
        # Apply patch to update javax to jakarta
        result = tools.apply_code_patch(
            file_path,
            "import javax.servlet.ServletException;",
            "import jakarta.servlet.ServletException;"
        )
        
        assert "Patch applied successfully" in result
        
        # Verify the change
        updated = tools.read_code_range(file_path, 1, 10)
        assert "jakarta.servlet.ServletException" in updated
        assert "javax.servlet.ServletException" not in updated


class TestMigrationConfig:
    """Test configuration management"""
    
    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            "project_path": "/test/path",
            "project_name": "TestProject",
            "llm_provider": "anthropic",
            "llm_model": "claude-3-5-sonnet-20241022"
        }
        
        config = MigrationConfig(**config_dict)
        
        assert config.project_path == "/test/path"
        assert config.project_name == "TestProject"
        assert config.build_tool == "maven"  # Default value
        assert config.max_repair_attempts == 100  # Default value
    
    def test_config_yaml_roundtrip(self):
        """Test saving and loading config from YAML"""
        config = MigrationConfig(
            project_path="/test/path",
            project_name="TestProject",
            llm_provider="openai",
            llm_model="gpt-4-turbo"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config.to_yaml(f.name)
            
            # Load it back
            loaded_config = MigrationConfig.from_yaml(f.name)
            
            assert loaded_config.project_path == config.project_path
            assert loaded_config.llm_provider == config.llm_provider


class TestMigrationMetrics:
    """Test metrics tracking"""
    
    def test_record_and_report_metrics(self):
        """Test recording metrics and generating reports"""
        metrics = MigrationMetrics()
        
        # Record some errors and fixes
        metrics.record_error("compilation", {
            "error_type": "CANNOT_FIND_SYMBOL",
            "file": "UserService.java"
        })
        
        metrics.record_error("compilation", {
            "error_type": "CANNOT_FIND_SYMBOL",
            "file": "UserController.java"
        })
        
        metrics.record_fix("compilation", {
            "error_type": "CANNOT_FIND_SYMBOL",
            "fix": "Updated import"
        })
        
        # Generate report
        report = metrics.generate_report()
        
        assert report["metrics"]["compilation_errors_found"] == 2
        assert report["metrics"]["compilation_errors_fixed"] == 1
        assert report["summary"]["success_rate"] == 50.0
        
        # Check error distribution
        assert "CANNOT_FIND_SYMBOL" in report["error_distribution"]
        assert report["error_distribution"]["CANNOT_FIND_SYMBOL"] == 2


class TestAIRepairAgent:
    """Test the AI repair agent functionality"""
    
    @patch('langchain_repair_agent.ChatAnthropic')
    def test_agent_initialization(self, mock_llm):
        """Test agent initialization"""
        agent = JavaMigrationAgent("/test/path", "anthropic", "claude-3-5-sonnet-20241022")
        
        assert agent.project_path == Path("/test/path")
        assert len(agent.tools) > 0
        assert agent.agent_executor is not None
    
    @patch('subprocess.run')
    def test_agent_tools_execution(self, mock_run):
        """Test that agent tools can be executed"""
        # Mock subprocess for search
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="UserService.java:10: Found match"
        )
        
        tools = JavaMigrationTools("/test/path")
        result = tools.search_codebase("test query")
        
        assert "UserService.java:10" in result
        mock_run.assert_called_once()


class TestIntegrationScenarios:
    """Test complete migration scenarios"""
    
    @pytest.mark.integration
    def test_javax_to_jakarta_migration(self, setup_test_project):
        """Test a complete javax to jakarta migration scenario"""
        # This would be an integration test that:
        # 1. Sets up a project with javax imports
        # 2. Runs OpenRewrite to do bulk conversion
        # 3. Uses AI agent to fix remaining issues
        # 4. Verifies all imports are now jakarta
        pass
    
    @pytest.mark.integration  
    def test_spring_security_migration(self, setup_test_project):
        """Test Spring Security configuration migration"""
        # This would test:
        # 1. WebSecurityConfigurerAdapter removal
        # 2. Conversion to SecurityFilterChain
        # 3. AI agent handling of complex security configs
        pass


# Fixtures for mocking LLM responses
@pytest.fixture
def mock_llm_responses():
    """Mock responses from the LLM"""
    return {
        "javax_to_jakarta": {
            "thoughts": "The error is due to javax package not being found. Need to update to jakarta.",
            "command": "apply_code_patch",
            "args": {
                "file_path": "src/main/java/com/example/UserService.java",
                "old_text": "import javax.servlet.ServletException;",
                "new_text": "import jakarta.servlet.ServletException;"
            }
        },
        "spring_security_fix": {
            "thoughts": "WebSecurityConfigurerAdapter is removed in Spring Boot 3. Need to use SecurityFilterChain.",
            "command": "apply_code_patch",
            "args": {
                "file_path": "src/main/java/com/example/SecurityConfig.java",
                "old_text": "public class SecurityConfig extends WebSecurityConfigurerAdapter {",
                "new_text": "public class SecurityConfig {"
            }
        }
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])