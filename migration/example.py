#!/usr/bin/env python3
"""
Example usage of the Intelligent Java Migration System

This script demonstrates how to use the agent-based migration system
to migrate a Java project from Java 8/11/17 to Java 21 with Spring Boot 3.x.
"""

import os
import sys
import logging
from pathlib import Path

from langchain_openai import ChatOpenAI

# Add the migration directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from working_orchestrator import WorkingMigrationOrchestrator as MigrationOrchestrator, MigrationConfig
    print("✅ Using working LangGraph orchestrator")
except ImportError:
    try:
        from simple_orchestrator import SimpleMigrationOrchestrator as MigrationOrchestrator, MigrationConfig
        print("✅ Using simplified orchestrator (fallback)")
    except ImportError:
        print("⚠️ Using original orchestrator (may have issues)")
        from orchestrator import MigrationOrchestrator, MigrationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main example function demonstrating the migration system"""
    
    print("🚀 Intelligent Java Migration System - Example")
    print("=" * 60)
    
    # Example 1: Basic migration with default configuration
    print("\n📋 Example 1: Basic Migration")
    print("-" * 30)
    
    try:
        # Initialize the orchestrator with default settings
        orchestrator = MigrationOrchestrator()
        
        # Path to the example project (xsync)
        project_path = "./xsync"
        
        if Path(project_path).exists():
            print(f"🔍 Migrating project: {project_path}")
            
            # Run the migration
            result = orchestrator.migrate_repository(project_path)
            
            # Display results
            print(f"\n📊 Migration Results:")
            print(f"   Success: {'✅' if result['success'] else '❌'}")
            print(f"   Project: {result['project_name']}")
            print(f"   Duration: {result.get('duration', 'Unknown')}")
            print(f"   Phases Completed: {len(result.get('phases_completed', []))}")
            print(f"   Recipes Applied: {len(result.get('applied_recipes', []))}")
            print(f"   Fixes Applied: {len(result.get('fixes_applied', []))}")
            
            if result.get('requires_human_intervention'):
                print(f"   ⚠️  Human intervention required")
                print(f"   Error: {result.get('error_message', 'Unknown error')}")
            
        else:
            print(f"❌ Project path does not exist: {project_path}")
            print("   Please ensure the xsync project is available")
    
    except Exception as e:
        logger.error(f"Basic migration example failed: {e}")
        print(f"❌ Example failed: {e}")
    
    # Example 2: Custom configuration migration
    print("\n📋 Example 2: Custom Configuration Migration")
    print("-" * 45)
    
    try:
        # Create custom configuration
        custom_config = MigrationConfig(
            target_java_version="21",
            target_spring_version="6.0",
            target_spring_boot_version="3.2",
            enable_jakarta_migration=True,
            enable_junit5_migration=True,
            dry_run=True,  # Dry run mode for testing
            backup_enabled=True,
            max_retry_attempts=5
        )
        
        # Initialize orchestrator with custom config
        custom_orchestrator = MigrationOrchestrator(config=custom_config)
        
        print(f"🔧 Configuration:")
        print(f"   Target Java Version: {custom_config.target_java_version}")
        print(f"   Target Spring Boot: {custom_config.target_spring_boot_version}")
        print(f"   Jakarta Migration: {custom_config.enable_jakarta_migration}")
        print(f"   JUnit 5 Migration: {custom_config.enable_junit5_migration}")
        print(f"   Dry Run Mode: {custom_config.dry_run}")
        print(f"   Backup Enabled: {custom_config.backup_enabled}")
        
        # For this example, we'll just show the configuration
        # In a real scenario, you would call:
        result = custom_orchestrator.migrate_repository(project_path)
        
        print("✅ Custom configuration example completed")
        
    except Exception as e:
        logger.error(f"Custom configuration example failed: {e}")
        print(f"❌ Example failed: {e}")
    
    # Example 3: Demonstrate individual tool usage
    print("\n📋 Example 3: Individual Tool Usage")
    print("-" * 35)
    
    try:
        # Demonstrate Maven Central API
        from tools.maven_api import MavenCentralAPI
        
        print("🔍 Testing Maven Central API...")
        maven_api = MavenCentralAPI()
        
        # Check latest version of a common dependency
        latest_version = maven_api.get_latest_version("org.springframework", "spring-core")
        print(f"   Latest Spring Core version: {latest_version}")
        
        # Check Java 21 compatibility
        compatibility = maven_api.check_java_compatibility(
            "org.springframework", "spring-core", latest_version, "21"
        )
        print(f"   Java 21 compatible: {compatibility['compatible']}")
        
        # Demonstrate Command Executor
        from tools.command_executor import CommandExecutor
        
        print("\n🔧 Testing Command Executor...")
        executor = CommandExecutor()
        
        # Get system info
        system_info = executor.get_system_info()
        print(f"   Java Home: {system_info.get('java_home', 'Not set')}")
        print(f"   Maven Home: {system_info.get('maven_home', 'Not set')}")
        
        # Demonstrate File Operations
        from tools.file_operations import FileOperations
        
        print("\n📁 Testing File Operations...")
        file_ops = FileOperations()
        
        # List Java files in the project (if it exists)
        if Path("./xsync").exists():
            java_files = file_ops.list_files("./xsync", "*.java", recursive=True)
            print(f"   Found {java_files['file_count']} Java files")
        
        print("✅ Individual tools example completed")
        
    except Exception as e:
        logger.error(f"Tools example failed: {e}")
        print(f"❌ Example failed: {e}")
    
    # Example 4: Analysis-only mode
    print("\n📋 Example 4: Analysis-Only Mode")
    print("-" * 30)
    
    try:
        from agents.analysis_agent import AnalysisAgent
        from langchain_anthropic import ChatAnthropic
        
        print("🔍 Running repository analysis...")
        
        # Initialize analysis agent
        # llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.1)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=16384, api_key=os.getenv("OPENAI_API_KEY"))
        analysis_agent = AnalysisAgent(llm)
        
        if Path("./xsync").exists():
            # Analyze the project
            analysis_result = analysis_agent.analyze("./xsync")
            
            print(f"📊 Analysis Results:")
            print(f"   Java Version: {analysis_result.get('java_version', 'Unknown')}")
            print(f"   Build System: {analysis_result.get('build_system', 'Unknown')}")
            print(f"   Java Files: {analysis_result.get('java_file_count', 0)}")
            print(f"   Dependencies: {analysis_result.get('dependency_count', 0)}")
            print(f"   Complexity: {analysis_result.get('complexity', 'Unknown')}")
            
            if 'llm_insights' in analysis_result:
                print(f"   LLM Complexity Assessment: {analysis_result['llm_insights'].get('complexity', 'Unknown')}")
        
        print("✅ Analysis-only example completed")
        
    except Exception as e:
        logger.error(f"Analysis example failed: {e}")
        print(f"❌ Example failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 All examples completed!")
    print("\nTo run a full migration:")
    print("1. Ensure your Java project has a pom.xml file")
    print("2. Set up your environment variables (JAVA_HOME, etc.)")
    print("3. Configure your LLM API keys (ANTHROPIC_API_KEY)")
    print("4. Run: python example.py")
    print("\nFor more advanced usage, see the repo.md documentation.")


if __name__ == "__main__":
    main()