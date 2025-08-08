import os
import sys
from typing import Dict, Any, List
from dotenv import load_dotenv

# Add migration directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'migration'))
from supervisor_orchestrator import SupervisorMigrationOrchestrator

# Migration orchestrator handles message formatting internally

load_dotenv()


def create_supervisor_system():
    """Create and configure the migration supervisor"""
    return SupervisorMigrationOrchestrator()


def main():
    """Test the supervisor locally"""
    supervisor = create_supervisor_system()
    
    print("Migration Supervisor Demo")
    print("Available workers: analysis_expert, execution_expert, error_expert")
    print("Try providing a project path for migration")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        
        try:
            # Use migrate_project method instead of invoke
            if os.path.exists(user_input.strip()):
                result = supervisor.migrate_project(user_input.strip())
                if result.get('success'):
                    print(f"Migration successful: {result.get('result')}\n")
                else:
                    print(f"Migration failed: {result.get('error')}\n")
            else:
                print("Please provide a valid project path or 'quit' to exit\n")
                
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()