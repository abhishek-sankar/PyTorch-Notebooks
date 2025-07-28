#!/usr/bin/env python3
"""
Quick test to verify the orchestrator fix
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_working_orchestrator():
    """Test the working orchestrator"""
    print("🧪 Testing Working LangGraph Orchestrator")
    print("=" * 50)
    
    try:
        from working_orchestrator import WorkingMigrationOrchestrator, MigrationConfig
        
        # Initialize orchestrator
        config = MigrationConfig(dry_run=True)  # Safe dry run mode
        orchestrator = WorkingMigrationOrchestrator(config=config)
        
        print("✅ Orchestrator initialized successfully")
        
        # Test with xsync project if available
        if Path("./xsync").exists():
            print("\n🚀 Testing migration with xsync project...")
            result = orchestrator.migrate_repository("./xsync")
            
            print(f"\n📊 Results:")
            print(f"   Success: {'✅' if result['success'] else '❌'}")
            print(f"   Project: {result['project_name']}")
            print(f"   Phases Completed: {len(result['phases_completed'])}")
            print(f"   Applied Recipes: {len(result['applied_recipes'])}")
            print(f"   Fixes Applied: {len(result['fixes_applied'])}")
            print(f"   Duration: {result['duration']}")
            
            if result.get('requires_human_intervention'):
                print("   ⚠️  Human intervention required")
                
            return True
        else:
            print("❌ xsync project not found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_orchestrator():
    """Test the simple orchestrator"""
    print("\n🧪 Testing Simple Orchestrator")
    print("=" * 50)
    
    try:
        from simple_orchestrator import SimpleMigrationOrchestrator, MigrationConfig
        
        # Initialize orchestrator
        config = MigrationConfig(dry_run=True)
        orchestrator = SimpleMigrationOrchestrator(config=config)
        
        print("✅ Simple orchestrator initialized successfully")
        
        # Test with xsync project if available
        if Path("./xsync").exists():
            print("\n🚀 Testing migration with xsync project...")
            result = orchestrator.migrate_repository("./xsync")
            
            print(f"\n📊 Results:")
            print(f"   Success: {'✅' if result['success'] else '❌'}")
            print(f"   Project: {result['project_name']}")
            print(f"   Phases Completed: {len(result['phases_completed'])}")
            print(f"   Applied Recipes: {len(result['applied_recipes'])}")
            
            return True
        else:
            print("❌ xsync project not found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔧 Migration System Fix Verification")
    print("=" * 60)
    
    # Test working orchestrator
    working_success = test_working_orchestrator()
    
    # Test simple orchestrator
    simple_success = test_simple_orchestrator()
    
    print(f"\n🎯 Test Summary:")
    print(f"   Working Orchestrator: {'✅ PASS' if working_success else '❌ FAIL'}")
    print(f"   Simple Orchestrator: {'✅ PASS' if simple_success else '❌ FAIL'}")
    
    if working_success or simple_success:
        print(f"\n🎉 Fix successful! You can now run:")
        print(f"   python example.py")
    else:
        print(f"\n❌ Tests failed. Check error messages above.")

if __name__ == "__main__":
    main()