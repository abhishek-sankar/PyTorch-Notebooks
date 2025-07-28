# Intelligent Java Migration System

## Overview
An advanced agent-based system for automating Java 8/11/17 to Java 21 migrations with Spring Boot 3.x and Jakarta EE support using LangChain/LangGraph orchestration. The system provides human-like decision making through intelligent agents that can analyze, execute, and fix migration issues automatically.

## Problem Statement
- Manual Java version migrations are time-consuming and error-prone
- Spring Boot 3 introduces breaking changes (javax → jakarta, deprecated libraries)
- Organizations need consistent, repeatable migration processes
- Complex dependency conflicts require expert knowledge

## 🏗️ System Architecture

### Core Components (Implemented)

#### 1. **Migration Orchestrator** (`orchestrator.py`)
- **LangGraph StateGraph**: Manages migration workflow as a state machine
- **Agent Coordination**: Routes decisions between specialized agents
- **State Persistence**: SQLite-backed checkpointing for recovery
- **Human Escalation**: Automatic escalation for complex scenarios
- **Error Recovery**: Retry logic with intelligent backoff

#### 2. **Analysis Agent** (`agents/analysis_agent.py`)
- **Repository Analysis**: Deep project structure and dependency analysis
- **Maven Central Integration**: Real-time version checking and compatibility
- **LLM-Powered Insights**: Intelligent complexity assessment and recommendations
- **Risk Assessment**: Automated migration difficulty scoring
- **Framework Detection**: Identifies Spring Boot, JUnit, Jakarta usage patterns

#### 3. **Execution Agent** (`agents/execution_agent.py`)
- **OpenRewrite Integration**: Automated recipe execution with monitoring
- **Recipe Orchestration**: LLM determines optimal execution order
- **Progress Tracking**: Real-time monitoring with timeout management
- **Compilation Validation**: Automatic validation after each recipe
- **Adaptive Strategy**: Dynamic adjustment based on execution results

#### 4. **Error Fixing Agent** (`agents/error_agent.py`)
- **Error Categorization**: Intelligent classification of compilation errors
- **Pattern-Based Fixes**: Automated fixes for common migration issues
- **LLM-Assisted Resolution**: Complex error analysis and fixing
- **Code Transformation**: Direct file modification with backup support
- **Iterative Improvement**: Learning from previous fix attempts

### 🛠️ Advanced Tooling (Implemented)

#### **File Operations** (`tools/file_operations.py`)
- **Safe File Manipulation**: Read/write with backup and rollback capabilities
- **Pattern-Based Modifications**: Find/replace with regex support
- **Change Tracking**: Complete operation history and audit trail
- **Backup Management**: Automatic backup creation and cleanup

#### **Command Executor** (`tools/command_executor.py`)
- **Monitored Execution**: Command execution with timeout and progress tracking
- **Maven Integration**: Specialized support for Maven commands (compile, test, package)
- **Process Management**: Safe process termination and resource cleanup
- **Output Parsing**: Intelligent parsing of Maven and OpenRewrite output

#### **Maven Central API** (`tools/maven_api.py`)
- **Version Discovery**: Real-time latest version checking with caching
- **Compatibility Analysis**: Java 21 compatibility assessment for dependencies
- **Dependency Analysis**: Comprehensive dependency update recommendations
- **Rate Limiting**: Respectful API usage with automatic throttling

#### **OpenRewrite Client** (`tools/openrewrite_client.py`)
- **Recipe Execution**: Direct OpenRewrite recipe execution and monitoring
- **Configuration Management**: Dynamic rewrite.yml generation
- **Compatibility Validation**: Project compatibility checking
- **Recipe Discovery**: Automatic detection of applicable recipes

### 🔄 Agent Communication (LangGraph Implementation)

#### **State-Based Orchestration**
- **Shared State**: All agents operate on a common `MigrationState` object
- **Message Passing**: Agents communicate through structured messages using LangChain
- **Event-Driven Flow**: Conditional edges based on execution results and error states
- **Persistence**: SQLite-based state checkpointing for recovery and resume

#### **Workflow Coordination**
```python
# LangGraph workflow structure
workflow = StateGraph(MigrationState)
workflow.add_node("analyze_repository", self._analyze_repository)
workflow.add_node("execute_recipes", self._execute_recipes)  
workflow.add_node("fix_errors", self._fix_errors)
workflow.add_conditional_edges(
    "execute_recipes",
    self._should_fix_errors,
    {"fix_errors": "fix_errors", "validate_tests": "validate_tests"}
)
```

#### **Benefits of LangGraph Approach**
- **Deterministic Flow**: Predictable execution paths with clear state transitions
- **Error Recovery**: Automatic retry and rollback capabilities
- **Human Integration**: Seamless escalation points with context preservation
- **Debugging**: Complete execution trace and state inspection

## Technical Specifications

### Prerequisites
- Python 3.9+
- Java 11+ (for running OpenRewrite)
- Maven 3.6+
- Git
- Docker (optional, for Redis/RabbitMQ)

### Key Dependencies
```python
# requirements.txt
langchain>=0.1.0
langchain-openai>=0.0.5
smolagents>=0.1.0  # For specialized micro-agents
openai>=1.0.0
redis>=5.0.0
gitpython>=3.1.0
pydantic>=2.0.0
```

### Project Constraints
- **Project Type**: Maven only (single module)
- **Project Size**: 30-50,000 lines of code
- **Migration Type**: Complete migration (no partial support)
- **Migration Path**: Java 11 → 21, Spring Boot 2.x → 3.x

## Migration Capabilities

### Supported Transformations
1. **Java Version Upgrades**
   - Java 11 → 17 → 21 (incremental approach)
   - New language feature adoption

2. **Spring Boot Migration**
   - Spring Boot 2.7 → 3.x
   - Spring Security updates
   - Spring Data changes

3. **Dependency Migrations**
   - javax.* → jakarta.* namespace
   - JUnit 4 → JUnit 5
   - Jackson version alignment
   - Dozer → MapStruct/ModelMapper migration

4. **Custom Transformations**
   - Organization-specific patterns
   - Legacy code modernization

### OpenRewrite Integration

#### Core Recipes
```yaml
# rewrite.yml
type: specs.openrewrite.org/v1beta/recipe
name: com.organization.JavaMigration
recipes:
  - org.openrewrite.java.migrate.Java11to17
  - org.openrewrite.java.migrate.Java17to21
  - org.openrewrite.java.spring.boot3.UpgradeSpringBoot_3_0
  - org.openrewrite.java.migrate.javax.MigrateJavaxToJakarta
  - org.openrewrite.java.testing.junit5.JUnit4to5Migration
```

#### Custom Recipe Creation
- Dynamic recipe generation based on detected patterns
- Template-based approach for common transformations
- Storage in project-specific recipe catalog

## Human-in-the-Loop Protocol

### Escalation Triggers
1. **Error Loop Detection**
   - Same error encountered 3+ times
   - No progress after 5 consecutive actions
   
2. **Ambiguous Migrations**
   - Multiple valid migration paths
   - Custom business logic in deprecated APIs
   
3. **High-Risk Changes**
   - Core functionality modifications
   - Security-related code changes
   
4. **Unknown Patterns**
   - Code patterns not covered by existing recipes
   - Complex custom annotations

### Risk-Based Escalation Levels

| Risk Level | Criteria | Human Involvement |
|------------|----------|-------------------|
| Low | Standard recipe available, all tests pass | Auto-approve |
| Medium | Minor test failures, known workarounds | Review summary |
| High | Major API changes, multiple test failures | Detailed review |
| Critical | Security implications, core logic changes | Full approval required |

### Human Decision Interface
```json
{
  "escalation_type": "ambiguous_migration",
  "context": {
    "file": "src/main/java/com/example/Service.java",
    "issue": "Dozer mapping with custom converters",
    "current_code": "...",
    "options": [
      {
        "id": "mapstruct",
        "description": "Migrate to MapStruct with custom mappers",
        "confidence": 0.8,
        "impact": "Requires new dependencies"
      },
      {
        "id": "manual",
        "description": "Create manual mapping methods",
        "confidence": 0.6,
        "impact": "More code but no new dependencies"
      }
    ]
  },
  "recommendation": "mapstruct"
}
```

### Decision Persistence Options

1. **File-Based** (Simple)
   ```json
   // migrations/decisions.json
   {
     "project_id": "...",
     "decisions": [
       {
         "timestamp": "...",
         "context": "...",
         "choice": "...",
         "rationale": "..."
       }
     ]
   }
   ```

2. **Database** (Scalable)
   - SQLite for single-machine use
   - PostgreSQL for team environments
   
3. **Git-Based** (Recommended)
   - Store in `.migration/decisions/` directory
   - Version controlled with the code
   - Easy to share across team

## Implementation Workflow

### Phase 1: Initial Setup
```bash
# Create migration branch
git checkout -b migration/java21-upgrade

# Initialize migration workspace
migration-agent init --project-path ./
```

### Phase 2: Analysis
```bash
# Generate migration report
migration-agent analyze --output migration-report.md
```

### Phase 3: Incremental Migration
```python
# Pseudo-code for migration flow
def migrate_project():
    # 1. Core dependencies first
    migrate_java_version(11, 17)
    run_tests_and_checkpoint()
    
    migrate_java_version(17, 21)
    run_tests_and_checkpoint()
    
    # 2. Framework upgrades
    migrate_spring_boot()
    migrate_javax_to_jakarta()
    
    # 3. Testing framework
    migrate_junit4_to_5()
    
    # 4. Other dependencies
    migrate_jackson()
    handle_dozer_migration()
    
    # 5. Custom patterns
    apply_custom_recipes()
```

### Git Strategy
- Each major step creates a commit
- Failed attempts are stashed with metadata
- Successful migrations are tagged
- Easy rollback to any checkpoint

## Memory and State Management

### LangChain Memory Configuration
```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory

# Redis-backed memory for persistence
message_history = RedisChatMessageHistory(
    url="redis://localhost:6379",
    ttl=86400,  # 24 hour TTL
    session_id="migration_session_x"
)

memory = ConversationSummaryBufferMemory(
    chat_memory=message_history,
    max_token_limit=200000,
    return_messages=True
)
```

### Migration State Schema
```python
class MigrationState(BaseModel):
    project_path: str
    current_phase: str
    completed_steps: List[str]
    failed_attempts: Dict[str, List[str]]  # step -> error messages
    checkpoints: List[str]  # git commit hashes
    human_decisions: List[Decision]
    risk_assessment: Dict[str, str]
```

## Error Recovery

### Failure Handling
1. **Git Stash Failed Attempts**
   ```bash
   git stash push -m "Failed: javax migration - ClassNotFoundException"
   ```

2. **Update Agent Memory**
   ```python
   memory.add_failed_attempt({
       "action": "apply_jakarta_recipe",
       "error": "ClassNotFoundException",
       "context": "Custom servlet filters",
       "timestamp": "..."
   })
   ```

3. **Prevent Retry**
   - Check memory before attempting actions
   - Try alternative approaches
   - Escalate to human if no alternatives

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Set Anthropic API key
export ANTHROPIC_API_KEY="your_api_key_here"
```

### 2. Basic Migration
```python
from orchestrator import MigrationOrchestrator

# Initialize with default configuration
orchestrator = MigrationOrchestrator()

# Migrate the xsync project
result = orchestrator.migrate_repository("./xsync")

print(f"Success: {result['success']}")
print(f"Applied recipes: {len(result['applied_recipes'])}")
```

### 3. Run Example
```bash
# Run the comprehensive example
python example.py

# This will demonstrate:
# - Basic migration workflow
# - Custom configuration
# - Individual tool usage
# - Analysis-only mode
```

### 4. Custom Configuration
```python
from orchestrator import MigrationConfig

config = MigrationConfig(
    target_java_version="21",
    target_spring_boot_version="3.2",
    enable_jakarta_migration=True,
    dry_run=False,  # Set to True for testing
    backup_enabled=True
)

orchestrator = MigrationOrchestrator(config=config)
result = orchestrator.migrate_repository("/path/to/project")
```

## Success Metrics
- **Migration Success Rate**: % of projects fully migrated
- **Human Intervention Rate**: Average escalations per project
- **Time Efficiency**: Hours saved vs manual migration
- **Test Coverage**: % of tests passing post-migration
- **Code Quality**: Static analysis scores maintained/improved

## Development Roadmap

### MVP Features
- [ ] Basic Maven project analysis
- [ ] Core OpenRewrite recipe execution  
- [ ] Simple human escalation
- [ ] Git-based checkpointing

### Phase 2 Features
- [ ] Custom recipe generation
- [ ] Intelligent error recovery
- [ ] Dozer migration strategies
- [ ] Risk-based escalation

### Future Enhancements
- [ ] Multi-module project support
- [ ] Gradle support
- [ ] Learning from past migrations
- [ ] Parallel agent execution
- [ ] Migration time estimation

## Known Limitations
- Single module Maven projects only
- Requires human availability for escalations
- Custom business logic may require manual intervention
- Some legacy patterns may not have automated solutions

## Testing the System

### Test Project Structure
```
test-projects/
├── simple-web-app/     # Basic Spring Boot 2.7 app
├── jakarta-complex/    # Heavy javax usage
├── dozer-heavy/       # Extensive Dozer mappings
└── junit4-tests/      # JUnit 4 test suite
```

### Validation Criteria
1. All projects compile successfully
2. Test pass rate >= 95%
3. No deprecated API usage warnings
4. Performance benchmarks maintained

## Directory Structure

```
java-migration-agent/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── analysis_agent.py
│   │   ├── dependency_agent.py
│   │   ├── migration_agent.py
│   │   ├── testing_agent.py
│   │   └── human_interface_agent.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── memory.py
│   │   ├── state.py
│   │   └── communication.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── git_tools.py
│   │   ├── maven_tools.py
│   │   ├── openrewrite_tools.py
│   │   └── ast_tools.py
│   └── utils/
│       ├── __init__.py
│       ├── risk_assessment.py
│       └── report_generator.py
├── recipes/
│   ├── core/
│   │   └── standard-migration.yml
│   └── custom/
│       └── .gitkeep
├── tests/
│   ├── unit/
│   ├── integration/
│   └── test_projects/
└── docs/
    ├── architecture.md
    ├── agent_specifications.md
    └── human_interface_guide.md
```

## Contributing

### Agent Development Guidelines
1. Each agent should inherit from `BaseAgent` class
2. Implement standard interfaces: `analyze()`, `execute()`, `rollback()`
3. Use structured logging for all operations
4. Include comprehensive error handling
5. Write unit tests for all new functionality

### Code Style
- Follow PEP 8 for Python code
- Use type hints for all function parameters
- Document all public methods
- Keep functions focused and under 50 lines

## License
[Your chosen license]

## Support
For questions or issues:
- Create an issue in the repository
- Contact the development team
- Refer to the documentation in `/docs`