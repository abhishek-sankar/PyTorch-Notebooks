# Copy the content from the setup guide artifact here
# Due to size, you'll need to copy this from the Claude interface
# Java 11 to 21 Migration Pipeline - Setup Guide

## Overview

This pipeline automates the migration of Java applications from:
- Java 11 → Java 21
- Spring Boot 2.x → Spring Boot 3.x
- JUnit 4 → JUnit 5
- javax.* → jakarta.*

The pipeline uses a hybrid approach:
1. **OpenRewrite** for deterministic, rule-based transformations
2. **AI Agents** (LangChain + Claude/GPT-4) for complex, context-specific fixes

## Prerequisites

### System Requirements
- Python 3.9+
- Java 11 (for running the initial project)
- Java 21 (target version)
- Git
- Maven or Gradle
- 16GB+ RAM recommended
- Unix-like environment (Linux/macOS) or WSL2 on Windows

### Required Tools
```bash
# Check Python version
python --version  # Should be 3.9+

# Check Java versions
java -version     # Current project version
/usr/lib/jvm/java-21/bin/java -version  # Target version

# Check build tools
mvn --version     # For Maven projects
gradle --version  # For Gradle projects

# Optional but recommended
gh --version      # GitHub CLI for PR creation
rg --version      # ripgrep for faster code search
```

## Installation

### 1. Clone the Pipeline Repository
```bash
git clone https://github.com/your-org/java-migration-pipeline.git
cd java-migration-pipeline
```

### 2. Create Python Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys
```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key-here"

# For OpenAI GPT-4
export OPENAI_API_KEY="your-api-key-here"

# Save to .env file for persistence
echo "ANTHROPIC_API_KEY=your-api-key-here" >> .env
echo "OPENAI_API_KEY=your-api-key-here" >> .env
```

### 5. Configure OpenRewrite in Your Project

#### For Maven Projects
Add to your `pom.xml`:

```xml
<build>
  <plugins>
    <plugin>
      <groupId>org.openrewrite.maven</groupId>
      <artifactId>rewrite-maven-plugin</artifactId>
      <version>5.34.1</version>
      <configuration>
        <activeRecipes>
          <recipe>org.openrewrite.java.migrate.Java8toJava11</recipe>
        </activeRecipes>
      </configuration>
      <dependencies>
        <dependency>
          <groupId>org.openrewrite.recipe</groupId>
          <artifactId>rewrite-migrate-java</artifactId>
          <version>2.0.0</version>
        </dependency>
        <dependency>
          <groupId>org.openrewrite.recipe</groupId>
          <artifactId>rewrite-spring</artifactId>
          <version>5.0.0</version>
        </dependency>
        <dependency>
          <groupId>org.openrewrite.recipe</groupId>
          <artifactId>rewrite-testing-frameworks</artifactId>
          <version>2.0.0</version>
        </dependency>
      </dependencies>
    </plugin>
  </plugins>
</build>
```

#### For Gradle Projects
Add to your `build.gradle`:

```gradle
plugins {
    id 'org.openrewrite.rewrite' version '6.0.0'
}

rewrite {
    activeRecipe('org.openrewrite.java.migrate.Java8toJava11')
}

dependencies {
    rewrite 'org.openrewrite.recipe:rewrite-migrate-java:2.0.0'
    rewrite 'org.openrewrite.recipe:rewrite-spring:5.0.0'
    rewrite 'org.openrewrite.recipe:rewrite-testing-frameworks:2.0.0'
}
```

## Configuration

### 1. Create Migration Configuration
Create a `migration-config.yaml` file:

```yaml
# Project settings
project_path: /absolute/path/to/your/java/project
project_name: MyJavaApplication
build_tool: maven  # or gradle

# OpenRewrite settings
openrewrite_version: 2.0.0
skip_recipes: []  # Add recipes to skip if needed
custom_recipes: []  # Add custom recipe names

# AI Agent settings
llm_provider: anthropic  # or openai
llm_model: claude-3-5-sonnet-20241022  # or gpt-4-turbo
max_repair_attempts: 100
repair_timeout_minutes: 120

# Pipeline settings
commit_strategy: atomic  # or batch
create_pull_request: true
branch_name: migration/java21-spring3

# Quality settings
run_tests_after_each_fix: false  # Set to true for safer but slower migration
required_test_coverage: 80.0
static_analysis_enabled: true
```

### 2. Prepare Your Project

#### Ensure Clean Git State
```bash
cd /path/to/your/java/project
git status  # Should show clean working directory
git checkout -b pre-migration-backup  # Create backup branch
git checkout main  # Return to main branch
```

#### Run Initial Build
```bash
# Maven
mvn clean install

# Gradle
./gradlew clean build
```

The build must pass before starting migration!

#### Check Test Coverage (Optional but Recommended)
```bash
# Maven with JaCoCo
mvn clean test jacoco:report
# Open target/site/jacoco/index.html

# Gradle with JaCoCo
./gradlew clean test jacocoTestReport
# Open build/reports/jacoco/test/html/index.html
```

## Running the Migration

### 1. Dry Run (Recommended First Step)
```bash
python migration_orchestrator.py --config migration-config.yaml --dry-run
```

This will:
- Run OpenRewrite recipes in dry-run mode
- Show what changes would be made
- Generate reports without modifying code

### 2. Full Migration
```bash
python migration_orchestrator.py --config migration-config.yaml --verbose
```

### 3. Migration with Options
```bash
# Skip test fixing phase (faster but less complete)
python migration_orchestrator.py --config migration-config.yaml --skip-tests

# Limit number of AI repair attempts
python migration_orchestrator.py --config migration-config.yaml --max-errors 50

# Enable debug logging
python migration_orchestrator.py --config migration-config.yaml --verbose
```

## Monitoring Progress

### Real-time Logs
The pipeline creates detailed logs in:
```
your-project/migration-logs/migration_YYYYMMDD_HHMMSS.log
```

### Metrics Dashboard
Monitor progress via the generated metrics:
```
your-project/migration-report.json
your-project/migration-report_errors.csv
your-project/migration-report_fixes.csv
```

### Git History
Track changes via atomic commits:
```bash
git log --oneline --grep="AI-FIX\|OpenRewrite"
```

## Troubleshooting

### Common Issues

#### 1. OpenRewrite Recipe Failures
```
Error: Recipe org.openrewrite.java.spring.boot3.UpgradeSpringBoot_3_3 not found
```
**Solution**: Update OpenRewrite dependencies in pom.xml/build.gradle

#### 2. AI Agent Timeout
```
Error: AI repair timeout after 120 minutes
```
**Solution**: Increase `repair_timeout_minutes` in config or fix errors manually

#### 3. Compilation Loops
```
Warning: Same error appearing repeatedly
```
**Solution**: The AI might be stuck. Check the error and provide manual fix, then restart

#### 4. API Rate Limits
```
Error: Rate limit exceeded for API
```
**Solution**: Add delays or use a different API key

### Manual Intervention

When the pipeline requires manual intervention:

1. Note the last successful commit:
   ```bash
   git log -1 --oneline
   ```

2. Fix the blocking issue manually

3. Commit your fix:
   ```bash
   git add -A
   git commit -m "MANUAL-FIX: Description of what you fixed"
   ```

4. Resume the pipeline:
   ```bash
   python migration_orchestrator.py --config migration-config.yaml --resume
   ```

## Best Practices

### Before Migration
1. **Ensure High Test Coverage**: Aim for 80%+ coverage on critical paths
2. **Update Dependencies**: Update to latest minor versions first
3. **Clean Code**: Run static analysis and fix major issues
4. **Document Customizations**: List any custom Spring configurations or Java hacks

### During Migration
1. **Monitor Resource Usage**: The AI agent can be memory-intensive
2. **Review Commits Regularly**: Don't let too many changes accumulate
3. **Test Incrementally**: Run smoke tests after major phases
4. **Keep Notes**: Document any manual interventions needed

### After Migration
1. **Comprehensive Testing**: Run full test suite including integration tests
2. **Performance Testing**: Compare before/after performance metrics
3. **Security Scan**: Run dependency and code security scans
4. **Update Documentation**: Reflect new Java 21 features and Spring Boot 3 changes

## Customization

### Adding Custom OpenRewrite Recipes

1. Create a custom recipe in `rewrite.yml`:
```yaml
type: specs.openrewrite.org/v1beta/recipe
name: com.yourorg.CustomMigration
displayName: Custom Migration Recipe
description: Custom fixes for your application
recipeList:
  - org.openrewrite.java.ChangeMethodName:
      methodPattern: com.yourorg.OldClass oldMethod()
      newMethodName: newMethod
```

2. Add to migration config:
```yaml
custom_recipes:
  - com.yourorg.CustomMigration
```

### Extending AI Agent Tools

Add custom tools in `langchain_repair_agent.py`:

```python
def tool_check_database_compatibility(self, entity_class: str) -> str:
    """Check if JPA entity is compatible with Jakarta Persistence"""
    # Your custom logic here
    return "Compatibility check result"

# Add to _create_tools() method
Tool(
    name="check_database_compatibility",
    description="Check JPA entity compatibility",
    func=self.tool_check_database_compatibility
)
```

## Support and Contribution

### Getting Help
- Check the [FAQ](docs/FAQ.md)
- Search existing [issues](https://github.com/your-org/java-migration-pipeline/issues)
- Join our [Discord](https://discord.gg/your-channel)

### Contributing
- Fork the repository
- Create a feature branch
- Submit a pull request with tests

### License
This project is licensed under the Apache 2.0 License.