# Copy the content from the third artifact here
# Due to size, you'll need to copy this from the Claude interface
"""
Main Orchestrator for Java Migration Pipeline
Combines OpenRewrite and AI agents for complete migration
"""

import yaml
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import click
import subprocess
from datetime import datetime
import pandas as pd

# Import our modules
from java_migration_pipeline import (
    MigrationPipeline, 
    OpenRewriteOrchestrator,
    ErrorParser,
    CompilationError
)
from langchain_repair_agent import (
    JavaMigrationAgent,
    LangChainMigrationPipeline
)

# ===== Configuration =====

@dataclass
class MigrationConfig:
    """Configuration for the migration pipeline"""
    # Project settings
    project_path: str
    project_name: str
    build_tool: str = "maven"  # or "gradle"
    
    # OpenRewrite settings
    openrewrite_version: str = "2.0.0"
    skip_recipes: List[str] = None
    custom_recipes: List[str] = None
    
    # AI Agent settings
    llm_provider: str = "anthropic"  # or "openai"
    llm_model: str = "claude-3-5-sonnet-20241022"
    max_repair_attempts: int = 100
    repair_timeout_minutes: int = 120
    
    # Pipeline settings
    commit_strategy: str = "atomic"  # or "batch"
    create_pull_request: bool = True
    branch_name: str = "migration/java21-spring3"
    
    # Quality settings
    run_tests_after_each_fix: bool = False
    required_test_coverage: float = 80.0
    static_analysis_enabled: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'MigrationConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, output_path: str):
        """Save configuration to YAML file"""
        with open(output_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

# ===== Metrics Tracking =====

class MigrationMetrics:
    """Track and report migration metrics"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.metrics = {
            "openrewrite_files_modified": 0,
            "compilation_errors_found": 0,
            "compilation_errors_fixed": 0,
            "test_failures_found": 0,
            "test_failures_fixed": 0,
            "ai_repair_attempts": 0,
            "human_interventions_required": 0,
            "total_commits": 0
        }
        self.error_log = []
        self.fix_log = []
    
    def record_error(self, error_type: str, details: Dict):
        """Record an error occurrence"""
        self.error_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "details": details
        })
        
        if error_type == "compilation":
            self.metrics["compilation_errors_found"] += 1
        elif error_type == "test":
            self.metrics["test_failures_found"] += 1
    
    def record_fix(self, fix_type: str, details: Dict):
        """Record a successful fix"""
        self.fix_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": fix_type,
            "details": details
        })
        
        if fix_type == "compilation":
            self.metrics["compilation_errors_fixed"] += 1
        elif fix_type == "test":
            self.metrics["test_failures_fixed"] += 1
        
        self.metrics["ai_repair_attempts"] += 1
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive metrics report"""
        duration = (datetime.now() - self.start_time).total_seconds() / 60
        
        return {
            "summary": {
                "duration_minutes": round(duration, 2),
                "success_rate": self._calculate_success_rate(),
                "automation_rate": self._calculate_automation_rate()
            },
            "metrics": self.metrics,
            "error_distribution": self._analyze_error_distribution(),
            "performance": {
                "avg_fix_time": self._calculate_avg_fix_time(),
                "errors_per_hour": self._calculate_error_rate()
            }
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        total_errors = self.metrics["compilation_errors_found"] + self.metrics["test_failures_found"]
        total_fixed = self.metrics["compilation_errors_fixed"] + self.metrics["test_failures_fixed"]
        
        if total_errors == 0:
            return 100.0
        
        return round((total_fixed / total_errors) * 100, 2)
    
    def _calculate_automation_rate(self) -> float:
        """Calculate automation vs manual intervention rate"""
        total_actions = self.metrics["ai_repair_attempts"] + self.metrics["human_interventions_required"]
        
        if total_actions == 0:
            return 100.0
        
        return round((self.metrics["ai_repair_attempts"] / total_actions) * 100, 2)
    
    def _analyze_error_distribution(self) -> Dict[str, int]:
        """Analyze distribution of error types"""
        distribution = {}
        for error in self.error_log:
            error_type = error["details"].get("error_type", "unknown")
            distribution[error_type] = distribution.get(error_type, 0) + 1
        return distribution
    
    def _calculate_avg_fix_time(self) -> float:
        """Calculate average time to fix an error"""
        if not self.fix_log:
            return 0.0
        
        # Simplified - in reality would track time per fix
        duration = (datetime.now() - self.start_time).total_seconds() / 60
        return round(duration / len(self.fix_log), 2)
    
    def _calculate_error_rate(self) -> float:
        """Calculate errors fixed per hour"""
        duration_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        if duration_hours == 0:
            return 0.0
        
        total_fixed = self.metrics["compilation_errors_fixed"] + self.metrics["test_failures_fixed"]
        return round(total_fixed / duration_hours, 2)
    
    def save_report(self, output_path: str):
        """Save detailed report to file"""
        report = self.generate_report()
        
        # Save JSON report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save error log as CSV
        if self.error_log:
            df = pd.DataFrame(self.error_log)
            df.to_csv(output_path.replace('.json', '_errors.csv'), index=False)
        
        # Save fix log as CSV
        if self.fix_log:
            df = pd.DataFrame(self.fix_log)
            df.to_csv(output_path.replace('.json', '_fixes.csv'), index=False)

# ===== Main Orchestrator =====

class MigrationOrchestrator:
    """Main orchestrator that coordinates the entire migration process"""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.project_path = Path(config.project_path)
        self.metrics = MigrationMetrics()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.openrewrite = OpenRewriteOrchestrator(
            config.project_path,
            config.build_tool
        )
        self.ai_pipeline = LangChainMigrationPipeline(
            config.project_path,
            config.llm_provider
        )
        self.error_parser = ErrorParser()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_dir = self.project_path / "migration-logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger("MigrationOrchestrator")
    
    def run(self) -> Dict[str, Any]:
        """Execute the complete migration pipeline"""
        self.logger.info(f"Starting migration for project: {self.config.project_name}")
        
        try:
            # Step 1: Pre-flight checks
            if not self._preflight_checks():
                return {"success": False, "reason": "Pre-flight checks failed"}
            
            # Step 2: Create migration branch
            self._create_migration_branch()
            
            # Step 3: Run OpenRewrite phases
            self.logger.info("Phase 1: Running OpenRewrite transformations")
            openrewrite_success = self._run_openrewrite_phases()
            
            if not openrewrite_success:
                return {"success": False, "reason": "OpenRewrite phase failed"}
            
            # Step 4: AI-assisted compilation fixes
            self.logger.info("Phase 2: AI-assisted compilation fixes")
            compilation_success = self._run_compilation_fixes()
            
            if not compilation_success:
                return {"success": False, "reason": "Failed to achieve compilation"}
            
            # Step 5: AI-assisted test fixes
            self.logger.info("Phase 3: AI-assisted test fixes")
            test_success = self._run_test_fixes()
            
            # Step 6: Quality checks
            self.logger.info("Phase 4: Running quality checks")
            quality_passed = self._run_quality_checks()
            
            # Step 7: Create pull request
            if self.config.create_pull_request:
                pr_url = self._create_pull_request()
                self.logger.info(f"Pull request created: {pr_url}")
            
            # Generate final report
            report = self.metrics.generate_report()
            self.metrics.save_report(
                str(self.project_path / "migration-report.json")
            )
            
            return {
                "success": compilation_success and test_success,
                "metrics": report,
                "pull_request": pr_url if self.config.create_pull_request else None
            }
            
        except Exception as e:
            self.logger.error(f"Migration failed with error: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def _preflight_checks(self) -> bool:
        """Run pre-flight checks before starting migration"""
        checks = []
        
        # Check if project exists
        if not self.project_path.exists():
            self.logger.error(f"Project path does not exist: {self.project_path}")
            return False
        
        # Check build tool
        if self.config.build_tool == "maven":
            if not (self.project_path / "pom.xml").exists():
                self.logger.error("No pom.xml found for Maven project")
                return False
        elif self.config.build_tool == "gradle":
            if not (self.project_path / "build.gradle").exists():
                self.logger.error("No build.gradle found for Gradle project")
                return False
        
        # Check git status
        git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.project_path,
            capture_output=True,
            text=True
        )
        
        if git_status.stdout.strip():
            self.logger.warning("Working directory has uncommitted changes")
            # Optionally fail here or stash changes
        
        # Check initial build
        self.logger.info("Running initial build to verify project state")
        build_result = self._build_project(skip_tests=False)
        
        if not build_result["success"]:
            self.logger.error("Initial build failed. Project must build successfully before migration.")
            return False
        
        # Check test coverage if required
        if self.config.required_test_coverage > 0:
            coverage = self._check_test_coverage()
            if coverage < self.config.required_test_coverage:
                self.logger.warning(f"Test coverage ({coverage}%) is below required threshold ({self.config.required_test_coverage}%)")
                # Optionally fail here
        
        self.logger.info("Pre-flight checks passed")
        return True
    
    def _create_migration_branch(self):
        """Create and checkout migration branch"""
        branch_name = self.config.branch_name
        
        # Create new branch from current HEAD
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=self.project_path,
            check=True
        )
        
        self.logger.info(f"Created migration branch: {branch_name}")
    
    def _run_openrewrite_phases(self) -> bool:
        """Execute OpenRewrite phases sequentially"""
        recipes = self.openrewrite.recipes
        
        # Filter out skipped recipes
        if self.config.skip_recipes:
            recipes = [r for r in recipes if r["recipe"] not in self.config.skip_recipes]
        
        # Add custom recipes
        if self.config.custom_recipes:
            for custom_recipe in self.config.custom_recipes:
                recipes.append({
                    "phase": f"custom_{len(recipes)+1}",
                    "recipe": custom_recipe,
                    "description": f"Custom recipe: {custom_recipe}"
                })
        
        for i, recipe in enumerate(recipes):
            self.logger.info(f"Running OpenRewrite recipe {i+1}/{len(recipes)}: {recipe['description']}")
            
            # Run the recipe
            result = self.openrewrite.run_phase(i, dry_run=False)
            
            if not result["success"]:
                self.logger.error(f"Recipe failed: {recipe['recipe']}")
                self.logger.error(result["stderr"])
                return False
            
            # Analyze what changed
            changed_files = self._get_changed_files()
            self.metrics.metrics["openrewrite_files_modified"] += len(changed_files)
            
            # Commit changes
            if changed_files:
                self._commit_changes(f"OpenRewrite: {recipe['description']}")
                self.metrics.metrics["total_commits"] += 1
            
            self.logger.info(f"Recipe completed. Modified {len(changed_files)} files")
        
        return True
    
    def _run_compilation_fixes(self) -> bool:
        """Run AI-assisted compilation fixes"""
        max_attempts = self.config.max_repair_attempts
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            
            # Try to build
            self.logger.info(f"Compilation attempt {attempt}/{max_attempts}")
            build_result = self._build_project(skip_tests=True)
            
            if build_result["success"]:
                self.logger.info("✓ Compilation successful!")
                return True
            
            # Parse compilation errors
            errors = self.error_parser.parse_compilation_errors(build_result["output"])
            
            if not errors:
                self.logger.error("Build failed but no compilation errors found")
                self.metrics.metrics["human_interventions_required"] += 1
                return False
            
            self.logger.info(f"Found {len(errors)} compilation errors")
            
            # Record errors
            for error in errors:
                self.metrics.record_error("compilation", asdict(error))
            
            # Fix first error using AI agent
            error = errors[0]
            self.logger.info(f"Attempting to fix: {error.error_type.value} in {error.file_path}:{error.line_number}")
            
            # Convert to dict for the agent
            error_dict = asdict(error)
            error_dict["error_type"] = error.error_type.value
            
            fix_result = self.ai_pipeline.agent.fix_compilation_error(error_dict)
            
            if fix_result["success"]:
                self.logger.info("✓ Fix applied successfully")
                self.metrics.record_fix("compilation", {
                    "error": error_dict,
                    "fix_output": fix_result["output"]
                })
                
                # Commit the fix
                if self.config.commit_strategy == "atomic":
                    commit_msg = f"AI-FIX: {error.error_type.value} in {error.file_path}:{error.line_number}"
                    self._commit_changes(commit_msg)
                    self.metrics.metrics["total_commits"] += 1
            else:
                self.logger.warning(f"✗ Failed to fix error: {fix_result.get('error', 'Unknown')}")
                self.metrics.metrics["human_interventions_required"] += 1
                
                # Continue to next error or bail out based on strategy
                if attempt > 10:  # Give up after 10 consecutive failures
                    self.logger.error("Too many consecutive failures. Manual intervention required.")
                    return False
        
        self.logger.error(f"Max repair attempts ({max_attempts}) reached")
        return False
    
    def _run_test_fixes(self) -> bool:
        """Run AI-assisted test fixes"""
        max_attempts = self.config.max_repair_attempts
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            
            # Run tests
            self.logger.info(f"Test run attempt {attempt}/{max_attempts}")
            test_result = self._build_project(skip_tests=False)
            
            if test_result["success"]:
                self.logger.info("✓ All tests passing!")
                return True
            
            # Parse test failures
            failures = self.error_parser.parse_test_failures(test_result["output"])
            
            if not failures:
                self.logger.error("Tests failed but no failures parsed")
                # Try to extract some info from the output
                if "BUILD FAILURE" in test_result["output"]:
                    self.logger.info("Attempting to parse build failure output")
                    # Additional parsing logic here
                self.metrics.metrics["human_interventions_required"] += 1
                return False
            
            self.logger.info(f"Found {len(failures)} test failures")
            
            # Record failures
            for failure in failures:
                self.metrics.record_error("test", asdict(failure))
            
            # Fix first failure
            failure = failures[0]
            self.logger.info(f"Attempting to fix test: {failure.test_class}::{failure.test_method}")
            
            fix_result = self.ai_pipeline.agent.fix_test_failure(asdict(failure))
            
            if fix_result["success"]:
                self.logger.info("✓ Test fix applied successfully")
                self.metrics.record_fix("test", {
                    "failure": asdict(failure),
                    "fix_output": fix_result["output"]
                })
                
                # Commit the fix
                if self.config.commit_strategy == "atomic":
                    commit_msg = f"AI-FIX: Test {failure.test_class}::{failure.test_method}"
                    self._commit_changes(commit_msg)
                    self.metrics.metrics["total_commits"] += 1
                    
                # Optionally run just this test to verify
                if self.config.run_tests_after_each_fix:
                    single_test_result = self._run_single_test(failure.test_class, failure.test_method)
                    if not single_test_result["success"]:
                        self.logger.warning("Fixed test still failing, will retry")
            else:
                self.logger.warning(f"✗ Failed to fix test: {fix_result.get('error', 'Unknown')}")
                self.metrics.metrics["human_interventions_required"] += 1
        
        self.logger.error(f"Max test repair attempts ({max_attempts}) reached")
        return False
    
    def _run_quality_checks(self) -> bool:
        """Run static analysis and other quality checks"""
        if not self.config.static_analysis_enabled:
            return True
        
        self.logger.info("Running quality checks")
        
        # Run SpotBugs/PMD/Checkstyle
        if self.config.build_tool == "maven":
            checks = [
                ("SpotBugs", ["mvn", "spotbugs:check"]),
                ("PMD", ["mvn", "pmd:check"]),
                ("Checkstyle", ["mvn", "checkstyle:check"])
            ]
        else:
            checks = [
                ("SpotBugs", ["./gradlew", "spotbugsMain"]),
                ("PMD", ["./gradlew", "pmdMain"]),
                ("Checkstyle", ["./gradlew", "checkstyleMain"])
            ]
        
        all_passed = True
        for check_name, cmd in checks:
            result = subprocess.run(cmd, cwd=self.project_path, capture_output=True)
            if result.returncode != 0:
                self.logger.warning(f"{check_name} found issues")
                all_passed = False
            else:
                self.logger.info(f"✓ {check_name} passed")
        
        return all_passed
    
    def _create_pull_request(self) -> str:
        """Create a pull request for the migration"""
        # Push the branch
        subprocess.run(
            ["git", "push", "-u", "origin", self.config.branch_name],
            cwd=self.project_path,
            check=True
        )
        
        # Create PR description
        report = self.metrics.generate_report()
        pr_description = f"""## Java 11 to 21 Migration

This pull request contains the automated migration from:
- Java 11 → Java 21
- Spring Boot 2.x → Spring Boot 3.x
- JUnit 4 → JUnit 5
- javax.* → jakarta.*

### Migration Statistics
- **Duration**: {report['summary']['duration_minutes']} minutes
- **Success Rate**: {report['summary']['success_rate']}%
- **Automation Rate**: {report['summary']['automation_rate']}%
- **Files Modified**: {self.metrics.metrics['openrewrite_files_modified']}
- **Compilation Errors Fixed**: {self.metrics.metrics['compilation_errors_fixed']}
- **Test Failures Fixed**: {self.metrics.metrics['test_failures_fixed']}
- **Total Commits**: {self.metrics.metrics['total_commits']}

### Error Distribution
{json.dumps(report['error_distribution'], indent=2)}

### Review Checklist
- [ ] All tests pass
- [ ] No compilation warnings
- [ ] Static analysis passes
- [ ] Performance benchmarks acceptable
- [ ] Security scan clean
- [ ] Documentation updated

### Next Steps
1. Review the changes carefully
2. Run integration tests
3. Deploy to staging environment
4. Monitor for any runtime issues
"""
        
        # Using GitHub CLI if available
        try:
            result = subprocess.run(
                [
                    "gh", "pr", "create",
                    "--title", f"Migration: {self.config.project_name} to Java 21 & Spring Boot 3",
                    "--body", pr_description,
                    "--base", "main",
                    "--head", self.config.branch_name
                ],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            self.logger.warning("GitHub CLI not available or PR creation failed")
            return "Manual PR creation required"
    
    def _build_project(self, skip_tests: bool = True) -> Dict[str, Any]:
        """Build the project"""
        if self.config.build_tool == "maven":
            cmd = ["mvn", "clean", "install"]
            if skip_tests:
                cmd.append("-DskipTests")
        else:
            cmd = ["./gradlew", "clean", "build"]
            if skip_tests:
                cmd.extend(["-x", "test"])
        
        result = subprocess.run(
            cmd,
            cwd=self.project_path,
            capture_output=True,
            text=True
        )
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout + "\n" + result.stderr,
            "return_code": result.returncode
        }
    
    def _run_single_test(self, test_class: str, test_method: str) -> Dict[str, Any]:
        """Run a single test"""
        if self.config.build_tool == "maven":
            cmd = ["mvn", "test", f"-Dtest={test_class}#{test_method}"]
        else:
            cmd = ["./gradlew", "test", f"--tests", f"{test_class}.{test_method}"]
        
        result = subprocess.run(
            cmd,
            cwd=self.project_path,
            capture_output=True,
            text=True
        )
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout + "\n" + result.stderr
        }
    
    def _check_test_coverage(self) -> float:
        """Check test coverage percentage"""
        if self.config.build_tool == "maven":
            # Run JaCoCo
            subprocess.run(
                ["mvn", "clean", "test", "jacoco:report"],
                cwd=self.project_path,
                check=True
            )
            
            # Parse the coverage report (simplified)
            report_path = self.project_path / "target/site/jacoco/index.html"
            if report_path.exists():
                # Extract coverage percentage from HTML
                # This is simplified - use proper XML parsing in production
                with open(report_path, 'r') as f:
                    content = f.read()
                    # Look for coverage percentage
                    import re
                    match = re.search(r'Total.*?(\d+)%', content)
                    if match:
                        return float(match.group(1))
        
        return 0.0
    
    def _get_changed_files(self) -> List[str]:
        """Get list of files changed since last commit"""
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=self.project_path,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return [f for f in result.stdout.strip().split('\n') if f]
        return []
    
    def _commit_changes(self, message: str):
        """Commit current changes"""
        subprocess.run(
            ["git", "add", "-A"],
            cwd=self.project_path,
            check=True
        )
        
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self.project_path,
            check=True
        )

# ===== CLI Interface =====

@click.command()
@click.option('--config', '-c', required=True, help='Path to configuration YAML file')
@click.option('--dry-run', is_flag=True, help='Run in dry-run mode (OpenRewrite only)')
@click.option('--skip-tests', is_flag=True, help='Skip test fixing phase')
@click.option('--max-errors', type=int, default=100, help='Maximum errors to fix')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(config: str, dry_run: bool, skip_tests: bool, max_errors: int, verbose: bool):
    """Java Migration Pipeline - Automated migration to Java 21 & Spring Boot 3"""
    
    # Load configuration
    try:
        migration_config = MigrationConfig.from_yaml(config)
        migration_config.max_repair_attempts = max_errors
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        return 1
    
    # Set up logging
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create orchestrator
    orchestrator = MigrationOrchestrator(migration_config)
    
    # Run migration
    click.echo(f"Starting migration for: {migration_config.project_name}")
    click.echo(f"Project path: {migration_config.project_path}")
    click.echo(f"LLM: {migration_config.llm_provider}/{migration_config.llm_model}")
    
    if dry_run:
        click.echo("Running in DRY-RUN mode - no changes will be made")
        # Just run OpenRewrite analysis
        for i in range(len(orchestrator.openrewrite.recipes)):
            result = orchestrator.openrewrite.run_phase(i, dry_run=True)
            click.echo(f"Recipe {i+1}: {'✓' if result['success'] else '✗'}")
        return 0
    
    # Run full migration
    result = orchestrator.run()
    
    # Display results
    if result["success"]:
        click.echo("\n✓ Migration completed successfully!")
        metrics = result["metrics"]
        click.echo(f"\nSummary:")
        click.echo(f"  Duration: {metrics['summary']['duration_minutes']} minutes")
        click.echo(f"  Success Rate: {metrics['summary']['success_rate']}%")
        click.echo(f"  Files Modified: {metrics['metrics']['openrewrite_files_modified']}")
        click.echo(f"  Errors Fixed: {metrics['metrics']['compilation_errors_fixed'] + metrics['metrics']['test_failures_fixed']}")
        
        if result.get("pull_request"):
            click.echo(f"\nPull Request: {result['pull_request']}")
    else:
        click.echo("\n✗ Migration failed!", err=True)
        click.echo(f"Reason: {result.get('reason', 'Unknown')}", err=True)
        return 1
    
    return 0

if __name__ == "__main__":
    main()

# ===== Example Configuration File =====
"""
# migration-config.yaml
project_path: /path/to/your/project
project_name: MyJavaApplication
build_tool: maven

# OpenRewrite settings
openrewrite_version: 2.0.0
skip_recipes: []
custom_recipes: []

# AI Agent settings  
llm_provider: anthropic
llm_model: claude-3-5-sonnet-20241022
max_repair_attempts: 100
repair_timeout_minutes: 120

# Pipeline settings
commit_strategy: atomic
create_pull_request: true
branch_name: migration/java21-spring3

# Quality settings
run_tests_after_each_fix: false
required_test_coverage: 80.0
static_analysis_enabled: true
"""