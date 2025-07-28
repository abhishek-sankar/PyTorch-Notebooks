from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class MigrationPhase(str, Enum):
    INITIALIZATION = "initialization"
    ANALYSIS = "analysis"
    JAVA_VERSION_UPGRADE = "java_version_upgrade"
    SPRING_BOOT_MIGRATION = "spring_boot_migration"
    JAKARTA_MIGRATION = "jakarta_migration"
    JUNIT_MIGRATION = "junit_migration"
    DEPENDENCY_UPDATES = "dependency_updates"
    CUSTOM_PATTERNS = "custom_patterns"
    TESTING = "testing"
    VALIDATION = "validation"
    COMPLETED = "completed"
    FAILED = "failed"


class Decision(BaseModel):
    id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    context: str
    issue: str
    options: List[Dict[str, Any]]
    chosen_option: Optional[str] = None
    rationale: Optional[str] = None
    escalation_type: str


class RecipeExecution(BaseModel):
    recipe_name: str
    status: str  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    files_modified: List[str] = []
    git_commit_hash: Optional[str] = None


class RiskAssessment(BaseModel):
    overall_risk: str  # low, medium, high, critical
    complexity_score: int = Field(ge=1, le=10)
    estimated_duration: str
    known_issues: List[str] = []
    recommendations: List[str] = []


class MigrationState(BaseModel):
    # Basic project information
    project_path: str
    project_name: str
    current_java_version: Optional[str] = None
    target_java_version: str = "21"
    
    # Migration progress
    current_phase: MigrationPhase = MigrationPhase.INITIALIZATION
    completed_steps: List[str] = []
    failed_attempts: Dict[str, List[str]] = {}  # step -> error messages
    
    # Git integration
    original_branch: Optional[str] = None
    migration_branch: Optional[str] = None
    checkpoints: List[str] = []  # git commit hashes
    
    # Recipe execution tracking
    recipe_executions: List[RecipeExecution] = []
    
    # Human interaction
    human_decisions: List[Decision] = []
    pending_decision: Optional[Decision] = None
    
    # Risk and analysis
    risk_assessment: Optional[RiskAssessment] = None
    analysis_results: Dict[str, Any] = {}
    
    # Configuration
    config: Dict[str, Any] = {}
    
    # Metadata
    session_id: str
    start_time: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    # Status tracking
    is_paused: bool = False
    pause_reason: Optional[str] = None
    error_count: int = 0
    max_error_count: int = 10

    def add_completed_step(self, step: str):
        if step not in self.completed_steps:
            self.completed_steps.append(step)
            self.last_updated = datetime.now()

    def add_failed_attempt(self, step: str, error: str):
        if step not in self.failed_attempts:
            self.failed_attempts[step] = []
        self.failed_attempts[step].append(error)
        self.error_count += 1
        self.last_updated = datetime.now()

    def add_checkpoint(self, commit_hash: str):
        self.checkpoints.append(commit_hash)
        self.last_updated = datetime.now()

    def get_last_checkpoint(self) -> Optional[str]:
        return self.checkpoints[-1] if self.checkpoints else None

    def should_escalate_to_human(self) -> bool:
        # Check if same error repeated multiple times
        for step, errors in self.failed_attempts.items():
            if len(errors) >= 3:
                return True
        
        # Check if error count is too high
        if self.error_count >= self.max_error_count:
            return True
            
        # Check if there's a pending decision
        if self.pending_decision is not None:
            return True
            
        return False

    def get_progress_percentage(self) -> float:
        total_phases = len(MigrationPhase)
        current_phase_index = list(MigrationPhase).index(self.current_phase)
        return (current_phase_index / total_phases) * 100

    class Config:
        arbitrary_types_allowed = True