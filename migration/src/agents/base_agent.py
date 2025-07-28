from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.state import MigrationState
from ..core.memory import MigrationMemory
from ..core.communication import Message, MessageType, message_bus


class BaseAgent(ABC):
    """Base class for all migration agents"""
    
    def __init__(
        self,
        name: str,
        llm: Optional[ChatOpenAI] = None,
        memory: Optional[MigrationMemory] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.llm = llm or ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0.1,
            max_tokens=4000
        )
        self.memory = memory
        self.config = config or {}
        
        # Set up logging
        self.logger = logging.getLogger(f"migration.{name}")
        
        # Subscribe to relevant messages
        self._setup_message_subscriptions()
    
    def _setup_message_subscriptions(self):
        """Set up message subscriptions for this agent"""
        # Base implementation - agents can override this
        pass
    
    @abstractmethod
    def analyze(self, state: MigrationState) -> Dict[str, Any]:
        """Analyze the current state and return analysis results"""
        pass
    
    @abstractmethod
    def execute(self, state: MigrationState, **kwargs) -> Dict[str, Any]:
        """Execute the agent's primary function"""
        pass
    
    def rollback(self, state: MigrationState, checkpoint: str) -> bool:
        """Rollback to a previous checkpoint"""
        try:
            # Base implementation - agents can override for specific rollback logic
            self.logger.info(f"Rolling back to checkpoint: {checkpoint}")
            return True
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
    
    def _call_llm(
        self, 
        prompt: str, 
        system_message: str = "", 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Call the LLM with the given prompt"""
        messages = []
        
        if system_message:
            messages.append(SystemMessage(content=system_message))
        
        # Add context if provided
        if context:
            context_str = f"Context: {context}\n\n"
            prompt = context_str + prompt
        
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise
    
    def _send_message(self, message: Message) -> List[Any]:
        """Send a message via the message bus"""
        return message_bus.publish(message)
    
    def _log_action(self, action: str, details: Dict[str, Any]):
        """Log an action with structured data"""
        self.logger.info(f"Action: {action}", extra={
            "agent": self.name,
            "action": action,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
    
    def _log_error(self, error: str, context: Dict[str, Any]):
        """Log an error with context"""
        self.logger.error(f"Error: {error}", extra={
            "agent": self.name,
            "error": error,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
    
    def _handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors with standard format"""
        error_data = {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "agent": self.name
        }
        
        self._log_error(str(error), context)
        
        # Add to memory if available
        if self.memory:
            self.memory.add_failed_attempt(
                step=context.get("step", "unknown"),
                error=str(error),
                context=str(context)
            )
        
        return error_data
    
    def _create_success_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a standard success response"""
        return {
            "success": True,
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            **data
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current status of the agent"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "status": "active",
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }


class AnalysisCapable(ABC):
    """Mixin for agents that can perform analysis"""
    
    @abstractmethod
    def perform_analysis(self, target: str, analysis_type: str) -> Dict[str, Any]:
        """Perform specific type of analysis"""
        pass


class ExecutionCapable(ABC):
    """Mixin for agents that can execute commands/recipes"""
    
    @abstractmethod
    def execute_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """Execute a command or recipe"""
        pass


class ErrorHandlingCapable(ABC):
    """Mixin for agents that can handle and fix errors"""
    
    @abstractmethod
    def diagnose_error(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose an error and suggest fixes"""
        pass
    
    @abstractmethod
    def apply_fix(self, fix_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a fix strategy"""
        pass