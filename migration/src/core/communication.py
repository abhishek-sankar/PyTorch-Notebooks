from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import json


class MessageType(str, Enum):
    ANALYSIS_REQUEST = "analysis_request"
    ANALYSIS_RESULT = "analysis_result"
    EXECUTION_REQUEST = "execution_request"
    EXECUTION_RESULT = "execution_result"
    ERROR_REPORT = "error_report"
    ERROR_FIX_REQUEST = "error_fix_request"
    HUMAN_ESCALATION = "human_escalation"
    HUMAN_RESPONSE = "human_response"
    STATE_UPDATE = "state_update"
    CHECKPOINT_CREATED = "checkpoint_created"
    RECIPE_RECOMMENDATION = "recipe_recommendation"


class MessagePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Message(BaseModel):
    id: str = Field(default_factory=lambda: f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    type: MessageType
    sender: str  # agent name or component
    recipient: Optional[str] = None  # specific recipient or None for broadcast
    priority: MessagePriority = MessagePriority.MEDIUM
    
    # Message content
    content: str
    data: Dict[str, Any] = {}
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: str
    correlation_id: Optional[str] = None  # for tracking related messages
    
    # Response handling
    requires_response: bool = False
    response_timeout_seconds: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "priority": self.priority.value,
            "content": self.content,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "requires_response": self.requires_response,
            "response_timeout_seconds": self.response_timeout_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary"""
        data["type"] = MessageType(data["type"])
        data["priority"] = MessagePriority(data["priority"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Create message from JSON string"""
        return cls.from_dict(json.loads(json_str))


class MessageBus:
    """Simple in-memory message bus for agent communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[callable]] = {}
        self.message_history: List[Message] = []
        self.max_history_size = 1000
    
    def subscribe(self, message_type: MessageType, handler: callable, subscriber_name: str):
        """Subscribe to messages of a specific type"""
        key = f"{message_type.value}:{subscriber_name}"
        if key not in self.subscribers:
            self.subscribers[key] = []
        self.subscribers[key].append(handler)
    
    def unsubscribe(self, message_type: MessageType, subscriber_name: str):
        """Unsubscribe from messages of a specific type"""
        key = f"{message_type.value}:{subscriber_name}"
        if key in self.subscribers:
            del self.subscribers[key]
    
    def publish(self, message: Message) -> List[Any]:
        """Publish a message to subscribers"""
        self._add_to_history(message)
        
        results = []
        
        # Send to specific recipient if specified
        if message.recipient:
            key = f"{message.type.value}:{message.recipient}"
            if key in self.subscribers:
                for handler in self.subscribers[key]:
                    try:
                        result = handler(message)
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        print(f"Error in message handler {handler}: {e}")
        else:
            # Broadcast to all subscribers of this message type
            for key, handlers in self.subscribers.items():
                if key.startswith(f"{message.type.value}:"):
                    for handler in handlers:
                        try:
                            result = handler(message)
                            if result is not None:
                                results.append(result)
                        except Exception as e:
                            print(f"Error in message handler {handler}: {e}")
        
        return results
    
    def _add_to_history(self, message: Message):
        """Add message to history with size limit"""
        self.message_history.append(message)
        if len(self.message_history) > self.max_history_size:
            self.message_history = self.message_history[-self.max_history_size:]
    
    def get_message_history(
        self, 
        message_type: Optional[MessageType] = None,
        sender: Optional[str] = None,
        limit: int = 100
    ) -> List[Message]:
        """Get message history with optional filtering"""
        filtered_messages = self.message_history
        
        if message_type:
            filtered_messages = [m for m in filtered_messages if m.type == message_type]
        
        if sender:
            filtered_messages = [m for m in filtered_messages if m.sender == sender]
        
        return filtered_messages[-limit:]
    
    def clear_history(self):
        """Clear message history"""
        self.message_history.clear()


# Global message bus instance
message_bus = MessageBus()


def create_analysis_request(session_id: str, project_path: str, sender: str) -> Message:
    """Helper function to create analysis request message"""
    return Message(
        type=MessageType.ANALYSIS_REQUEST,
        sender=sender,
        recipient="analysis_agent",
        content=f"Analyze project at {project_path}",
        data={"project_path": project_path},
        session_id=session_id,
        requires_response=True,
        response_timeout_seconds=300
    )


def create_execution_request(
    session_id: str, 
    recipes: List[str], 
    project_path: str, 
    sender: str
) -> Message:
    """Helper function to create execution request message"""
    return Message(
        type=MessageType.EXECUTION_REQUEST,
        sender=sender,
        recipient="execution_agent",
        content=f"Execute recipes: {', '.join(recipes)}",
        data={"recipes": recipes, "project_path": project_path},
        session_id=session_id,
        requires_response=True,
        response_timeout_seconds=600
    )


def create_error_report(
    session_id: str, 
    error_message: str, 
    context: Dict[str, Any], 
    sender: str
) -> Message:
    """Helper function to create error report message"""
    return Message(
        type=MessageType.ERROR_REPORT,
        sender=sender,
        priority=MessagePriority.HIGH,
        content=f"Error occurred: {error_message}",
        data={"error_message": error_message, "context": context},
        session_id=session_id
    )


def create_human_escalation(
    session_id: str,
    escalation_type: str,
    context: Dict[str, Any],
    options: List[Dict[str, Any]],
    sender: str
) -> Message:
    """Helper function to create human escalation message"""
    return Message(
        type=MessageType.HUMAN_ESCALATION,
        sender=sender,
        priority=MessagePriority.CRITICAL,
        content=f"Human intervention required: {escalation_type}",
        data={
            "escalation_type": escalation_type,
            "context": context,
            "options": options
        },
        session_id=session_id,
        requires_response=True
    )