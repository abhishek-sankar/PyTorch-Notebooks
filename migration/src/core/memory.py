import json
import sqlite3
from typing import Dict, List, Optional, Any
from datetime import datetime
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import redis
import os
from .state import MigrationState


class MigrationMemory:
    def __init__(
        self, 
        session_id: str,
        redis_url: str = "redis://localhost:6379",
        sqlite_path: str = "./migration_state.db",
        max_token_limit: int = 200000
    ):
        self.session_id = session_id
        self.redis_url = redis_url
        self.sqlite_path = sqlite_path
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            self.redis_available = True
        except (redis.ConnectionError, redis.TimeoutError):
            print(f"Warning: Redis not available at {redis_url}. Using in-memory chat history.")
            self.redis_available = False
            self.redis_client = None
        
        # Initialize LangChain memory
        if self.redis_available:
            message_history = RedisChatMessageHistory(
                url=redis_url,
                ttl=86400,  # 24 hour TTL
                session_id=session_id
            )
        else:
            # Fallback to in-memory history
            from langchain.memory.chat_message_histories import ChatMessageHistory
            message_history = ChatMessageHistory()
        
        self.conversation_memory = ConversationSummaryBufferMemory(
            chat_memory=message_history,
            max_token_limit=max_token_limit,
            return_messages=True
        )
        
        # Initialize SQLite for state persistence
        self._init_sqlite()
    
    def _init_sqlite(self):
        """Initialize SQLite database for state persistence"""
        self.sqlite_conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        cursor = self.sqlite_conn.cursor()
        
        # Create states table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS migration_states (
                session_id TEXT PRIMARY KEY,
                state_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create decisions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS human_decisions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                decision_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES migration_states (session_id)
            )
        ''')
        
        # Create failed attempts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failed_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                step TEXT NOT NULL,
                error_message TEXT NOT NULL,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES migration_states (session_id)
            )
        ''')
        
        self.sqlite_conn.commit()
    
    def save_state(self, state: MigrationState):
        """Save migration state to SQLite"""
        cursor = self.sqlite_conn.cursor()
        state_json = state.json()
        
        cursor.execute('''
            INSERT OR REPLACE INTO migration_states (session_id, state_data, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (self.session_id, state_json))
        
        self.sqlite_conn.commit()
    
    def load_state(self) -> Optional[MigrationState]:
        """Load migration state from SQLite"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            'SELECT state_data FROM migration_states WHERE session_id = ?',
            (self.session_id,)
        )
        
        result = cursor.fetchone()
        if result:
            state_data = json.loads(result[0])
            return MigrationState(**state_data)
        return None
    
    def add_message(self, message: str, is_human: bool = True):
        """Add a message to conversation memory"""
        if is_human:
            msg = HumanMessage(content=message)
        else:
            msg = AIMessage(content=message)
        
        self.conversation_memory.chat_memory.add_message(msg)
    
    def add_failed_attempt(self, step: str, error: str, context: str = ""):
        """Log a failed attempt"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
            INSERT INTO failed_attempts (session_id, step, error_message, context)
            VALUES (?, ?, ?, ?)
        ''', (self.session_id, step, error, context))
        
        self.sqlite_conn.commit()
        
        # Also add to conversation memory
        self.add_message(
            f"Failed attempt - Step: {step}, Error: {error}, Context: {context}",
            is_human=False
        )
    
    def get_failed_attempts(self, step: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get failed attempts, optionally filtered by step"""
        cursor = self.sqlite_conn.cursor()
        
        if step:
            cursor.execute('''
                SELECT step, error_message, context, created_at 
                FROM failed_attempts 
                WHERE session_id = ? AND step = ?
                ORDER BY created_at DESC
            ''', (self.session_id, step))
        else:
            cursor.execute('''
                SELECT step, error_message, context, created_at 
                FROM failed_attempts 
                WHERE session_id = ?
                ORDER BY created_at DESC
            ''', (self.session_id,))
        
        return [
            {
                "step": row[0],
                "error_message": row[1],
                "context": row[2],
                "created_at": row[3]
            }
            for row in cursor.fetchall()
        ]
    
    def should_retry_step(self, step: str, max_retries: int = 3) -> bool:
        """Check if a step should be retried based on previous failures"""
        attempts = self.get_failed_attempts(step)
        return len(attempts) < max_retries
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get conversation history"""
        return self.conversation_memory.chat_memory.messages
    
    def clear_conversation(self):
        """Clear conversation memory"""
        self.conversation_memory.clear()
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """Get memory variables for LLM context"""
        return self.conversation_memory.load_memory_variables({})
    
    def close(self):
        """Close database connections"""
        if self.sqlite_conn:
            self.sqlite_conn.close()


class MemoryManager:
    """Factory class for creating and managing migration memory instances"""
    
    _instances: Dict[str, MigrationMemory] = {}
    
    @classmethod
    def get_memory(cls, session_id: str, **kwargs) -> MigrationMemory:
        """Get or create a memory instance for a session"""
        if session_id not in cls._instances:
            cls._instances[session_id] = MigrationMemory(session_id, **kwargs)
        return cls._instances[session_id]
    
    @classmethod
    def cleanup_session(cls, session_id: str):
        """Clean up a memory instance"""
        if session_id in cls._instances:
            cls._instances[session_id].close()
            del cls._instances[session_id]
    
    @classmethod
    def cleanup_all(cls):
        """Clean up all memory instances"""
        for session_id in list(cls._instances.keys()):
            cls.cleanup_session(session_id)