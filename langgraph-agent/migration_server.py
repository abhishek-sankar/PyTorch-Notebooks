"""
Migration Server with LangGraph Streaming Support
Integrates the SupervisorMigrationOrchestrator with LangGraph SDK-compatible streaming
"""
import os
import sys
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime
import uuid
import json
import asyncio

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from migration.supervisor_orchestrator import SupervisorMigrationOrchestrator

load_dotenv()

app = FastAPI(title="LangGraph Migration Server")

# In-memory storage for threads and messages
threads_db = {}
messages_db = {}

# Enable CORS for the chat UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize migration orchestrator
print("Initializing Migration Orchestrator...")
migration_orchestrator = SupervisorMigrationOrchestrator()
print("Migration orchestrator ready!")

class MigrationRequest(BaseModel):
    project_path: str
    
class ChatMessage(BaseModel):
    content: str
    type: str = "human"

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    thread_id: str = "default"

@app.get("/")
async def root():
    return {"message": "LangGraph Migration Server is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/info")
async def get_info():
    """Return server info"""
    return {
        "version": "1.0.0",
        "name": "Migration Supervisor Server"
    }

@app.get("/assistants/{assistant_id}")
async def get_assistant(assistant_id: str):
    """Get assistant details"""
    return {
        "assistant_id": assistant_id,
        "name": "Migration Supervisor",
        "description": "Multi-agent supervisor for Java project migrations",
        "config": {},
        "graph_id": assistant_id
    }

@app.get("/assistants")
async def get_assistants():
    """Return available assistants"""
    return [
        {
            "assistant_id": "supervisor",
            "name": "Migration Supervisor", 
            "description": "Multi-agent supervisor for Java project migrations"
        }
    ]

@app.get("/threads")
async def get_threads():
    """Get all threads"""
    return []

@app.post("/threads")
async def create_thread():
    """Create a new thread"""
    thread_id = str(uuid.uuid4())
    return {"thread_id": thread_id}

@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """Get a specific thread"""
    return {
        "thread_id": thread_id,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "metadata": {}
    }

@app.post("/threads/{thread_id}/history")
async def get_thread_history(thread_id: str, request: Dict[str, Any]):
    """Get thread message history with proper checkpoint structure"""
    messages = messages_db.get(thread_id, [])
    
    # Return history in the format expected by the SDK
    history = []
    for i, msg_batch in enumerate([messages]):  # Batch messages per checkpoint
        checkpoint_id = str(uuid.uuid4())
        history_item = {
            "values": {"messages": msg_batch},
            "next": [],
            "checkpoint": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
                "checkpoint_ns": "",
                "checkpoint_map": {}
            },
            "metadata": {
                "step": i,
                "source": "loop"
            },
            "created_at": datetime.now().isoformat(),
            "parent_checkpoint": {
                "checkpoint_id": f"parent_{checkpoint_id}" if i > 0 else None
            } if i > 0 else None,
            "tasks": []
        }
        history.append(history_item)
    
    return history

@app.post("/threads/search")
async def search_threads(request: Dict[str, Any]):
    """Search for threads"""
    threads = []
    for thread_id, thread_data in threads_db.items():
        # Get latest message for preview
        thread_messages = messages_db.get(thread_id, [])
        latest_message = ""
        if thread_messages:
            for msg in reversed(thread_messages):
                if msg.get("type") == "human" and msg.get("content"):
                    if isinstance(msg["content"], str):
                        latest_message = msg["content"][:100]
                        break
                    elif isinstance(msg["content"], list) and msg["content"]:
                        latest_message = msg["content"][0].get("text", "")[:100]
                        break
        
        threads.append({
            "thread_id": thread_id,
            "created_at": thread_data["created_at"],
            "updated_at": thread_data["updated_at"],
            "metadata": thread_data["metadata"],
            "values": {"messages": thread_messages[-5:] if thread_messages else []},
            "name": latest_message or f"Migration {thread_id[:8]}"
        })
    
    return threads

@app.post("/threads/{thread_id}/runs/stream")
async def stream_migration_run(thread_id: str, request: Dict[str, Any]):
    """LangGraph-compatible streaming endpoint for migrations"""
    
    async def generate_stream():
        try:
            input_data = request.get("input", {})
            messages = input_data.get("messages", [])
            
            # Ensure thread exists
            if thread_id not in threads_db:
                now = datetime.now().isoformat()
                threads_db[thread_id] = {
                    "thread_id": thread_id,
                    "created_at": now,
                    "updated_at": now,
                    "metadata": {}
                }
                messages_db[thread_id] = []
            
            # Generate run metadata
            run_id = f"run_{thread_id}_{uuid.uuid4()}"
            
            # LangGraph expected format: start with metadata event
            yield f"event: metadata\n"
            yield f"data: {json.dumps({'run_id': run_id, 'thread_id': thread_id})}\n\n"
            
            # Extract project path from messages
            if not messages:
                yield f"event: error\n"
                yield f"data: {json.dumps({'error': 'No messages provided'})}\n\n"
                return
            
            project_path = ""
            latest_message = messages[-1]
            
            # Handle LangGraph message format
            if isinstance(latest_message, dict):
                content = latest_message.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            project_path = item.get("text", "").strip()
                            break
                        elif isinstance(item, str):
                            project_path = item.strip()
                            break
                    if not project_path:
                        project_path = str(content).strip()
                else:
                    project_path = str(content).strip()
            else:
                project_path = str(latest_message).strip()
            
            print(f"STREAMING: Starting migration for: {project_path}")
            
            # Convert messages to LangGraph format
            langgraph_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    langgraph_messages.append({
                        "id": msg.get("id", str(uuid.uuid4())),
                        "type": msg.get("type", "human"),
                        "content": msg.get("content", "")
                    })
            
            # Add initial human message
            if not langgraph_messages:
                langgraph_messages.append({
                    "id": str(uuid.uuid4()),
                    "type": "human",
                    "content": f"Migrate project at {project_path}"
                })
            
            # Start migration streaming
            final_result = None
            
            # Stream migration execution with real-time progress
            for progress in migration_orchestrator.migrate_project_stream(project_path):
                
                if progress.get("type") == "progress":
                    # Create an AI message for progress updates
                    agent_name = progress.get("agent", "supervisor")
                    step = progress.get("step", 0)
                    message_content = progress.get("message", "Working...")
                    
                    # Add tool info if available
                    if progress.get("tool"):
                        message_content += f" (using {progress['tool']})"
                    
                    ai_message = {
                        "id": f"progress_{step}_{uuid.uuid4()}",
                        "type": "ai", 
                        "content": f"[{agent_name.upper()}] {message_content}"
                    }
                    
                    # Append to conversation instead of replacing
                    langgraph_messages.append(ai_message)
                    
                    print(f"MIGRATION: YIELDING PROGRESS: Step {step} - {agent_name}")
                    
                    # Send values event in LangGraph format
                    yield f"event: values\n"
                    yield f"data: {json.dumps({'messages': langgraph_messages.copy()})}\n\n"
                    
                    # Small delay to ensure streaming visibility
                    import time
                    time.sleep(0.1)
                    
                elif progress.get("type") == "complete":
                    final_result = progress
                    break
            
            # Send final result
            if final_result:
                final_content = ""
                if final_result.get('success'):
                    result_text = final_result.get('result', 'Migration completed successfully')
                    duration = final_result.get('duration', 0)
                    steps = final_result.get('steps', 0)
                    
                    final_content = f"""Migration completed successfully! ✅

**Project:** {project_path}
**Duration:** {duration:.2f} seconds  
**Steps:** {steps}

**Results:**
{result_text}"""
                else:
                    error_text = final_result.get('error', 'Unknown error occurred')
                    final_content = f"""Migration failed ❌

**Project:** {project_path}
**Error:** {error_text}

Please check the logs for more details."""
                
                final_message = {
                    "id": f"final_{run_id}",
                    "type": "ai",
                    "content": final_content
                }
                
                final_messages = langgraph_messages + [final_message]
                
                # Store conversation
                messages_db[thread_id] = final_messages
                threads_db[thread_id]["updated_at"] = datetime.now().isoformat()
                
                # Send final values
                yield f"event: values\n"
                yield f"data: {json.dumps({'messages': final_messages})}\n\n"
            
            # End the stream properly
            yield f"event: end\n"
            yield f"data: {json.dumps({'run_id': run_id})}\n\n"
            
        except Exception as e:
            print(f"STREAMING ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            yield f"event: error\n"
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting Migration Supervisor Server on port 2025...")
    print("Available agents: analysis_expert, execution_expert, error_expert")
    uvicorn.run(app, host="0.0.0.0", port=2025)