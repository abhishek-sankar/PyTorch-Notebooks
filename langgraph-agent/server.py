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

# Add migration directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'migration'))
from supervisor_orchestrator import SupervisorMigrationOrchestrator

load_dotenv()

app = FastAPI(title="LangGraph Supervisor Server")

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

# Initialize migration supervisor
supervisor = SupervisorMigrationOrchestrator()

class ChatMessage(BaseModel):
    content: str
    type: str = "human"

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    thread_id: str = "default"

class ChatResponse(BaseModel):
    content: str
    type: str = "ai"

@app.get("/")
async def root():
    return {"message": "LangGraph Supervisor Server is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint for the migration supervisor"""
    try:
        # Get the latest message content (assuming it's a project path)
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        latest_message = request.messages[-1]
        project_path = latest_message.content.strip()
        
        # Check if it's a valid project path
        if not os.path.exists(project_path):
            return ChatResponse(
                content=f"Error: Project path '{project_path}' does not exist. Please provide a valid project path for migration.",
                type="ai"
            )
        
        # Use migrate_project method
        result = supervisor.migrate_project(project_path)
        
        if result.get('success'):
            content = f"Migration completed successfully!\n\n{result.get('result', '')}\n\nDuration: {result.get('duration', 0):.2f} seconds"
        else:
            content = f"Migration failed: {result.get('error', 'Unknown error')}"
        
        return ChatResponse(content=content, type="ai")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/threads")
async def get_threads():
    """Get all threads"""
    return []

@app.post("/threads")
async def create_thread():
    """Create a new thread"""
    import uuid
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

@app.post("/threads/{thread_id}/runs")
async def create_run(thread_id: str, request: Dict[str, Any]):
    """LangGraph-compatible runs endpoint for migration"""
    try:
        # Extract input from request
        input_data = request.get("input", {})
        messages = input_data.get("messages", [])
        
        # Get the latest message content (should be a project path)
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        latest_message = messages[-1]
        if isinstance(latest_message, dict):
            project_path = latest_message.get("content", "").strip()
        else:
            project_path = str(latest_message).strip()
        
        # Check if it's a valid project path
        if not os.path.exists(project_path):
            return {
                "run_id": f"run_{thread_id}",
                "thread_id": thread_id,
                "status": "error",
                "output": {"error": f"Project path '{project_path}' does not exist"}
            }
        
        # Use migrate_project method
        result = supervisor.migrate_project(project_path)
        
        # Format response for LangGraph chat UI
        return {
            "run_id": f"run_{thread_id}",
            "thread_id": thread_id,
            "status": "success" if result.get('success') else "error",
            "output": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistants/{assistant_id}")
async def get_assistant(assistant_id: str):
    """Get assistant details"""
    return {
        "assistant_id": assistant_id,
        "name": "Migration Supervisor",
        "description": "Multi-agent supervisor for Java project migration with analysis, execution, and error-fixing agents",
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
            "description": "Multi-agent supervisor for Java project migration with analysis, execution, and error-fixing agents"
        }
    ]

@app.get("/info")
async def get_info():
    """Return server info"""
    return {
        "version": "1.0.0",
        "name": "Migration Supervisor Server"
    }

@app.post("/threads/search")
async def search_threads(request: Dict[str, Any]):
    """Search for threads"""
    # Return all threads with conversation previews
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
            "values": {"messages": thread_messages[-5:] if thread_messages else []},  # Last 5 messages for preview
            "name": latest_message or f"Thread {thread_id[:8]}"
        })
    
    return threads

@app.post("/threads/{thread_id}/runs/stream")
async def stream_run(thread_id: str, request: Dict[str, Any]):
    """Streaming endpoint for migration with real-time responses"""
    
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
            
            # Yield metadata first
            run_id = f"run_{thread_id}_{uuid.uuid4()}"
            checkpoint_id = str(uuid.uuid4())
            
            yield f"event: metadata\n"
            yield f"data: {json.dumps({'run_id': run_id, 'thread_id': thread_id})}\n\n"
            
            # Get the latest message content (should be a project path)
            if not messages:
                yield f"event: error\n"
                yield f"data: {json.dumps({'error': 'no_messages', 'message': 'No messages provided'})}\n\n"
                return
            
            latest_message = messages[-1]
            if isinstance(latest_message, dict):
                project_path = latest_message.get("content", "").strip()
            else:
                project_path = str(latest_message).strip()
            
            # Check if it's a valid project path
            if not os.path.exists(project_path):
                yield f"event: error\n"
                yield f"data: {json.dumps({'error': 'invalid_path', 'message': f'Project path does not exist: {project_path}'})}\n\n"
                return
            
            # Use migrate_project method
            result = supervisor.migrate_project(project_path)
            
            # Format the response 
            if result.get('success'):
                formatted_messages = [
                    {
                        "id": "human_input",
                        "type": "human",
                        "content": project_path
                    },
                    {
                        "id": "migration_result",
                        "type": "ai",
                        "content": f"Migration completed successfully!\n\n{result.get('result', '')}\n\nDuration: {result.get('duration', 0):.2f} seconds"
                    }
                ]
            else:
                formatted_messages = [
                    {
                        "id": "human_input", 
                        "type": "human",
                        "content": project_path
                    },
                    {
                        "id": "migration_error",
                        "type": "ai", 
                        "content": f"Migration failed: {result.get('error', 'Unknown error')}"
                    }
                ]
            
            # Store the conversation
            messages_db[thread_id] = formatted_messages
            threads_db[thread_id]["updated_at"] = datetime.now().isoformat()
            
            # Create proper checkpoint structure
            values_data = {
                "messages": formatted_messages,
                "checkpoint": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_ns": "",
                    "checkpoint_map": {}
                },
                "metadata": {
                    "step": 1,
                    "source": "loop"
                },
                "next": [],
                "tasks": []
            }
            
            # Yield values event
            yield f"event: values\n"
            yield f"data: {json.dumps(values_data)}\n\n"
            
            # End the stream
            yield f"event: end\n"
            yield f"data: {json.dumps({'run_id': run_id})}\n\n"
            
        except Exception as e:
            # Yield error event
            yield f"event: error\n"
            yield f"data: {json.dumps({'error': 'internal_error', 'message': str(e)})}\n\n"
    
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
    print("Starting Migration Supervisor Server on port 2024...")
    print("Available workers: analysis_expert, execution_expert, error_expert")
    uvicorn.run(app, host="0.0.0.0", port=2024)