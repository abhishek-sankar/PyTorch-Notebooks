import os
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

from supervisor_app import create_supervisor_system
from langchain_core.messages import HumanMessage, AIMessage

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

# Initialize supervisor
supervisor = create_supervisor_system()

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
    """Main chat endpoint for the supervisor"""
    try:
        # Convert messages to LangChain format
        messages = []
        for msg in request.messages:
            if msg.type == "human":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))
        
        # Get response from supervisor
        response = supervisor.invoke({"messages": messages})
        
        # Extract response content
        if isinstance(response, dict) and "messages" in response:
            last_message = response["messages"][-1]
            if hasattr(last_message, 'content'):
                content = last_message.content
            else:
                content = str(last_message)
        else:
            content = str(response)
        
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
    """LangGraph-compatible runs endpoint"""
    try:
        # Extract input from request
        input_data = request.get("input", {})
        messages = input_data.get("messages", [])
        
        # Convert messages to LangChain format
        langchain_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                if msg.get("type") == "human":
                    langchain_messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("type") == "ai": 
                    langchain_messages.append(AIMessage(content=msg.get("content", "")))
            else:
                langchain_messages.append(msg)
        
        # Invoke supervisor
        response = supervisor.invoke({"messages": langchain_messages})
        
        # Format response for LangGraph chat UI
        return {
            "run_id": f"run_{thread_id}",
            "thread_id": thread_id,
            "status": "success",
            "output": response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/assistants/{assistant_id}")
async def get_assistant(assistant_id: str):
    """Get assistant details"""
    return {
        "assistant_id": assistant_id,
        "name": "LangGraph Supervisor",
        "description": "Multi-agent supervisor with weather and calculator agents",
        "config": {},
        "graph_id": assistant_id
    }

@app.get("/assistants")
async def get_assistants():
    """Return available assistants"""
    return [
        {
            "assistant_id": "supervisor",
            "name": "LangGraph Supervisor",
            "description": "Multi-agent supervisor with weather and calculator agents"
        }
    ]

@app.get("/info")
async def get_info():
    """Return server info"""
    return {
        "version": "1.0.0",
        "name": "LangGraph Supervisor Server"
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
    """Streaming endpoint for real-time responses using SSE format"""
    
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
            
            # Convert new messages from UI format to LangChain format
            # The supervisor with MemorySaver will handle conversation history automatically
            langchain_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get("type") == "human":
                        langchain_messages.append(HumanMessage(content=msg.get("content", "")))
                    elif msg.get("type") == "ai":
                        langchain_messages.append(AIMessage(content=msg.get("content", "")))
                else:
                    langchain_messages.append(msg)
            
            # Invoke supervisor with proper thread configuration for memory persistence
            # The supervisor's MemorySaver will maintain conversation history per thread
            config = {"configurable": {"thread_id": thread_id}}
            response = supervisor.invoke({"messages": langchain_messages}, config=config)
            
            # Format the response properly
            if isinstance(response, dict) and "messages" in response:
                # Convert LangChain messages to proper format
                formatted_messages = []
                for i, msg in enumerate(response["messages"]):
                    formatted_msg = {
                        "id": getattr(msg, 'id', f"msg_{i}"),
                        "type": getattr(msg, 'type', 'ai'),
                        "content": getattr(msg, 'content', str(msg))
                    }
                    
                    # Add additional fields if they exist
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        formatted_msg["tool_calls"] = msg.tool_calls
                    if hasattr(msg, 'tool_call_id'):
                        formatted_msg["tool_call_id"] = msg.tool_call_id
                    if hasattr(msg, 'name'):
                        formatted_msg["name"] = msg.name
                    
                    formatted_messages.append(formatted_msg)
                
                # Store the complete conversation history (all messages from the response)
                # This preserves the full conversation context including agent interactions
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
    print("Starting LangGraph Supervisor Server on port 2024...")
    print("Available agents: weather_agent, calculator_agent")
    uvicorn.run(app, host="0.0.0.0", port=2024)