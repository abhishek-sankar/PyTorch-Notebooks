#!/usr/bin/env python3
"""
Test script to reproduce the streaming endpoint error
"""
import json
import asyncio
from server import stream_run

async def test_stream():
    # Simulate the exact request you're sending
    test_request = {
        "input": {
            "messages": [{
                "id": "b517ee61-d068-4bd0-b14c-5025cc22ef74",
                "type": "human",
                "content": [{"type": "text", "text": "/Users/abhisheksankar/Desktop/PyTorch-Notebooks/langgraph-agent/airline"}]
            }]
        },
        "stream_mode": ["values", "messages-tuple", "custom"],
        "assistant_id": "supervisor",
        "on_disconnect": "cancel"
    }
    
    thread_id = "test-thread"
    
    try:
        print("Testing streaming endpoint...")
        response = await stream_run(thread_id, test_request)
        
        # Collect the stream events
        events = []
        async for event in response.body_iterator:
            event_str = event.decode() if hasattr(event, 'decode') else str(event)
            print(f"Event: {event_str}")
            events.append(event_str)
        
        print(f"Total events: {len(events)}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_stream())