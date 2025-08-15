"""
Test streaming functionality for both weather and migration supervisors
"""
import asyncio
import aiohttp
import json
from datetime import datetime

async def test_weather_streaming():
    """Test the weather supervisor streaming"""
    print("=" * 60)
    print("TESTING WEATHER SUPERVISOR STREAMING")
    print("=" * 60)
    
    url = "http://localhost:2024/threads/test_weather/runs/stream"
    
    data = {
        "input": {
            "messages": [
                {
                    "id": "test_msg",
                    "type": "human",
                    "content": "What's the weather like in New York?"
                }
            ]
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                print(f"Response status: {response.status}")
                print(f"Response headers: {dict(response.headers)}")
                
                if response.status == 200:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            try:
                                event_data = json.loads(line[6:])  # Remove 'data: ' prefix
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] {event_data}")
                            except json.JSONDecodeError as e:
                                print(f"Failed to parse JSON: {line} - Error: {e}")
                        elif line.startswith('event: '):
                            event_type = line[7:]  # Remove 'event: ' prefix
                            print(f"Event: {event_type}")
                        elif line:
                            print(f"Raw line: {line}")
                else:
                    error_text = await response.text()
                    print(f"Error: {error_text}")
                    
    except Exception as e:
        print(f"Connection error: {e}")

async def test_migration_streaming():
    """Test the migration supervisor streaming"""
    print("=" * 60)
    print("TESTING MIGRATION SUPERVISOR STREAMING")
    print("=" * 60)
    
    url = "http://localhost:2025/threads/test_migration/runs/stream"
    
    data = {
        "input": {
            "messages": [
                {
                    "id": "test_migration_msg",
                    "type": "human",
                    "content": "../migration/flatworm"  # Path to test project
                }
            ]
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                print(f"Response status: {response.status}")
                print(f"Response headers: {dict(response.headers)}")
                
                if response.status == 200:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            try:
                                event_data = json.loads(line[6:])  # Remove 'data: ' prefix
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] {event_data}")
                            except json.JSONDecodeError as e:
                                print(f"Failed to parse JSON: {line} - Error: {e}")
                        elif line.startswith('event: '):
                            event_type = line[7:]  # Remove 'event: ' prefix
                            print(f"Event: {event_type}")
                        elif line:
                            print(f"Raw line: {line}")
                else:
                    error_text = await response.text()
                    print(f"Error: {error_text}")
                    
    except Exception as e:
        print(f"Connection error: {e}")

async def main():
    """Run both streaming tests"""
    print("Starting streaming tests...\n")
    
    # Test weather streaming
    await test_weather_streaming()
    
    print("\n" + "=" * 80 + "\n")
    
    # Test migration streaming  
    await test_migration_streaming()
    
    print("\nStreaming tests completed!")

if __name__ == "__main__":
    asyncio.run(main())