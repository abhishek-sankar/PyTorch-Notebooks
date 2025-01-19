# device_manager.py
from fastapi import FastAPI, HTTPException
from redis import Redis
from sqlalchemy import create_engine
from kafka import KafkaProducer
import boto3
import cv2
import numpy as np
from datetime import datetime
import json

app = FastAPI()

# Database connections
redis_client = Redis(host='localhost', port=6379, db=0)
postgres_engine = create_engine('postgresql://user:password@localhost:5432/devices')
kafka_producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
sqs = boto3.client('sqs', region_name='us-west-2')

class DeviceManager:
    def __init__(self):
        self.device_cache = {}
    
    async def register_device(self, device_id: str, metadata: dict):
        """Register a new IoT device in the system"""
        try:
            # Store device metadata in Redis for quick access
            redis_client.hmset(f"device:{device_id}", metadata)
            
            # Store detailed device info in Postgres
            with postgres_engine.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO devices (device_id, location, status, created_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (device_id, metadata['location'], 'active', datetime.now())
                )
            
            # Publish device registration event to Kafka
            kafka_producer.send(
                'device_events',
                key=device_id.encode(),
                value=json.dumps({
                    'event': 'registration',
                    'device_id': device_id,
                    'timestamp': datetime.now().isoformat()
                }).encode()
            )
            
            return {"status": "success", "device_id": device_id}
            
        except Exception as e:
            # Send error to SQS dead letter queue
            sqs.send_message(
                QueueUrl='https://sqs.us-west-2.amazonaws.com/dead-letter-queue',
                MessageBody=json.dumps({
                    'error': str(e),
                    'device_id': device_id,
                    'timestamp': datetime.now().isoformat()
                })
            )
            raise HTTPException(status_code=500, detail=str(e))

    async def process_device_image(self, device_id: str, image_data: bytes):
        """Process images from device using computer vision"""
        # Convert image bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Perform basic image processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Store processed results in Redis cache
        redis_client.set(
            f"device:{device_id}:latest_processed",
            edges.tobytes(),
            ex=3600  # expire after 1 hour
        )
        
        return {"status": "success", "processed": True}

    async def get_device_status(self, device_id: str):
        """Get real-time device status from cache"""
        status = redis_client.hgetall(f"device:{device_id}")
        if not status:
            raise HTTPException(status_code=404, detail="Device not found")
        return status

# API Routes
@app.post("/devices/register")
async def register_device(device_id: str, metadata: dict):
    device_manager = DeviceManager()
    return await device_manager.register_device(device_id, metadata)

@app.post("/devices/{device_id}/process-image")
async def process_device_image(device_id: str, image_data: bytes):
    device_manager = DeviceManager()
    return await device_manager.process_device_image(device_id, image_data)

@app.get("/devices/{device_id}/status")
async def get_device_status(device_id: str):
    device_manager = DeviceManager()
    return await device_manager.get_device_status(device_id)