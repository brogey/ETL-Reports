# ingestion/app.py

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from config import init_kafka_producer
from routes import router
import config  # Import the config module

app = FastAPI()

# Initialize Prometheus Instrumentator
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Include routes
app.include_router(router)

# Initialize Kafka producer on startup
@app.on_event("startup")
def startup_event():
    init_kafka_producer()

# Shutdown resources if necessary
@app.on_event("shutdown")
def shutdown_event():
    if config.producer is not None:
        config.producer.close()
