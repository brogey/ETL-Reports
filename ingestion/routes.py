# ingestion/routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import io
import requests
from minio.error import S3Error
from kafka.errors import KafkaError
import config  # Import the config module

router = APIRouter()

class URLPayload(BaseModel):
    url: str

@router.post("/upload/")
async def upload_file(payload: URLPayload):
    if not config.producer:
        config.logger.error("Kafka producer not initialized.")
        raise HTTPException(status_code=500, detail="Kafka producer not initialized.")

    try:
        # Fetch the content from the provided URL
        response = requests.get(payload.url)
        if response.status_code != 200:
            config.logger.error(f"Failed to fetch data from URL: {payload.url}, Status Code: {response.status_code}")
            raise HTTPException(status_code=400, detail="Failed to fetch data from URL.")

        file_content = response.content
        file_name = payload.url.split('/')[-1]  # Extract the file name from the URL
        config.logger.debug(f"Fetched file '{file_name}' with size {len(file_content)} bytes.")

        # Store file in MinIO
        object_name = f"content/{file_name}"
        config.minio_client.put_object(
            bucket_name=config.MINIO_BUCKET,
            object_name=object_name,
            data=io.BytesIO(file_content),
            length=len(file_content),
            content_type='application/octet-stream'
        )
        config.logger.info(f"Stored '{object_name}' in MinIO.")

        # Create the JSON payload for Kafka, including 'object_name'
        json_payload = {
            "object_name": object_name
        }

        # Send data to Kafka
        future = config.producer.send(config.KAFKA_TOPIC, json_payload)
        config.producer.flush()
        result = future.get(timeout=10)
        config.logger.info(f"Sent data to Kafka topic '{config.KAFKA_TOPIC}' with offset {result.offset}.")

        return {"message": "File fetched, uploaded, and data sent to Kafka successfully.", "object_name": object_name}

    except requests.exceptions.RequestException as e:
        config.logger.error(f"Error fetching data from URL: {e}")
        raise HTTPException(status_code=400, detail="Error fetching data from URL.")
    except S3Error as e:
        config.logger.error(f"Failed to store file in MinIO: {e}")
        raise HTTPException(status_code=500, detail="Failed to store file in MinIO.")
    except KafkaError as e:
        config.logger.error(f"Failed to send data to Kafka: {e}")
        raise HTTPException(status_code=500, detail="Failed to send data to Kafka.")
    except Exception as e:
        config.logger.exception("An unexpected error occurred.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
