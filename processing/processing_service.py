# processing/processing.py

import os
import json
import io
import logging
import logging.handlers
import time
from minio import Minio
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from minio.error import S3Error
from prometheus_client import start_http_server, Summary, Counter

# Environment Variables
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
SOURCE_BUCKET = os.getenv("SOURCE_BUCKET", "mybucket")
TARGET_BUCKET = os.getenv("TARGET_BUCKET", "processed-bucket")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "test-topic")        # Topic to consume original files
CHUNK_TOPIC = os.getenv("CHUNK_TOPIC", "chunk-topic")       # Topic to send chunk metadata
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))             # Number of characters per chunk
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "/logs/data_processing.log")

# Configure Logging
logger = logging.getLogger("processing")
logger.setLevel(logging.INFO)

# Create a handler to write logs to the shared file
file_handler = logging.handlers.WatchedFileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.INFO)

# Create formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

# Optionally, add a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Define Prometheus metrics
PROCESS_TIME = Summary('process_time_seconds', 'Time spent processing files')
FILES_PROCESSED = Counter('files_processed_total', 'Total number of files processed')

def init_minio_client():
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False  # Set to True if using HTTPS
        )
        # Create target bucket if it doesn't exist
        if not client.bucket_exists(TARGET_BUCKET):
            client.make_bucket(TARGET_BUCKET)
            logger.info(f"Bucket '{TARGET_BUCKET}' created.")
        else:
            logger.info(f"Bucket '{TARGET_BUCKET}' already exists.")
        return client
    except Exception as e:
        logger.critical(f"Failed to initialize MinIO client: {e}")
        raise

def init_kafka_producer():
    for attempt in range(5):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                api_version=(0, 11, 5),
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            logger.info("Connected to Kafka producer successfully.")
            return producer
        except KafkaError as e:
            logger.error(f"Kafka producer connection failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    logger.critical("Failed to connect to Kafka producer after multiple attempts.")
    raise Exception("Failed to connect to Kafka producer after multiple attempts.")

def recursive_text_splitter(text, max_chunk_size):
    """
    Recursively splits text into chunks of up to 'max_chunk_size' characters.
    """
    chunks = []

    def split_recursively(subtext):
        if len(subtext) <= max_chunk_size:
            chunks.append(subtext)
            return
        else:
            # Find the last space character within the max_chunk_size
            split_point = max_chunk_size
            while split_point > 0 and subtext[split_point - 1] != ' ':
                split_point -= 1
            if split_point == 0:
                # If no space is found, split at max_chunk_size
                split_point = max_chunk_size
            # Recursively split the remaining text
            chunks.append(subtext[:split_point].rstrip())
            split_recursively(subtext[split_point:].lstrip())

    split_recursively(text)
    return chunks

@PROCESS_TIME.time()
def process_file(client, producer, object_name):
    """
    Downloads a file from MinIO, splits its content recursively, uploads the chunks,
    and sends metadata to Kafka.
    """
    try:
        # Download the file from MinIO
        logger.info(f"Downloading '{object_name}' from bucket '{SOURCE_BUCKET}'.")
        response = client.get_object(SOURCE_BUCKET, object_name)
        file_content = response.read().decode('utf-8')
        response.close()
        response.release_conn()

        # Split the content into chunks using recursive text splitter
        chunks = recursive_text_splitter(file_content, CHUNK_SIZE)
        logger.info(f"Split '{object_name}' into {len(chunks)} chunks.")

        # Upload each chunk to the target bucket and send metadata to Kafka
        for idx, chunk in enumerate(chunks, start=1):
            # Generate a unique name for each chunk
            sanitized_object_name = object_name.replace('/', '_')  # Replace '/' to avoid nested paths
            chunk_object_name = f"chunks/{sanitized_object_name}_chunk_{idx}.txt"
            client.put_object(
                bucket_name=TARGET_BUCKET,
                object_name=chunk_object_name,
                data=io.BytesIO(chunk.encode('utf-8')),
                length=len(chunk.encode('utf-8')),
                content_type="text/plain"
            )
            logger.info(f"Uploaded chunk '{chunk_object_name}' to bucket '{TARGET_BUCKET}'.")

            # Send metadata to Kafka
            metadata = {
                "bucket_name": TARGET_BUCKET,
                "object_name": chunk_object_name,
                "original_file": object_name,
                "chunk_index": idx,
                "total_chunks": len(chunks)
            }
            producer.send(CHUNK_TOPIC, metadata)
            producer.flush()
            logger.info(f"Sent metadata for chunk '{chunk_object_name}' to Kafka topic '{CHUNK_TOPIC}'.")

        FILES_PROCESSED.inc()

    except S3Error as e:
        logger.error(f"MinIO S3Error while processing '{object_name}': {e}")
    except Exception as e:
        logger.error(f"Error processing file '{object_name}': {e}")

def main():
    # Start Prometheus metrics server
    start_http_server(8001)  # Expose metrics on port 8001

    # Initialize MinIO client
    client = init_minio_client()

    # Initialize Kafka consumer
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='processing-group',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        logger.info(f"Connected to Kafka topic '{KAFKA_TOPIC}' as consumer.")
    except KafkaError as e:
        logger.critical(f"Failed to connect to Kafka as consumer: {e}")
        raise

    # Initialize Kafka producer
    producer = init_kafka_producer()

    # Consume messages
    for message in consumer:
        try:
            data = message.value
            logger.info(f"Received message: {data}")

            # Extract object name
            object_name = data.get("object_name")
            if not object_name:
                logger.warning("Received message without 'object_name'. Skipping.")
                continue

            # Process the file
            process_file(client, producer, object_name)

        except Exception as e:
            logger.error(f"Error consuming message: {e}")

if __name__ == "__main__":
    main()
