# retrieval/retrieval.py

import os
import json
import io
import logging
import logging.handlers
from minio import Minio
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from minio.error import S3Error

# Environment Variables
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
CHUNK_TOPIC = os.getenv("CHUNK_TOPIC", "chunk-topic")
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "/logs/data_retrieval.log")

# Configure Logging
logger = logging.getLogger("retrieval")
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

def init_minio_client():
    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False  # Set to True if using HTTPS
        )
        logger.info("Initialized MinIO client.")
        return client
    except Exception as e:
        logger.critical(f"Failed to initialize MinIO client: {e}")
        raise

def main():
    # Initialize MinIO client
    client = init_minio_client()

    # Initialize Kafka consumer
    try:
        consumer = KafkaConsumer(
            CHUNK_TOPIC,
            bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='retrieval-group',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        logger.info(f"Connected to Kafka topic '{CHUNK_TOPIC}' as consumer.")
    except KafkaError as e:
        logger.critical(f"Failed to connect to Kafka as consumer: {e}")
        raise

    # Batch processing variables
    docs_to_add = []
    batch_size = 10  # Adjust based on your preference


    # Consume messages
    for message in consumer:
        try:
            data = message.value
            logger.info(f"Received chunk metadata: {data}")

            bucket_name = data.get("bucket_name")
            object_name = data.get("object_name")
            original_file = data.get("original_file")
            chunk_index = data.get("chunk_index")
            total_chunks = data.get("total_chunks")

            if not bucket_name or not object_name:
                logger.warning("Incomplete chunk metadata received. Skipping.")
                continue

            # Retrieve the chunk from MinIO
            logger.info(f"Retrieving chunk '{object_name}' from bucket '{bucket_name}'.")
            response = client.get_object(bucket_name, object_name)
            chunk_content = response.read().decode('utf-8')
            response.close()
            response.release_conn()

            # Process the chunk as needed
            # For demonstration, we'll just log the size of the chunk
            chunk_size = len(chunk_content)
            logger.info(f"Retrieved chunk '{object_name}' of size {chunk_size} bytes.")

            # TODO: Implement any additional processing of the chunk here

            # Create a Document object
            doc = Document(
                page_content=chunk_content,
                metadata={
                    'source': object_name,
                    'original_file': original_file,
                    'chunk_index': chunk_index,
                    'total_chunks': total_chunks
                }
            )

            # Append to batch list
            docs_to_add.append(doc)

            if len(docs_to_add) >= batch_size:
                # Add documents to vectorstore without splitting
                vectorstore.add_documents(docs_to_add)
                logger.info(f"Successfully added {len(docs_to_add)} documents to Qdrant.")
                docs_to_add = []


        except Exception as e:
            logger.error(f"Error processing chunk metadata: {e}")

if __name__ == "__main__":
    main()
