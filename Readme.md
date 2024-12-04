1. ingestion is the fastapi service 
2. promethues and grafana integration is remaining
3. we fetch the data from locust. Locust will use post request to the endpoint of our fastapi service at http://localhost:8000/
4. the data is stored in minio and the minio bucket details are sent to processing service to divide it in chunks.
5. the processing service divides the text into chunk and stored in the bucket.
6. minio can be accessed via link http://127.0.0.1:9001/browser with username and password - minioadmin
7. use http://host.docker.internal:9090 for grafana data source

Explanation of Changes
app.py:

Simplified to focus on application setup and event handling.
Imports init_kafka_producer from config.py and includes routes from routes.py.
config.py:

Consolidates environment variables, configurations, logging setup, and client initializations.
Defines init_minio() and init_kafka_producer() functions.
Initializes minio_client and producer at the module level.
routes.py:

Contains the /upload/ endpoint logic.
Imports necessary configurations and clients from config.py.
Minimal Changes to Other Files:

Only updated the Dockerfile to ensure all files are copied into the Docker image.
No changes to processing.py, its Dockerfile, or docker-compose.yml.