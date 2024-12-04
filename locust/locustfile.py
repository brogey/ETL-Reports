from locust import HttpUser, task, between
import logging
import os

# Configure Logging
logger = logging.getLogger("locust")
logger.setLevel(logging.INFO)

class IngestionUser(HttpUser):
    wait_time = between(1, 5)
    host = os.getenv("LOCUST_HOST")

    def on_start(self):
        if self.host is None:
            logger.error("Host is not set. Please ensure that the host is specified.")
        else:
            logger.info(f"Starting load test against host: {self.host}")

    @task
    def send_url(self):
        logger.info("Executing send_url task")
        # List files in the content directory
        files = os.listdir('content')  # You can dynamically list files if needed
        for file_name in files:
            # The ingestion service will fetch the file from the Locust service
            file_url = f"http://locust:8000/{file_name}"
            payload = {'url': file_url}
            with self.client.post("/upload/", json=payload, catch_response=True) as response:
                if response.status_code != 200:
                    logger.error(f"Failed to send URL: Status Code {response.status_code}, Response: {response.text}")
                    response.failure(f"Failed to send URL: Status Code {response.status_code}, Response: {response.text}")
                else:
                    logger.info("URL sent successfully")
                    response.success()