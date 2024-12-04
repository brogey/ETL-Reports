#!/bin/bash

# Start the HTTP server in the background
cd content
python3 -m http.server 8000 &
cd ..

# Start Locust
locust -f locustfile.py --headless -u "${LOCUST_USERS}" -r "${LOCUST_SPAWN_RATE}" --run-time "${LOCUST_RUN_TIME}"
