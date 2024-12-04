FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY ./app .

# Expose Streamlit port
EXPOSE 8501

# Command to run the application
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py", "--server.address", "0.0.0.0"]