#!/bin/bash

# This script renames directories and files in your ETL project
# and updates references within code and configuration files.

# Ensure the script is run from the project root directory
# Check if 'docker-compose.yml' exists in the current directory
if [ ! -f docker-compose.yml ]; then
    echo "Please run this script from the project root directory where 'docker-compose.yml' is located."
    exit 1
fi

# Function to replace text in files
replace_in_files() {
    local search="$1"
    local replace="$2"
    local file_types="$3"
    grep -rl --exclude-dir=".git" --exclude="rename_etl.sh" "$search" . | xargs sed -i "s/$search/$replace/g"
}

# Rename directories
mv myapp ingestion
mv processor processing
mv retriever retrieval

# Rename Python files within the directories
mv ingestion/app.py ingestion/ingestion_service.py
mv processing/processor.py processing/processor_service.py
mv retrieval/retriever.py retrieval/retriever_service.py

echo "Renamed directories and files."

# Update references in code and configuration files

# Update import statements and file references in Python files
echo "Updating references in Python files..."
replace_in_files "from app" "from ingestion_service" "*.py"
replace_in_files "import app" "import ingestion_service" "*.py"
replace_in_files "from processor" "from processor_service" "*.py"
replace_in_files "import processor" "import processor_service" "*.py"
replace_in_files "from retriever" "from retriever_service" "*.py"
replace_in_files "import retriever" "import retriever_service" "*.py"

replace_in_files "from ingestion.app" "from ingestion.ingestion_service" "*.py"
replace_in_files "from processing.processor" "from processing.processor_service" "*.py"
replace_in_files "from retrieval.retriever" "from retrieval.retriever_service" "*.py"

# Update file paths in code
replace_in_files "'app.py'" "'ingestion_service.py'" "*.py"
replace_in_files "'processor.py'" "'processor_service.py'" "*.py"
replace_in_files "'retriever.py'" "'retriever_service.py'" "*.py"

# Update Dockerfile CMD entries
echo "Updating Dockerfile CMD entries..."
sed -i 's/CMD \["python", "app.py"\]/CMD ["python", "ingestion_service.py"]/' ingestion/Dockerfile
sed -i 's/CMD \["python", "processor.py"\]/CMD ["python", "processor_service.py"]/' processing/Dockerfile
sed -i 's/CMD \["python", "retriever.py"\]/CMD ["python", "retriever_service.py"]/' retrieval/Dockerfile

# Update docker-compose.yml
echo "Updating docker-compose.yml..."
sed -i 's/build: \.\/myapp/build: .\/ingestion/' docker-compose.yml
sed -i 's/container_name: myapp/container_name: ingestion_service/' docker-compose.yml
sed -i 's/service_name: myapp/service_name: ingestion_service/' docker-compose.yml
sed -i 's/build: \.\/processor/build: .\/processing/' docker-compose.yml
sed -i 's/container_name: processor/container_name: processing_service/' docker-compose.yml
sed -i 's/service_name: processor/service_name: processing_service/' docker-compose.yml
sed -i 's/build: \.\/retriever/build: .\/retrieval/' docker-compose.yml
sed -i 's/container_name: retriever/container_name: retrieval_service/' docker-compose.yml
sed -i 's/service_name: retriever/service_name: retrieval_service/' docker-compose.yml

# Update log file paths in code and docker-compose.yml
echo "Updating log file paths..."
replace_in_files "myapp.log" "data_ingestion.log" "*.py" "*.yml"
replace_in_files "processor.log" "data_processing.log" "*.py" "*.yml"
replace_in_files "retriever.log" "data_retrieval.log" "*.py" "*.yml"

replace_in_files "myapp" "ingestion" "docker-compose.yml"
replace_in_files "processor" "processing" "docker-compose.yml"
replace_in_files "retriever" "retrieval" "docker-compose.yml"

# Update service names in docker-compose.yml
sed -i 's/^\s*myapp:/  ingestion:/' docker-compose.yml
sed -i 's/^\s*processor:/  processing:/' docker-compose.yml
sed -i 's/^\s*retriever:/  retrieval:/' docker-compose.yml

echo "Updating Prometheus configuration..."
if [ -f prometheus.yml ]; then
    sed -i 's/job_name:.*myapp/job_name: ingestion_service/' prometheus.yml
    sed -i 's/targets:.*myapp:8000/targets: ["ingestion_service:8000"]/' prometheus.yml
    sed -i 's/job_name:.*processor/job_name: processing_service/' prometheus.yml
    sed -i 's/targets:.*processor:8001/targets: ["processing_service:8001"]/' prometheus.yml
fi

echo "Renaming and updates completed successfully."

echo "You can now rebuild and run your Docker containers."
