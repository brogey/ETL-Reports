# locust/Dockerfile

FROM python:3.11-slim

WORKDIR /locust

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY ./start.sh /start.sh
RUN chmod +x start.sh

EXPOSE 8089  
EXPOSE 8000  

CMD ["./start.sh"]
