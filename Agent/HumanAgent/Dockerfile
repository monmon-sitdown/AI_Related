FROM ubuntu:20.04

RUN apt-get update && apt-get install -y coturn python3 python3-pip python3-venv redis-server && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && python3 -m pip install fastapi uvicorn langchain_core langchain_openai langchain_community openai redis google-search-results

COPY turnserver.conf /etc/turnserver.conf

COPY redis.conf /etc/redis/redis.conf

VOLUME /data

WORKDIR /app

COPY . /app

EXPOSE 8000 3478 6379

CMD ["sh", "-c", "turnserver -c /etc/turnserver.conf --listening-ip=0.0.0.0 --listening-port=3478 & redis-server /etc/redis/redis.conf --protected-mode no & sleep 3 && uvicorn server:app --host 0.0.0.0 --port 8000"]