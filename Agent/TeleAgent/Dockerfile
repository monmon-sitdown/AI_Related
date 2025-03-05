FROM python:3.11.4

WORKDIR /aiserver

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "server.py"]