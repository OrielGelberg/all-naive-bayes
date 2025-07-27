FROM python:3.11
LABEL maintainer="o@example.com"
WORKDIR /app

COPY . .

RUN pip install --upgrade pip -r requirements.txt


EXPOSE 80
CMD ["uvicorn", "Server_request:app", "--port", "80", "--host", "0.0.0.0"]
