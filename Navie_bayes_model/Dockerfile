FROM python:3.11
LABEL maintainer="o@example.com"
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip -r requirements.txt

COPY . .

EXPOSE 80
CMD ["uvicorn", "App_model:app", "--port", "80", "--host", "0.0.0.0"]
