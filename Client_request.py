# server_client.py
import requests
from pprint import pprint

class ServerClient:
    def __init__(self, server_url="http://127.0.0.1:8000"):
        self.server_url = server_url

    def send_prediction_request(self, input_data):
        url = f"{self.server_url}/predict"
        response = requests.get(url, params=input_data)
        print("GET response:")
        print(response)
        print(response.status_code)
        pprint(response.json())
        return response.json() if response.status_code == 200 else None

if __name__ == "__main__":
    client = ServerClient()
    params = {"age": "youth", "income": "high", "student": "no", "credit_rating": "excellent"}
    client.send_prediction_request(params)