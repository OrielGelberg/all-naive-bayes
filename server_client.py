# server_client.py
"""
Client for receiving data from server
"""
import requests
import json


class ServerClient:
    def __init__(self, server_url="http://127.0.0.1:8000"):
        """Initialize server client"""
        self.server_url = server_url

    def get_prediction_data(self, endpoint="/predict", params=None):
        """Get data from server for prediction"""
        try:
            url = f"{self.server_url}{endpoint}"
            
            if params:
                response = requests.get(url, params=params)
            else:
                response = requests.get(url)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Server error: {response.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            print("Could not connect to server")
            return None
        except Exception as e:
            print(f"Error communicating with server: {e}")
            return None

    def send_prediction_result(self, endpoint="/result", data=None):
        """Send prediction result back to server"""
        try:
            url = f"{self.server_url}{endpoint}"
            response = requests.post(url, json=data)
            
            if response.status_code == 200:
                return True
            else:
                print(f"Failed to send result: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error sending result: {e}")
            return False

    def is_server_available(self):
        """Check if server is available"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False