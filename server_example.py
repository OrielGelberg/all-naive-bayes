# server_example.py
"""
Example server for testing the client integration
"""
from fastapi import FastAPI, Request
import uvicorn


app = FastAPI()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/predict")
async def get_prediction_data(request: Request):
    """Return data for prediction"""
    query_params = dict(request.query_params)
    
    # If query parameters provided, return them
    if query_params:
        return query_params
    
    # Otherwise return sample data
    sample_data = {
        "age": "youth",
        "income": "medium", 
        "student": "yes",
        "credit_rating": "fair"
    }
    return sample_data


@app.post("/result")
async def receive_result(data: dict):
    """Receive prediction result"""
    print(f"Received prediction result: {data}")
    return {"status": "received", "data": data}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)