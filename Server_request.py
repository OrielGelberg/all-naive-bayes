# server_example.py
from fastapi import FastAPI, Request
import uvicorn
from NaiveBayesApp import NaiveBayesApp

app = FastAPI()

classifier = NaiveBayesApp()
classifier.load_and_train("Data.csv")

@app.get("/predict")
async def predict(request: Request):
    query = dict(request.query_params)
    print(query)
    prediction, method = classifier.predict_from_input(query)
    return {"target": prediction, "result": method}

if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)
