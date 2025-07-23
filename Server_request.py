# server_example.py
from fastapi import FastAPI, Request
import uvicorn
from NaiveBayesApp import NaiveBayesApp
from Predict import PredictNaiveBayes as pnb

app = FastAPI()

csv_file = "Data.csv"
classifier = NaiveBayesApp()
classifier.load_and_train(csv_file)
model, class_probabilities =classifier.train_model()
tester = classifier.tester()



@app.get("/predict")
async def predict(request: Request):
    query = dict(request.query_params)
    print(request)
    print(query)
    prediction, method = pnb.predict(query,model, class_probabilities)
    return {"target": prediction, "result": method , "אחוזי דיוק": tester}

if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)
