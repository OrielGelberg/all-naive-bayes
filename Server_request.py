# server_example.py
from fastapi import FastAPI, Request
import uvicorn
from main import NaiveBayesApp
from Predict import PredictNaiveBayes as pnb

app = FastAPI()

csv_file = "Data.csv"
classifier = NaiveBayesApp()
classifier.load_csv(csv_file)
model, class_probabilities, target =classifier.train_model()
tester = classifier.tester()



@app.get("/predict")
async def predict(request: Request):
    query = dict(request.query_params)
    print(request)
    print(query)
    method = pnb.predict(query,model, class_probabilities)
    return {"target": target, "result": method , "Accuracy percentage": tester}

if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)
