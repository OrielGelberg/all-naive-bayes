# server_example.py
import requests
from fastapi import FastAPI, Request
import uvicorn
from Classifier import PredictNaiveBayes as pnb

app = FastAPI()

# tester = classifier.tester()


dic_model = requests.get("http://con1:80/model_to_predict").json()
# dic_model = requests.get("http://127.0.0.1:8000/model_to_predict").json()

print(dic_model)




@app.get("/Classifier")
async def predict(request: Request):
    query = dict(request.query_params)
    print(request)
    print(query)
    method = pnb.predict(query, dic_model["model"], dic_model["class_probabilities"])
    return { "target":dic_model["target"],"result": method,"Accuracy percentage":dic_model["tester"] }


if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8001)
