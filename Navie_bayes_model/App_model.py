# server_example.py
from fastapi import FastAPI, Request
import uvicorn
from main import NaiveBayesApp


app = FastAPI()

csv_file = "Data.csv"
classifier = NaiveBayesApp()
classifier.load_csv(csv_file)
model, class_probabilities, target =classifier.train_model()
tester = classifier.tester()



@app.get("/model_to_predict")
async def model_to_predict(request: Request):
    return {"target": target,"model": model ,"class_probabilities": class_probabilities,"tester": tester}






if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)
