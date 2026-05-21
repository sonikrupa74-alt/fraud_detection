from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = pickle.load(open("fraud_model.pkl", "rb"))

class FraudInput(BaseModel):
    time: float
    amount: float

@app.get("/")
def home():
    return {"Message": "Fraud Detection API is working"}

@app.post("/predict")
def predict(data: FraudInput):
    result = model.predict([[data.time, data.amount]])
    
    return {
        "time": data.time,
        "amount": data.amount,
        "FraudPrediction": int(result[0])
    }