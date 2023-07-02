import uvicorn
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
pickle_in = open("medical.pkl", "rb")
model = pickle.load(pickle_in)
class Medical(BaseModel):
    InscClaimAmtReimbursed: float
    IPAnnualReimbursementAmt: float
    IPAnnualDeductibleAmt: float
    OPAnnualReimbursementAmt: float
    OPAnnualDeductibleAmt: float
    Age: float
    DaysAdmitted: float
    TotalDiagnosis: float
    TotalProcedure: float
    EncounterType: int
    Gender: int
    Race: int
    RenalDiseaseIndicator: int
    ChronicCond_Alzheimer: int
    ChronicCond_Heartfailure: int
    ChronicCond_KidneyDisease: int
    ChronicCond_Cancer: int
    ChronicCond_ObstrPulmonary: int
    ChronicCond_Depression: int
    ChronicCond_Diabetes: int
    ChronicCond_IschemicHeart: int
    ChronicCond_Osteoporasis: int
    ChronicCond_rheumatoidarthritis: int
    ChronicCond_stroke: int
    IsDead: int

@ app.get('/')
def index():
    return {'message': 'Welcome to the Medical Insurance Prediction API'}

@app.post('/predict')
def predict_review(data:Medical):
    sentiment = ''
    received = data.dict()
    InscClaimAmtReimbursed = received['InscClaimAmtReimbursed']
    IPAnnualReimbursementAmt = received['IPAnnualReimbursementAmt']
    IPAnnualDeductibleAmt = received['IPAnnualDeductibleAmt']
    OPAnnualReimbursementAmt = received['OPAnnualReimbursementAmt']
    OPAnnualDeductibleAmt = received['OPAnnualDeductibleAmt']
    Age = received['Age']
    DaysAdmitted = received['DaysAdmitted']
    TotalDiagnosis = received['TotalDiagnosis']
    TotalProcedure = received['TotalProcedure']
    EncounterType = received['EncounterType']
    Gender = received['Gender']
    Race = received['Race']
    RenalDiseaseIndicator = received['RenalDiseaseIndicator']
    ChronicCond_Alzheimer = received['ChronicCond_Alzheimer']
    ChronicCond_Heartfailure = received['ChronicCond_Heartfailure']
    ChronicCond_KidneyDisease = received['ChronicCond_KidneyDisease']
    ChronicCond_Cancer = received['ChronicCond_Cancer']
    ChronicCond_ObstrPulmonary = received['ChronicCond_ObstrPulmonary']
    ChronicCond_Depression = received['ChronicCond_Depression']
    ChronicCond_Diabetes = received['ChronicCond_Diabetes']
    ChronicCond_IschemicHeart = received['ChronicCond_IschemicHeart']
    ChronicCond_Osteoporasis = received['ChronicCond_Osteoporasis']
    ChronicCond_rheumatoidarthritis = received['ChronicCond_rheumatoidarthritis']
    ChronicCond_stroke = received['ChronicCond_stroke']
    IsDead = received['IsDead']
    prediction = model.predict([[InscClaimAmtReimbursed, IPAnnualReimbursementAmt,
    IPAnnualDeductibleAmt, OPAnnualReimbursementAmt,
    OPAnnualDeductibleAmt, Age, DaysAdmitted,
    TotalDiagnosis, TotalProcedure, EncounterType, Gender, Race,
    RenalDiseaseIndicator, ChronicCond_Alzheimer,
    ChronicCond_Heartfailure, ChronicCond_KidneyDisease,
    ChronicCond_Cancer, ChronicCond_ObstrPulmonary,
    ChronicCond_Depression, ChronicCond_Diabetes,
    ChronicCond_IschemicHeart, ChronicCond_Osteoporasis,
    ChronicCond_rheumatoidarthritis, ChronicCond_stroke, IsDead]])
    result = prediction.tolist()[0]
    if result==0:
        sentiment = 'NOT FAULTY'
    if result ==1:
        sentiment = 'FAULTY'
    return {
        'prediction': result,
        'Sentiment Analysis':sentiment
    }

    
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)