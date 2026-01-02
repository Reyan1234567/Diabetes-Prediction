from enum import Enum
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import joblib
from clipper import iqr_clipper    

logistic_regression_model=joblib.load('diabeto_LR.joblib')
decision_tree_classifier_model=joblib.load('diabeto_DC.joblib')
app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Gender(Enum):
    FEMALE="Female"
    MALE="Male"
    OTHER="Other"

class Ethnicity(Enum):
    WHITE="White"       
    HISPANIC="Hispanic"     
    BLACK="Black"        
    ASIAN="Asian"        
    OTHER="Other" 

class Education(Enum):
    HIGHSCHOOL="Highschool"
    GRADUATE="Graduate"
    POSTGRADUATE="Postgraduate"
    NO_FORMAL="No formal"

class IncomeLevel(Enum):
    MIDDLE="Middle"
    LOWER_MIDDLE="Lower-Middle"
    UPPER_MIDDLE="Upper-Middle"
    LOW="Low"
    HIGH="High"

class SmokingStatus(Enum):
    NEVER="Never"
    FORMER="Former"
    CURRENT="Current"

class EmploymentStatus(Enum):
    EMPLOYED="Employed"
    RETIRED="Retired"
    UNEMPLOYED="Unemployed"
    STUDENT="Student"

class Ones_Zeros(Enum):
    Zero=0
    One=1

class Input(BaseModel):
    age: int
    alcohol_consumption_per_week: int
    physical_activity_minutes_per_week: int
    diet_score: float
    sleep_hours_per_day: float
    screen_time_hours_per_day: float
    bmi: float
    systolic_bp: int
    diastolic_bp: int
    heart_rate: int
    cholesterol_total: int
    hdl_cholesterol: int
    triglycerides: int
    gender: Gender
    ethnicity: Ethnicity
    education_level: Education
    income_level: IncomeLevel
    smoking_status: SmokingStatus
    employment_status: EmploymentStatus
    family_history_diabetes: Ones_Zeros
    hypertension_history: Ones_Zeros
    cardiovascular_history: Ones_Zeros


@app.post('/diabeto/logistic')
def check_diabetes(input: Input):
    try:
        #change to a dictionary
        data_dict = input.model_dump()
        
        #loop and check if each value in the dictionary has a value attribute....
        #if it does then you replace the one that is there with the .attribute thing...
        for key, value in data_dict.items():
            if hasattr(value, 'value'):
                data_dict[key] = value.value

        #make a df out of the dictionary
        df = pd.DataFrame([data_dict])
        
        #predict
        prediction = logistic_regression_model.predict(df)
        
        return {"prediction": "Smoker" if int(prediction[0])==1 else "Non-Smoker"}
    except Exception as e:
        raise HttpException(
            status_code=400,
            detail=str(e)
        )


@app.post('/diabeto/tree')
def check_diabetes(input: Input):
    try:
        data_dict = input.model_dump()
        
        for key, value in data_dict.items():
            if hasattr(value, 'value'):
                data_dict[key] = value.value

        df = pd.DataFrame([data_dict])
        
        prediction = decision_tree_classifier_model.predict(df)
        
        return {"prediction": "Smoker" if int(prediction[0])==1 else "Non-Smoker"}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

