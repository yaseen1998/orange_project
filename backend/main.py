from fastapi import FastAPI

from project.handle_outliar import ScallPredict, read_csv
import joblib,os
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

class InputData(BaseModel):
    Credit_History_Age : float
    Credit_Mix : float
    Interest_Loan_Interaction : float
    Outstanding_Debt : float
    Total_Financial_Obligations : float

    
@app.post("/predict/")
async def Predict(input_data:InputData):
    dcit_data = input_data.model_dump()
    print(dcit_data)
    # check Cache
    if os.path.exists("random_forest.pkl"):
        random_forest = joblib.load('random_forest.pkl')
    else:
        random_forest = read_csv()
        joblib.dump(random_forest, "random_forest.pkl")
    predict = ScallPredict(dcit_data,random_forest)
    return {"message": predict}


origins = ["*"]  # Replace with your specific allowed origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)