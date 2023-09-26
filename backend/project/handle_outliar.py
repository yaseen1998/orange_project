import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from project.apply_model import Train_model

def custom_encoding(y):
    mapping = {"Standard":1,"Good":2,"Poor":0,"Bad":0,'NM':0,"No":1,"Yes":2}
    y = y.map(mapping)
    return y


def out_liar(column_data):
    Q1 = column_data.quantile(0.25)
    Q3 = column_data.quantile(0.75)
    IQR = Q3 - Q1
    lowerBand = Q1 - 1.5 * IQR
    upperBand = Q3 + 1.5 * IQR
    column_data = column_data.apply(lambda x: upperBand if x > upperBand else lowerBand if x < lowerBand else x)
    return column_data

def feature_outliar(x):
    features=[ "Outstanding_Debt", "Interest_Loan_Interaction","Total_Financial_Obligations" ]   
    for feature in features:
        x[feature] = out_liar(x[feature])
    return x

def Standardization(x):
    features=[
    "Credit_History_Age",
    "Credit_Mix",
    "Interest_Loan_Interaction",
    "Outstanding_Debt", 
    "Total_Financial_Obligations"
]
    scaler = StandardScaler()
    X = x[features]
    scaled_data = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")
    
    return scaled_data



def over_sample(scaled_data,y):
    x_train, x_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2, random_state=42)
    sm = SMOTE(k_neighbors=5)
    x_train_sm, y_train_sm = sm.fit_resample(x_train, y_train)
    return x_train_sm,  y_train_sm


def read_csv():
    df = pd.read_csv("csv/credit_score_clean_extraction.csv")
    x = df.drop(['Credit_Score'],axis=1)
    y = custom_encoding(df["Credit_Score"])
    x = feature_outliar(x)
    standard_scale = Standardization(x)
    x_train_sm, y_train_sm = over_sample(standard_scale,y)
    return Train_model(x_train_sm,  y_train_sm,)
    
    
def ScallPredict(input_data,model):
    new_predict = pd.DataFrame(input_data,index=[0])
    scaler = joblib.load("scaler.pkl")
    scaled_data = scaler.transform(new_predict)
    predict = model.predict(scaled_data)
    credit_score = {0:"Poor",1:"Standard",2:"Good"}
    credict_score_predict = credit_score[predict[0]]
    return credict_score_predict