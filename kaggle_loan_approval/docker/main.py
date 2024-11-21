from joblib import load  # To load the model
from fastapi import FastAPI, HTTPException
from os import getenv
from pydantic import BaseModel, Field, model_validator, ValidationError  # For request validation
from pandas import DataFrame
from numpy import nan
from typing import Literal

# Define the input schema with reality checks
class PredictionRequest(BaseModel):
    id: int
    person_age: float = Field(..., gt=0, lt=150, description="Age must be between 0 and 150")
    person_income: float = Field(..., ge=0, description="Income must be non-negative")
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: float = Field(..., ge=0, lt=100, description="Employment length must be non-negative")
    loan_intent: Literal["EDUCATION", "HOME", "MEDICAL", "PERSONAL", "VENTURE", "AUTO"]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(..., gt=0, description="Loan amount must be greater than 0")
    loan_int_rate: float = Field(..., gt=0, lt=100, description="Interest rate must be between 0 and 100")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Loan percent of income must be between 0 and 1")
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(..., ge=0, description="Credit history length must be non-negative")

    # Apply custom filters to fields using @model_validator
    @model_validator(mode="before")
    def apply_filters(cls, values):
        # Define filter conditions here with 'st' meaning smaller than
        filter_conditions = {
            'person_age': (110, 'st'),  # person_age must be less than 110
            'loan_percent_income': (0.8, 'st'),  # loan_percent_income must be less than 0.8
            'person_income': (1200001, 'st'),  # person_income must be less than 1,200,001
            'person_emp_length': (100, 'st')  # person_emp_length must be less than 100
        }

        # Loop through the filter conditions and apply them to the values
        for field, (limit_value, condition) in filter_conditions.items():
            if field in values:
                value = values.get(field)
                
                if condition == 'st' and value >= limit_value:  # 'st' means less than
                    raise ValueError(f"{field} must be smaller than {limit_value}")
                elif condition == 'gt' and value <= limit_value:  # 'gt' means greater than
                    raise ValueError(f"{field} must be greater than {limit_value}")
        
        return values

# Define the output schema
class PredictionResponse(BaseModel):
    prediction: float

# Load the model (preloaded into memory)
model = load(getenv("MODEL_PATH"))

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the ML model API!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(input: PredictionRequest):
    try:
        # Convert input to the correct format expected by the model
        input_data = DataFrame([{
            'id': input.id,
            'person_age': input.person_age,
            'person_income': input.person_income,
            'person_home_ownership': input.person_home_ownership,
            'person_emp_length': input.person_emp_length,
            'loan_intent': input.loan_intent,
            'loan_grade': input.loan_grade,
            'loan_amnt': input.loan_amnt,
            'loan_int_rate': input.loan_int_rate,
            'loan_percent_income': input.loan_percent_income,
            'cb_person_default_on_file': input.cb_person_default_on_file,
            'cb_person_cred_hist_length': input.cb_person_cred_hist_length
        }])
        
        input_data.drop(['id'], axis=1, inplace=True)                  # Drop irrelevant columns as defined in drop_cols ..
        input_data = add_interaction_feature_raw(input_data, 'loan_int_rate', 'loan_grade')
        input_data = add_interaction_feature_number(input_data, 'loan_amnt', 'person_income', '/')
        input_data = add_interaction_feature_number(input_data, 'loan_amnt_person_income_division', 'loan_percent_income', '-',  True, False)

        cat_cols_raw =  [col for col in input_data.select_dtypes(include=['object', 'category']).columns.tolist()]

        # Cast all object columns to category data type (useful for models)
        for col in cat_cols_raw:
            input_data[col] = input_data[col].astype('category')

        # Get the prediction
        prediction = model.predict(input_data)[0]
        
        return PredictionResponse(prediction=prediction)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


def add_interaction_feature_number(df, col1, col2, operation, drop1=False, drop2=False):
    if operation == '+':
        feature = col1 + '_' + col2 + '_sum'
        df[feature] = df[col1] + df[col2]
    elif operation == '-':
        feature = col1 + '_' + col2 + '_diff'
        df[feature] = df[col1] - df[col2]
    elif operation == '*':
        feature = col1 + '_' + col2 + '_product'
        df[feature] = df[col1] * df[col2]
    elif operation == '/':
        feature = col1 + '_' + col2 + '_division'
        # Prevent division by zero
        df[feature] = df[col1] / df[col2].replace(0, nan)
    
    # Drop original columns if requested
    if drop1:
        df.drop(col1, axis=1, inplace=True)
    if drop2:
        df.drop(col2, axis=1, inplace=True)

    return df

def add_interaction_feature_raw(df, num_col, cat_col, drop1=False, drop2=False):
    df[cat_col + '_' + num_col + '_grouped'] = df.groupby(cat_col, observed=True)[num_col].transform('median')

    # Drop original columns if requested
    if drop1:
        df.drop(cat_col, axis=1, inplace=True)
    if drop2:
        df.drop(num_col, axis=1, inplace=True)

    return df
