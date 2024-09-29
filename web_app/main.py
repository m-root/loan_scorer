from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained model (which includes the preprocessor)
try:
    model = joblib.load('../model/loan_score_model.pkl')  # The full pipeline (preprocessor + model)
except FileNotFoundError as e:
    raise HTTPException(
        status_code=500,
        detail="Model file not found."
    )
except Exception as e:
    raise HTTPException(
        status_code=500,
        detail="An error occurred while loading the model."
    )


# Define the input data structure using Pydantic
class LoanData(BaseModel):
    principal: float
    total_owing_at_issue: float
    sector: str
    amount: float
    repayment_cycles: int  # Number of actual repayment cycles
    total_payments: float
    payment_frequency: float


# Define the bins and labels for loan amount categorization
amount_bins = [
    0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000,
    450000, 500000,550000, 600000, 650000, 700000, 750000, 800000,
    850000, 900000, 950000, 1000000, float('inf')
]

amount_labels = [
    '0-50k', '50k-100k', '100k-150k', '150k-200k', '200k-250k',
    '250k-300k', '300k-350k', '350k-400k', '400k-450k', '450k-500k',
    '500k-550k', '550k-600k', '600k-650k', '650k-700k', '700k-750k',
    '750k-800k', '800k-850k', '850k-900k', '900k-950k', '950k-1M', '1M+'
]


# Function to categorize the loan amount based on the bins
def categorize_loan_amount(amount: float) -> str:
    for i, bin_val in enumerate(amount_bins):
        if amount <= bin_val:
            return amount_labels[i]
    return '1M+'  # Default if amount exceeds the last bin


# Function to categorize the repayment cycle based on the number of cycles
def categorize_repayment_cycle(cycles: int) -> str:
    if cycles <= 5:
        return 'Low (0-5)'
    elif 6 <= cycles <= 10:
        return 'Medium (6-10)'
    else:
        return 'High (11+)'


# Define the home route
@app.get("/")
def read_root():
    return {"message": "Loan Performance Prediction API"}


# Define the prediction route
@app.post("/predict")
def predict_loan_performance(data: LoanData):
    try:
        # Categorize the loan amount and total payments into the appropriate bin
        amount_bin = categorize_loan_amount(data.amount)
        total_payments_bin = categorize_loan_amount(data.total_payments)

        # Categorize the repayment cycle
        repayment_cycle_batch = categorize_repayment_cycle(data.repayment_cycles)

        # Prepare the input data as a DataFrame
        input_data = pd.DataFrame({
            'sector': [data.sector],
            'amount_range': [amount_bin],
            'total_payments_amount_range': [total_payments_bin],
            'repayment_cycle_batch': [repayment_cycle_batch],
            'payment_frequency': [data.payment_frequency]
        })

        # DEBUG: Print the input data before passing to model
        print(f"Input Data (Before Preprocessing):\n{input_data}")

        # Make prediction using the loaded pipeline (preprocessor + model)
        prediction = model.predict(input_data)

        # Return the prediction result
        return {"predicted_performance": int(prediction[0])}

    except ValueError as e:
        # Handle specific error related to value errors in input data
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data: {str(e)}"
        )

    except Exception as e:
        # Handle any other generic exception
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", reload=True)
