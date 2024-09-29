from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
try:
    model = joblib.load('loan_performance_model.pkl')
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
    applying_for_loan_number: int
    total_payments: float
    payment_frequency: int
    cash_yield_15_dpd: float
    amount: float


# Define the bins and labels for loan amount categorization
amount_bins = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000,
               550000, 600000, 650000, 700000, 750000, 800000, 850000, 900000, 950000, 1000000, float('inf')]
amount_labels = ['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k-250k',
                 '250k-300k', '300k-350k', '350k-400k', '400k-450k', '450k-500k',
                 '500k-550k', '550k-600k', '600k-650k', '650k-700k', '700k-750k',
                 '750k-800k', '800k-850k', '850k-900k', '900k-950k', '950k-1M', '1M+']


# Function to categorize the loan amount based on the bins
def categorize_loan_amount(amount: float) -> str:
    for i, bin_val in enumerate(amount_bins):
        if amount <= bin_val:
            return amount_labels[i]
    return '1M+'  # Default if amount exceeds the last bin


# Define the home route
@app.get("/")
def read_root():
    return {"message": "Loan Performance Prediction API"}


# Define the prediction route
@app.post("/predict")
def predict_loan_performance(data: LoanData):
    try:
        # Categorize the loan amount into the appropriate bin
        amount_bin = categorize_loan_amount(data.amount)

        # Prepare the data in the correct format (a 2D array)
        input_data = np.array(
            [
                [
                    data.principal,
                    data.total_owing_at_issue,
                    data.applying_for_loan_number,
                    data.total_payments,
                    data.payment_frequency,
                    data.cash_yield_15_dpd,
                    amount_bin
                ]
            ]
        )

        # Make prediction using the loaded model
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
