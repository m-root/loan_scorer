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


# Define the home route
@app.get("/")
def read_root():
    return {"message": "Loan Performance Prediction API"}


# Define the prediction route
@app.post("/predict")
def predict_loan_performance(data: LoanData):
    try:
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
                    data.amount
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
