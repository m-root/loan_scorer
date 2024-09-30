# Loan Scoring App

This application provides an API for predicting loan performance based on loan and business details using a machine learning model. The app allows you to make predictions after training the model via the provided notebook.

## Prerequisites

Ensure that you have the following installed:

- [Poetry](https://python-poetry.org/docs/#installation) (for dependency management)
- Jupyter Notebook (for running the model training notebook)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/m-root/loan_scorer
   cd loan_scorer
   ```

2. **Install dependencies with Poetry**:
   Ensure you have `Poetry` installed, then run:
   ```bash
   poetry install
   ```

3. **Activate the virtual environment**:
   After installing the dependencies, activate the Poetry environment:
   ```bash
   poetry shell
   ```

## Training the Model

Before running the API, you must train the model using the provided notebook.

1. **Open the notebook**:
   Run the following command to start Jupyter Notebook and open the `model_training.ipynb`:
   ```bash
   jupyter notebook model_training.ipynb
   ```

2. **Run all the cells in the notebook**:
   Follow through the notebook to load data, preprocess, train the model, and save it. This will generate a model file (e.g., `loan_score_model.pkl`), which will be used later by the API.

3. **Verify the model**:
   Once the notebook completes, ensure the model is saved in the appropriate directory (e.g., `../model/loan_score_model.pkl`).

## Running the API

After training the model, you can now start the API:

1. **Run the FastAPI app**:
   In the same terminal, start the FastAPI app using `uvicorn`:
   ```bash
   uvicorn web_app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the API documentation**:
   You can view the API documentation by navigating to the following URL in your browser:
   ```
   http://localhost:8000/docs
   ```

## Making Predictions

You can now make predictions by sending a POST request to the `/predict` endpoint using tools like `Postman`, `Bruno` or `curl`. Here's an example payload:

```json
{
  "principal": 300000,
  "total_owing_at_issue": 345500,
  "sector": "Beauty Fashion",
  "amount": 45500,
  "repayment_cycles": 3,
  "total_payments": 345500,
  "payment_frequency": 2
}
```

---