# Loan-Default-Prediction

This project uses **Keras** to build a **deep learning model** for predicting whether a borrower will **repay a loan or default** based on historical LendingClub data.
The dataset contains multiple features about borrowers and their loans. Our goal is to train a **classification model** that predicts the loan_status (repayment vs. default).

## The Data

We use a **subset of the LendingClub dataset** (preprocessed with extra features).
The dataset includes information about loans issued from 2007-2010, including borrower financials, loan details, and credit history.

Key features include:

- loan_amnt: Loan amount requested
- term: Number of monthly payments (36 or 60 months)
- int_rate: Interest rate of the loan
- installment: Monthly installment owed by borrower
- grade, sub_grade: LendingClub assigned loan grade/subgrade
- emp_title: Borrower’s job title
- emp_length: Employment length in years
- home_ownership: Borrower’s home ownership status (RENT, OWN, MORTGAGE, OTHER)
- annual_inc: Self-reported annual income
- verification_status: Income verification status
- issue_d: Month the loan was funded
- loan_status: Target variable (repay vs. default)
- purpose: Purpose of the loan
- dti: Debt-to-income ratio
- revol_bal: Total revolving balance
- revol_util: Revolving line utilization
- open_acc: Number of open credit lines
- pub_rec: Number of derogatory public records
- mort_acc: Number of mortgage accounts
- pub_rec_bankruptcies: Number of public record bankruptcies

## Objectives

- Explore and preprocess LendingClub data for deep learning
- Encode categorical and numerical features for neural network input
- Build a **classification model using Keras** to predict loan repayment
- Evaluate model performance using metrics like **accuracy, precision, recall, and F1-score**
- Optimize model architecture and hyperparameters

## Tech Stack

- Python
- Keras & TensorFlow – Deep learning framework
- Pandas, NumPy – Data preprocessing and manipulation
- Scikit-learn – Train/test split, preprocessing, and evaluation metrics
- Matplotlib, Seaborn – Data visualization
- Jupyter Notebook / PyCharm – Development environment

  ## How to Run
1. Clone the repository:  
```bash
     git clone <your-repo-url>
```
2. Create and activate a virtual environment (Recommended):
```bash
     python -m venv venv
```
   - Linux/macOS:
     ```bash
          source venv/bin/activate
     ```
   - Windows:
     ```bash
          venv\Scripts\activate
     ```
3. Install dependencies:
 ```bash
      pip install <name of libraries>
```
3. Run the main Python script:
 ```bash
      python LoanDefaultPrediction.py
```

