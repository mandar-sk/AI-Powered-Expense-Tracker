# Expense Manager

Categorization of expense from account statement

Website : [Expense Manager App](https://ai-powered-expense-tracker-mandar-sk.streamlit.app/)

## How to download HDFC Bank statements as CSV

* Login to HDFC netbanking
* Accounts -> Enquire (Left Pane) -> A/c statement upto 5 years
* Select Account number, period and format as Delimited
* Finally save as type as "All Files" and name it as filename.csv

## You can also use the provided demo dataset under folder named data
* demo_bank_dataset.csv - Contains financial transactions from a UPI enabled bank account
* demo_bank_dataset2.csv - Contains financial transactions of a Salary account

## Screenshots

![1705837112988](image/README/1705837112988.png)

![1705837125004](image/README/1705837125004.png)

![1705837132092](image/README/1705837132092.png)

***Data Privacy:** We dont store any of the users data*

You can also download the categorized expense output csv by clicking on `Download CSV` button.

***Note:** Currently only works for HDFC bank statements*

## 🖥️ Installation & Running Locally

### 🔧 Requirements
- Python 3.9
- Creat an environment

### ▶️ Run the app
streamlit run streamlit_app.py

### 📦 Install dependencies
```bash
pip install -r requirements.txt
