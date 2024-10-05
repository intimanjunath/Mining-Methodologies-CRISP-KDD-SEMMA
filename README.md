# CRISP - DM - Telco Customer Churn Prediction

## Overview
This project aims to predict customer churn for a telecom company using machine learning techniques. Churn prediction is critical in the telecom industry as it helps businesses identify customers who are likely to discontinue their services, allowing for preemptive retention strategies. I followed the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology to structure our analysis. This repository contains the complete code and documentation for downloading, analyzing, and modeling the data to predict churn.

## Dataset
The dataset used for this project is the **[Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)**, which contains information on customer demographics, account details, service usage, and churn status.
- **Rows**: 7,043 customers
- **Columns**: 21 features including demographics, service-related attributes, and churn status (target variable)

## Project Structure

The project is structured according to the following CRISP-DM steps:

### 1. Business Understanding

The goal is to identify customers who are likely to churn based on their usage patterns and demographic attributes. This allows the company to implement retention strategies like discounts or service upgrades.

### 2. Data Understanding

The dataset includes features such as:
- **Contract Type**: Month-to-month, one-year, two-year contracts.
- **Payment Method**: Electronic check, mailed check, credit card, bank transfer.
- **Service Usage**: Internet services, streaming, technical support, tenure, and monthly charges.

### 3. Data Preparation

Several steps were taken to prepare the data for modeling:
- **Handling Missing Values**: Missing values in the `TotalCharges` column were filled with the median.
- **Encoding Categorical Features**: Categorical variables were transformed using one-hot encoding.
- **Feature Scaling**: Numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) were scaled using `StandardScaler`.

### 4. Modeling

Multiple machine learning models were trained, including:
- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

After performing hyperparameter tuning, the **Random Forest Classifier** was selected as the final model with an accuracy of **80.2%** on the test set.

### 5. Evaluation

The model was evaluated using the following metrics:
- **Accuracy**: 80.2%
- **Precision**: 79%
- **Recall**: 74%
- **F1-Score**: 76%

We also created a confusion matrix to visualize correct and incorrect predictions.

### 6. Deployment

The trained model was saved using `joblib` and is ready for deployment. An API could be built to allow business users to input customer data and predict churn in real-time.

## Repository Content

- `churn_prediction.ipynb`: The main Jupyter notebook containing the full code for data exploration, preparation, modeling, and evaluation.
- `churn_model.pkl`: The trained machine learning model saved as a pickle file for deployment.
- `README.md`: This documentation file.
- `requirements.txt`: List of required Python libraries for running the project.

## Visualizations

We used several visualizations to understand the data and evaluate the model:
- **Churn Distribution**: A bar chart showing the proportion of churned and non-churned customers.
- **Monthly Charges vs Churn**: A boxplot showing how monthly charges vary between churn and non-churn customers.
- **Contract Type and Churn**: A bar plot showing churn rates across different contract types.
- **Confusion Matrix**: A matrix visualizing the performance of the Random Forest model.

## Installation and Usage
  Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/telco-churn-prediction.git
   ```
#KDD - 
