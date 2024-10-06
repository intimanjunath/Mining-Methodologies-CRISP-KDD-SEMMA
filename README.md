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

Colab Notebook : https://github.com/intimanjunath/Mining-Methodologies-CRISP-KDD-SEMMA/blob/main/CRISP_DM__TelcoCustomer_Churn_Prediction.ipynb

Medium Article Link: https://medium.com/@manjunatha.inti/predicting-customer-churn-with-machine-learning-a-telco-case-study-7a7246d2d411

Research paper : https://github.com/intimanjunath/Mining-Methodologies-CRISP-KDD-SEMMA/blob/main/CRSIP_DM_paper.pdf

latex format Code : https://github.com/intimanjunath/Mining-Methodologies-CRISP-KDD-SEMMA/blob/main/CRISPDM_latex_format.tex

---
# KDD -  Network Intrusion Detection Using the KDD Process on UNSW-NB15 Dataset

## Project Overview
This project aims to detect network intrusions by analyzing the UNSW-NB15 dataset. The dataset contains a mix of normal and malicious traffic, making it ideal for demonstrating classification techniques in a security-focused data mining project. The workflow follows the KDD process, from data selection to model deployment.

## Dataset
The **UNSW-NB15** dataset was created to address limitations in older network datasets and provides a more realistic representation of modern network traffic. It contains normal and attack behaviors, including nine different types of attacks such as DoS, backdoor, and worms.

- **Dataset Link**: [UNSW-NB15 Dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)

## KDD Process Steps
### 1. Selection
We selected the **UNSW-NB15** dataset, which includes over 2.5 million records of network traffic data. The dataset contains labeled instances of both normal and attack traffic. The task is to classify network traffic as either normal or an attack.

### 2. Preprocessing
- **Handling Missing Data**: We filled missing values using the column mean for numeric data.
- **Encoding Categorical Features**: Columns like `protocol` were encoded into numerical values using label encoding.
- **Standardization**: Numeric features were standardized using `StandardScaler` to ensure that the model treats features on a comparable scale.

### 3. Transformation
Feature engineering was performed to create new insights from the dataset:
- **Data Flow Duration**: Created a new feature by subtracting `sbytes` (source bytes) from `dbytes` (destination bytes).
- **Dropped Irrelevant Columns**: Removed IP addresses (`srcip`, `dstip`) as they don't contribute to traffic classification.

### 4. Data Mining
We implemented a **Random Forest Classifier** to identify malicious network traffic. The model was trained on 70% of the data, and the remaining 30% was used for testing.

### 5. Evaluation
The model was evaluated using several metrics:
- **Accuracy**: 95%+ accuracy on test data.
- **ROC-AUC Score**: Used to assess model performance across different classification thresholds.
- **Confusion Matrix**: Visualized the trade-off between false positives and true negatives.

### 6. Deployment
The trained model was saved using `joblib` for future use. We deployed it via a **Flask API**, allowing real-time network traffic classification.

## Installation

To run the project on your local machine, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/network-intrusion-detection.git
    cd network-intrusion-detection
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Jupyter Notebook**:
    Open `network_intrusion_detection.ipynb` in Google Colab or Jupyter and run all cells.

Colab Notebook file : https://github.com/intimanjunath/Mining-Methodologies-CRISP-KDD-SEMMA/blob/main/KDD.ipynb

Medium Article Link: https://medium.com/@manjunatha.inti/building-a-kdd-driven-intrusion-detection-system-using-the-unsw-nb15-dataset-5540e60d099d

Research paper : https://github.com/intimanjunath/Mining-Methodologies-CRISP-KDD-SEMMA/blob/main/KDD_paper.pdf

latex format Code : https://github.com/intimanjunath/Mining-Methodologies-CRISP-KDD-SEMMA/blob/main/KDD_latex_format.tex

---

# SEMMA Process -  Credit Card Fraud Detection

This repository demonstrates the application of the SEMMA (Sample, Explore, Modify, Model, Assess) methodology to build a credit card fraud detection system using the **Credit Card Fraud Detection dataset** from Kaggle.

## Dataset
The dataset used is the **Credit Card Fraud Detection** dataset, which can be downloaded from Kaggle [here](https://www.kaggle.com/mlg-ulb/creditcardfraud).

- **Fraudulent Transactions**: Transactions classified as fraud (Class = 1)
- **Non-Fraudulent Transactions**: Transactions classified as legitimate (Class = 0)

## SEMMA Process

### 1. Sample
The first step in the SEMMA process is to sample the data. We sampled 100% of the dataset to ensure that the entire dataset is used for analysis. The dataset contains the following columns:

- `Time`: The time of the transaction
- `Amount`: The amount of the transaction
- `Class`: The target variable (0 = non-fraudulent, 1 = fraudulent)

### 2. Explore
We performed Exploratory Data Analysis (EDA) to gain insights into the dataset, such as:

- **Class distribution**: Visualization of fraudulent vs. non-fraudulent transactions.
- **Correlation Analysis**: Checking correlations between features, which helps understand relationships and feature selection.

### 3. Modify
The dataset was preprocessed by scaling the `Amount` and `Time` features using `StandardScaler`. Feature scaling ensures that the machine learning algorithms, especially those relying on distances (like RandomForest), perform optimally. 

Additionally, we dropped the target variable `Class` from the feature set (`X`) to ensure that the model doesn't have access to it during training.

### 4. Model
We used a **RandomForestClassifier** for modeling the data. Random forests are an ensemble learning method that builds multiple decision trees and merges them together to get a more accurate and stable prediction. The model was trained using the training data (80%) and tested on the testing data (20%).

### 5. Assess
After training the model, we evaluate its performance using various metrics:

- **Confusion Matrix**: This table helps visualize the model's prediction performance by comparing the predicted values to the actual values.
- **Classification Report**: This report includes precision, recall, F1-score, and accuracy for each class (fraud and non-fraud).
- **ROC Curve and AUC Score**: The ROC curve shows the model's ability to distinguish between classes, and the AUC score summarizes the overall performance.

### Deployment
After successfully training and assessing the model, the next step is deploying it. We save the trained model using the `pickle` library so that it can be loaded and used for predictions without retraining. Below is the code to save and deploy the model using Flask.

Colab Notebook file : https://github.com/intimanjunath/Mining-Methodologies-CRISP-KDD-SEMMA/blob/main/Semma.ipynb

Medium Article Link: https://medium.com/@manjunatha.inti/building-a-credit-card-fraud-detection-system-using-the-semma-process-f896bea44057

Research paper : https://github.com/intimanjunath/Mining-Methodologies-CRISP-KDD-SEMMA/blob/main/SEMMA_paper.pdf

latex format Code : https://github.com/intimanjunath/Mining-Methodologies-CRISP-KDD-SEMMA/blob/main/SEMMA_latex_format.tex
