# Credit Card Fraud Detection

## Overview
This project implements a **Credit Card Fraud Detection** system using **Logistic Regression**. The dataset consists of credit card transactions labeled as **legitimate (Class = 0)** or **fraudulent (Class = 1)**. The goal is to train a machine learning model to detect fraudulent transactions accurately.

## Dataset
- The dataset is loaded from `data.csv`.
- It contains transaction details including **Amount** and a **Class** label (0 for legitimate, 1 for fraud).
- It is highly imbalanced, with fraud transactions being a small fraction of the total.

## Steps in the Code

### 1. Data Loading & Exploration
- Load the dataset using `pandas`.
- Display the first and last few rows (`head()`, `tail()`).
- Get dataset shape and basic information (`info()`).
- Check for missing values (`isnull().sum()`).
- Analyze class distribution (`value_counts()`).

### 2. Data Preprocessing
- Separate **legit** and **fraudulent** transactions.
- Summarize the **Amount** column for both classes.
- Compute the mean values for different features grouped by `Class`.
- To balance the dataset, randomly sample 492 **legitimate** transactions to match the fraud count.
- Concatenate sampled legitimate transactions with all fraud transactions.

### 3. Splitting Features & Labels
- Separate features (`X`) and target variable (`Y`).
- Split data into **training (80%)** and **testing (20%)** sets using `train_test_split()`, ensuring class distribution remains balanced (`stratify=Y`).

### 4. Model Training
- Train a **Logistic Regression** model using `sklearn`.
- Fit the model on training data.

### 5. Model Evaluation
- Predict fraud on **training data** and compute accuracy.
- Predict fraud on **test data** and compute accuracy.

## Results
- The accuracy on both **training and test datasets** is printed.
- This helps assess the model’s ability to detect fraudulent transactions.



## Running the Project
Run the script in Jupyter Notebook or Google Colab to train and test the fraud detection model.

## Future Enhancements
- Try different machine learning models like **Random Forest** or **Neural Networks**.
- Use **feature scaling** to improve model performance.
- Implement **anomaly detection** techniques for better fraud detection.


