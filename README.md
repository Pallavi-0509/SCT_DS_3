# Decision Tree Classification on Bank Marketing Dataset

## Overview
This project uses a Decision Tree Classifier to predict whether a bank client will subscribe to a term deposit.  
The dataset is cleaned, encoded, split into training and testing sets, and then used to train a model.  
Finally, the accuracy, classification report, and decision tree visualization are displayed.

## Dataset
File used: bank-full.csv  
Source: UCI Bank Marketing Dataset  

The dataset contains information about clients and their responses to marketing campaigns.  
It includes columns such as age, job, marital status, education, balance, housing loan, personal loan, and others.  
The target column is y, which indicates if the client subscribed (yes or no).

## Steps in the Code

1. Import libraries  
   pandas for data handling  
   scikit-learn for model building and evaluation  
   matplotlib for visualization  

2. Load the dataset  
   data = pd.read_csv("bank-full.csv", sep=';')

3. Encode categorical columns  
   Converts text columns into numeric codes using category encoding.

4. Encode the target column  
   yes is mapped to 1 and no is mapped to 0.

5. Split the dataset  
   Data is split into 80 percent for training and 20 percent for testing.

6. Train the Decision Tree Classifier  
   The model is created using DecisionTreeClassifier with a maximum depth of 4.

7. Make predictions and evaluate the model  
   The model predictions are compared with actual values to calculate accuracy and generate a classification report.

8. Visualize the decision tree  
   A tree plot is shown using matplotlib that displays how decisions are made.

## Outputs
- Accuracy score of the model  
- Classification report with precision, recall, and f1-score  
- Decision tree diagram

## Model Details
Algorithm: Decision Tree Classifier  
Criterion: Gini Impurity  
Max Depth: 4  
Random State: 42  

## Requirements
Install Python libraries, pandas scikit-learn matplotlib before running the code.


## How to Run
1. Place the bank-full.csv file in your working directory.  
2. Update the file path in the code if needed.  
3. Run the Python script 
4. Check the accuracy and classification report in the terminal.  
5. The decision tree visualization will appear in a separate window.
