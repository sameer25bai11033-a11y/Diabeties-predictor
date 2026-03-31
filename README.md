# Diabeties-predictor
Diabetes Prediction System A Machine Learning-based classification system designed to predict the likelihood of diabetes in patients based on medical diagnostic measurements. This project utilizes Logistic Regression and comprehensive data preprocessing to achieve high-accuracy predictions.
Diabetes Prediction System
A Machine Learning-based classification system designed to predict the likelihood of diabetes in patients based on medical diagnostic measurements. This project utilizes Logistic Regression and comprehensive data preprocessing to achieve high-accuracy predictions.

 Overview
The goal of this project is to build a predictive model that can identify whether a patient has diabetes based on specific health metrics (e.g., Glucose levels, BMI, Age, etc.). It includes:

Exploratory Data Analysis (EDA): Visualizing feature correlations.

Data Preprocessing: Handling missing values and feature scaling.

Machine Learning: Training a Logistic Regression model.

Evaluation: Detailed metrics including Accuracy, Confusion Matrix, and Classification Report.

 Tech Stack
Language: Python 3.x

Libraries:

Pandas: Data manipulation and analysis.

NumPy: Numerical computing.

Matplotlib & Seaborn: Data visualization.

Scikit-Learn: Machine learning modeling and preprocessing.

 Dataset
The model expects a diabetes.csv file (typically the Pima Indians Diabetes Dataset) containing the following features:

Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age, and Outcome (Target).


Key Workflow
Data Loading: Loads the dataset and checks for null values.

Visualization: Generates a Heatmap to identify correlations between health factors and diabetes outcomes.

Preprocessing: * Splits data into Training (80%) and Testing (20%) sets.

Applies StandardScaler to ensure all features are on the same scale for better model performance.

Modeling: Fits a Logistic Regression classifier to the training data.

Evaluation: Outputs the final accuracy and a detailed classification report (Precision, Recall, F1-Score).

📝 Results
The model typically achieves a solid baseline accuracy. The generated heatmap helps in understanding which features (like Glucose or BMI) have the highest impact on the prediction.
