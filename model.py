# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data by scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Train a Decision Tree model
dt = DecisionTreeClassifier()
dt.fit(X_train_scaled, y_train)

# Train a Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)

# Evaluate the Logistic Regression model
y_pred_logreg = logreg.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_logreg))

# Evaluate the Decision Tree model
y_pred_dt = dt.predict(X_test_scaled)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))

# Evaluate the Random Forest model
y_pred_rf = rf.predict(X_test_scaled)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# Make predictions using the Logistic Regression model
new_data = [[5.1, 3.5, 1.4, 0.2]]
new_data_scaled = scaler.transform(new_data)
prediction = logreg.predict(new_data_scaled)
print("Logistic Regression Prediction:", iris.target_names[prediction[0]])

# Make predictions using the Decision Tree model
prediction = dt.predict(new_data_scaled)
print("Decision Tree Prediction:", iris.target_names[prediction[0]])

# Make predictions using the Random Forest model
prediction = rf.predict(new_data_scaled)
print("Random Forest Prediction:", iris.target_names[prediction[0]]) 

# Compare the performance of different models
models = [logreg, dt, rf]
model_names = ["Logistic Regression", "Decision Tree", "Random Forest"]
accuracies = [accuracy_score(y_test, y_pred_logreg), accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_rf)]

for i in range(len(models)):
    print(f"{model_names[i]} Accuracy: {accuracies[i]}")

# Deploy the model as a simple web app using Flask
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    new_data = [[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]]
    new_data_scaled = scaler.transform(new_data)
    prediction = logreg.predict(new_data_scaled)
    return jsonify({'prediction': iris.target_names[prediction[0]]})

if __name__ == '__main__':
    app.run(debug=True)