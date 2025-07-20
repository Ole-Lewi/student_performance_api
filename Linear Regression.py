#PREDICTING STUDENT SCORES BASED ON HOURS STUDIED

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Scores': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
}

df = pd.DataFrame(data)

# Features and target variable
X = df[['Hours']]  #double brackets to keep X as a 2D DataFrame
y = df['Scores']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

print("r2 score:", r2_score(y_pred, y_test))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predicting the score for a student who studies for 7.5 hours
hours_studied = np.array([[7.5]])
predicted_score = model.predict(hours_studied)
print(f"Predicted score for a student who studies for 7.5 hours: {predicted_score[0]}")

import joblib

# Save the model to a file
joblib.dump(model, 'Linear Regression.pkl')

print("Model saved as 'Linear Regression.pkl'")