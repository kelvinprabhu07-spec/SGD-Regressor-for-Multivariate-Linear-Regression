# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries such as NumPy, Pandas and Scikit-learn.
2.Create a dataset containing house features such as area, number of bedrooms and number of occupants.
3.Initialize the SGD Regressor model and train it using the input features and house prices.
4.Predict the house price and number of occupants for new input data and display the results. 

## Program:
```py
# Program to implement Multivariate Linear Regression
# using SGD Regressor.
# Developed by:
# RegisterNumber:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# Input data
# Features: Area, Bedrooms, House Age
X = np.array([
    [800, 1, 10],
    [1000, 2, 8],
    [1200, 2, 5],
    [1500, 3, 6],
    [1800, 3, 3],
    [2000, 4, 2],
    [900, 1, 12],
    [1300, 2, 4],
    [1600, 3, 5],
    [2200, 4, 1]
])

# Output: House Price (Lakhs), Occupants
Y = np.array([
    [25, 2],
    [32, 3],
    [38, 3],
    [48, 4],
    [58, 5],
    [65, 5],
    [28, 2],
    [42, 3],
    [52, 4],
    [72, 6]
])

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create SGD Regressor model
model = MultiOutputRegressor(
    SGDRegressor(max_iter=10000, tol=0.001, random_state=42)
)

# Train the model
model.fit(X_scaled, Y)

# New house details
new_house = np.array([[1400, 3, 4]])

# Scale input
new_house_scaled = scaler.transform(new_house)

# Predict output
prediction = model.predict(new_house_scaled)

print("Predicted House Price:", round(prediction[0][0], 2), "Lakhs")
print("Predicted Number of Occupants:", round(prediction[0][1]))

# Plot graph
plt.scatter(X[:, 0], Y[:, 0], color='blue', label='Actual Price')

plt.scatter(
    new_house[0][0],
    prediction[0][0],
    color='red',
    label='Predicted Price'
)

plt.xlabel("House Area")
plt.ylabel("House Price (Lakhs)")
plt.title("House Price Prediction using SGD Regressor")
plt.legend()
plt.grid()
plt.show()
```

## Output:
~~~
Predicted House Price: 48.70 Lakhs
Predicted Number of Occupants: 4
~~~
## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
