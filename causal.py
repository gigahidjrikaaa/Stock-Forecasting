### PYTHON PROGRAM FOR PREDICTING BLACKROCK INC STOCK PRICES ###
### Quantitative Methods: Regression Analysis ###

import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt
import yfinance as yf

# Load the data
stock = 'BLK'

# BlackRock Ticker Data
blk_data = yf.Ticker(stock)
blk_df = blk_data.history(period='1d', start='2010-1-1', end='2024-1-1')

# Select features (replace 'Open' with your desired feature)
data = blk_df.filter(['Close', 'Open'])  # Replace 'Open' with your feature

# Separate features (X) and target variable (y)
X = data.iloc[:, :-1].values  # All columns except the last
y = data.iloc[:, -1].values  # Last column (target)

# From here, we can use linear regression libraries like scikit-learn
from sklearn.linear_model import LinearRegression

# Create a linear regression model
model = LinearRegression()

# Train the model on the data
model.fit(X, y)

# Get the coefficients (slope and intercept)
coefficients = model.coef_
intercept = model.intercept_

# Print the coefficients
print("Coefficients:", coefficients)
print("Intercept:", intercept)

# Predict the target variable
y_pred = model.predict(X)

# Print the predicted values
print("Predicted values:", y_pred)

# Calculate the R-squared value
r_squared = model.score(X, y)
print("R-squared:", r_squared)

# Calculate the Mean Squared Error (MSE)
mse = np.mean((y - y_pred) ** 2)
print("Mean Squared Error:", mse)

# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# Plot the data and regression line (optional)
# You can plot the actual closing prices (y) vs opening prices (X[0]) and the predicted line using coefficients and intercept
plt.scatter(X[:, 0], y)  # Assuming opening price is the first feature
plt.plot(X[:, 0], y_pred, color='red')
plt.xlabel('Opening Price')
plt.ylabel('Closing Price')
plt.title('Linear Regression: Closing Price vs Opening Price')
plt.show()
