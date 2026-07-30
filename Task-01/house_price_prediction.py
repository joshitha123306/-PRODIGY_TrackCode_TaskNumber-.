# House Price Prediction using Linear Regression

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("train.csv")
# data = pd.read_csv("train (1).csv")   # Use this if your file is named train (1).csv

# Select input features
X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]

# Select target variable
y = data["SalePrice"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Model Performance")
print("-------------------------")
print("Mean Absolute Error :", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error  :", mean_squared_error(y_test, y_pred))
print("R2 Score            :", r2_score(y_test, y_pred))

# Predict a new house price
new_house = pd.DataFrame({
    "GrLivArea": [2000],
    "BedroomAbvGr": [3],
    "FullBath": [2]
})

predicted_price = model.predict(new_house)

print("\nNew House Prediction")
print("-------------------------")
print("Square Footage :", 2000)
print("Bedrooms       :", 3)
print("Bathrooms      :", 2)
print("Predicted Price:", predicted_price[0])