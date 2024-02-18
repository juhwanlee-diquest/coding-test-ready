from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Adjusted example dataset creation function to fit linear regression
def create_example_dataset_for_regression(num_samples=1000):
    np.random.seed(0)  # Ensure reproducibility
    data = {
        'feature1': np.random.rand(num_samples),
        'feature2': np.random.rand(num_samples),
        'feature3': np.random.rand(num_samples)  # Predicting this as a continuous outcome
    }
    df = pd.DataFrame(data)
    return df

# Create the dataset for regression
df = create_example_dataset_for_regression()

# Define the features and the target
X = df[['feature1', 'feature2']]  # Features
y = df['feature3']  # Target variable to predict

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
linear_reg_model = LinearRegression()

# Fit the model to the training data
linear_reg_model.fit(X_train, y_train)

# Predict the target on the testing set
y_pred = linear_reg_model.predict(X_test)

# Calculate the Mean Squared Error (MSE) to evaluate the model
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
