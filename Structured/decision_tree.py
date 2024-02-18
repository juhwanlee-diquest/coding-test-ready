from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Adjusted example dataset creation function for binary classification
def create_example_dataset_for_classification(num_samples=1000):
    np.random.seed(0)  # Ensure reproducibility
    data = {
        'feature1': np.random.rand(num_samples),
        'feature2': np.random.rand(num_samples),
        'feature3': np.random.rand(num_samples),
        'target': np.random.randint(0, 2, num_samples)  # Binary classification target variable
    }
    df = pd.DataFrame(data)
    return df

# Create the dataset for binary classification
df = create_example_dataset_for_classification()

# Define the features and the target
X = df[['feature1', 'feature2', 'feature3']]  # Features
y = df['target']  # Target variable for classification

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree classifier
decision_tree_model = DecisionTreeClassifier()

# Fit the model to the training data
decision_tree_model.fit(X_train, y_train)

# Predict the target on the testing set
y_pred = decision_tree_model.predict(X_test)

# Calculate the accuracy to evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
