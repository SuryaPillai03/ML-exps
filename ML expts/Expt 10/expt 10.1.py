import matplotlib.pyplot as plt  # This line is added to import the pyplot module
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Step 3: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 5: Create the XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
# Step 6: Train the model
xgb_model.fit(X_train, y_train)
# Step 7: Make predictions
y_pred = xgb_model.predict(X_test)
# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of XGBoost model: {accuracy * 100:.2f}%")
print("\nClassification Report By Surya:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
# Step 9: Feature Importance Plot
xgb.plot_importance(xgb_model)
plt.title('Feature Importance of XGBoost Model By Surya 211P015') # plt is now defined and can be used
plt.show()
