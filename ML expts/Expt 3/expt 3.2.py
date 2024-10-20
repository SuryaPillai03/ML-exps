import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Name: Surya
# UIN: 211P015
data = pd.read_csv('add.csv')
# Check the first few rows of the dataset to ensure it loaded correctly
print(data.head())
# Define features and target variable
x = data[['x', 'y']]
y = data['sum']
# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# Create and train the model
model = LinearRegression()
model.fit(x_train, y_train)
# Print the model's performance on training and testing sets
print(f"Training score: {model.score(x_train, y_train):.2f}")
print(f"Testing score: {model.score(x_test, y_test):.2f}")
y_pred = model.predict(x_test)
# Define the Tkinter application class
class AdditionApp:
  def __init__(self, root):
    self.root = root
    self.root.title("Addition Using Multiple Linear Regression By Surya")
    self.create_widgets()
  def create_widgets(self):
    # Create and place widgets on the window
    ttk.Label(self.root, text="Number 1:").grid(column=0, row=0, padx=10, pady=10)
    self.num1_var = tk.DoubleVar()
    self.num1_entry = ttk.Entry(self.root, textvariable=self.num1_var)
    self.num1_entry.grid(column=1, row=0, padx=10, pady=10)
    ttk.Label(self.root, text="Number 2:").grid(column=0,row=1,padx=10, pady=10)
    self.num2_var = tk.DoubleVar()
    self.num2_entry = ttk.Entry(self.root, textvariable=self.num2_var)
    self.num2_entry.grid(column=1, row=1, padx=10, pady=10)
    self.calc_button = ttk.Button(self.root, text="Calculate", command=self.calculate_sum)
    self.calc_button.grid(column=0, row=2, columnspan=2, pady=10)
    self.result_var = tk.StringVar()
    ttk.Label(self.root, text="Result:").grid(column=0, row=3, padx=10, pady=10)
    self.result_label = ttk.Label(self.root, textvariable=self.result_var)
    self.result_label.grid(column=1, row=3, padx=10, pady=10)
  def calculate_sum(self):
    # Retrieve the input values
    num1 = self.num1_var.get()
    num2 = self.num2_var.get()
    # Make a prediction based on the input values
    prediction = model.predict([[num1, num2]])[0]
    # Display the result
    self.result_var.set(f"{prediction:.2f}")
# Create the Tkinter root window and run the application
root = tk.Tk()
app = AdditionApp(root)
root.mainloop()
