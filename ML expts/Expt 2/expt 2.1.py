import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Ensure 'placement.csv' has the required data columns
df = pd.read_csv("placement.csv")


# Set up Tkinter root window
root = tk.Tk()
root.title("CGPA vs Package Prediction")
root.geometry("800x600")


# Figure setup for Matplotlib
fig = Figure(figsize=(8, 6))
scatter_plot = fig.add_subplot(111)


# Extract X and Y data from the DataFrame
x = df[['cgpa']].values  # Adjust if column name is different
y = df['package'].values  # Adjust if column name is different


# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


# Train Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)


# Print model accuracy for debugging purposes
print("Model accuracy:", model.score(x_test, y_test))


# Function to predict package based on CGPA input
def predict_package():
   try:
       cgpa = float(entry_cgpa.get())
       predicted_package = model.predict([[cgpa]])  # Ensure input is 2D array
       label_prediction.config(text=f"Predicted Package: {predicted_package[0]:.2f} LPA")
   except ValueError:
       label_prediction.config(text="Please enter a valid CGPA")


# Tkinter widgets setup
label_cgpa = tk.Label(root, text="Enter CGPA:")
label_cgpa.pack(pady=10)


entry_cgpa = tk.Entry(root, width=10)
entry_cgpa.pack()


button_predict = tk.Button(root, text="Predict Package", command=predict_package)
button_predict.pack(pady=10)


label_prediction = tk.Label(root, text="")
label_prediction.pack(pady=10)


# Scatter plot for CGPA vs Package with regression line
scatter_plot.scatter(df['cgpa'], df['package'], label="Data Points")
scatter_plot.plot(x_train, model.predict(x_train), color='red', label="Regression Line")
scatter_plot.set_xlabel('CGPA')
scatter_plot.set_ylabel('Package')
scatter_plot.set_title('CGPA vs Package By Surya')
scatter_plot.legend()


# Display the plot on the Tkinter GUI
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


# Run the Tkinter main loop
tk.mainloop()
