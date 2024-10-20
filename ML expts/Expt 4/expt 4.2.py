import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import classification_report

root = tk.Tk()
root.title("Logistic Regression Results By Surya 211P015")
report = classification_report (y_test, model.predict(X_test), output_dict=True)
report_text = pd.DataFrame (report).transpose().round(2).to_string()
text = tk. Text (root, height=10, width=60)
text.insert(tk. END, report_text)
text.pack()
gender_label = ttk. Label(root, text="Select Gender: ")
gender_label.pack(pady=5)
gender = ttk.Combobox (root, values=["Male", "Female"])
gender.pack(pady=5)
pclass_label = ttk. Label(root, text="Select Pclass:")
pclass_label.pack(pady=5)
pclass = ttk.Combobox (root, values=[1, 2, 3])
pclass.pack(pady=5)
def show_survivors():
  filtered = data[(data['Sex'] == gender.get()) & (data['Pclass'] == int(pclass.get())) & (data['Survived'] == 1)]
  result_text = f"Survivors: {len(filtered)}"
  result_label.config(text=result_text)
button = ttk.Button(root, text="Show Survivors", command=show_survivors)
button.pack(pady=10)
result_label = ttk. Label(root, text="")
result_label.pack(pady=5)
root.mainloop()
