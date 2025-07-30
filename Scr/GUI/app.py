# Import necessary libraries
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add parent directory to sys.path for imports

import tkinter as tk
from tkinter import ttk, messagebox
from predict_model import predict_price  # Import the prediction function

# Field definitions

# Numeric fields where users enter values
numeric_fields = [
    "Number of Floors", "Total Area", "Number of Rooms",
    "Number of Bathrooms", "Ceiling Height", "Floor"
]

# Dropdown fields with predefined options
categorical_options = {
    "Building Type": ['Stone', 'Monolith', 'Panel'],
    "Balcony": ['Not Available', 'Closed Balcony', 'Open Balcony', 'Multiple Balconies'],
    "Furniture": ['Partially Furnished', 'Available', 'Not Available'],
    "Renovation": ['Old Renovation', 'Capital Renovated', 'Designer Renovated',
                   'Partially Renovated', 'Not Renovated', 'Euro Renovation', 'Cosmetic Renovation'],
    "New Building": ['No', 'Yes'],
    "Elevator": ['Not Available', 'Available'],
    "District": ['Arabkir', 'Davtashen', 'Kentron', 'Malatia-Sebastia',
                 'Avan', 'Qanaqer-Zeytun', 'Nor Norq', 'Erebuni', 'Ajapnyak', 'Shengavit']
}

# Main Window Setup
root = tk.Tk()
root.title("Apartment Price Prediction")  # Window title
root.geometry("500x750")                    # Window size
root.resizable(False, False)                # Prevent resizing

entries = {}  # Dictionary to store input widgets

# Numeric Input Fields
row = 0
for field in numeric_fields:
    tk.Label(root, text=field + ":", font=("Arial", 11)).grid(row=row, column=0, sticky="w", padx=10, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=row, column=1, padx=10, pady=5)
    entries[field] = entry
    row += 1

# Dropdown Fields
for field, options in categorical_options.items():
    tk.Label(root, text=field + ":", font=("Arial", 11)).grid(row=row, column=0, sticky="w", padx=10, pady=5)
    combo = ttk.Combobox(root, values=options, state="readonly")
    combo.set(options[0])  # Default value
    combo.grid(row=row, column=1, padx=10, pady=5)
    entries[field] = combo
    row += 1

# Prediction Function
def on_predict():
    try:
        input_dict = {}

        # Collect numeric values
        for field in numeric_fields:
            value = entries[field].get()
            if not value.strip():  # If field is empty
                messagebox.showerror("Input Error", f"Please enter {field}")
                return
            input_dict[field] = float(value)  # Convert to float

        # Collect categorical values
        for field in categorical_options.keys():
            input_dict[field] = entries[field].get()

        print("GUI Input Dict:", input_dict)  # Debugging log

        # Make prediction
        price = predict_price(input_dict)
        messagebox.showinfo("Predicted Price", f"Estimated Price: {price:,.0f} $")

    except ValueError as e:
        messagebox.showerror("Prediction Error", str(e))

# Predict Button
tk.Button(
    root,
    text="Predict Price",
    command=on_predict,
    bg="green",
    fg="white",
    font=("Arial", 12, "bold")
).grid(row=row, column=0, columnspan=2, pady=20)

# Run the App

root.mainloop()