import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import tkinter as tk
from tkinter import ttk, messagebox
from predict_model import predict_price

# Field definitions
numeric_fields = [
    "Number of Floors", "Total Area", "Number of Rooms",
    "Number of Bathrooms", "Ceiling Height", "Floor"
]


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

# Create the main window
root = tk.Tk()
root.title("🏢 Apartment Price Prediction")
root.geometry("500x750")
root.resizable(False, False)

entries = {}

# Add numeric input fields
row = 0
for field in numeric_fields:
    tk.Label(root, text=field + ":", font=("Arial", 11)).grid(row=row, column=0, sticky="w", padx=10, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=row, column=1, padx=10, pady=5)
    entries[field] = entry
    row += 1

# Add dropdowns for categorical fields
for field, options in categorical_options.items():
    tk.Label(root, text=field + ":", font=("Arial", 11)).grid(row=row, column=0, sticky="w", padx=10, pady=5)
    combo = ttk.Combobox(root, values=options, state="readonly")
    combo.set(options[0])
    combo.grid(row=row, column=1, padx=10, pady=5)
    entries[field] = combo
    row += 1

# Prediction function
def on_predict():
    try:
        input_dict = {}

        # Numeric values
        for field in numeric_fields:
            value = entries[field].get()
            if not value.strip():
                messagebox.showerror("Input Error", f"Please enter {field}")
                return
            input_dict[field] = float(value)

        # Categorical values
        for field in categorical_options.keys():
            input_dict[field] = entries[field].get()

        print("GUI Input Dict:", input_dict)

        # Predict
        price = predict_price(input_dict)
        messagebox.showinfo("Predicted Price", f"Estimated Price: {price:,.0f} $")

    except ValueError as e:
        messagebox.showerror("Prediction Error", str(e))

# Predict button
tk.Button(
    root,
    text="Predict Price",
    command=on_predict,
    bg="green",
    fg="white",
    font=("Arial", 12, "bold")
).grid(row=row, column=0, columnspan=2, pady=20)

root.mainloop()