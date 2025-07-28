# 🏢 Yerevan Apartment Price Prediction

This project predicts apartment prices in Yerevan based on features such as total area, number of rooms, building type, renovation type, district, and more.  
I developed this project using machine learning techniques (Neural Networks with Keras) and a custom-trained model.

---

## **Project Structure**
- **Data/** – Raw and processed datasets.
- **Notebooks/** – Data visualization, exploration, and experiments.
- **Scr/Data_processing/** – Scripts for cleaning and preparing the dataset.
- **Scr/GUI/** – Tkinter-based graphical interface for price prediction.
- **models/** – Trained model and preprocessing objects (`.keras` and `.pkl` files).
- **predict_model.py** – Script to load the trained model and predict prices.
- **train_model.py** – Script to train the neural network model.

---

## **Features**
- Predicts apartment prices in USD.
- Cleans and preprocesses raw data (translations, filtering, and conversions).
- Provides various data visualizations (price distribution, heatmaps, scatter plots).
- Interactive **Tkinter GUI** for easy price prediction.
- Supports **multi-language GUI (English + Armenian)**.
- Model evaluation with metrics like **MAE, RMSE, and R²**.
- Includes **Real vs Predicted Prices** plot for visual model evaluation.

---

## **Installation**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate   # On Windows
    source venv/bin/activate   # On Linux/Mac

3. **Install required libraries:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt