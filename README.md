# 🏢 Yerevan Apartment Price Prediction

This project predicts apartment prices in Yerevan based on features such as total area, number of rooms, building type, renovation type, district, and more.  
I developed this project using machine learning techniques (Neural Networks with Keras).

---

## **Project Structure**
- **Data/** – Raw and processed datasets.
- **Models/** – Trained model and preprocessing objects (`.keras` and `.pkl` files).
- **Notebooks/** – Data visualization, exploration, and experiments.
- **Scr/Data_processing/** – Scripts for cleaning and preparing the dataset.
- **Scr/GUI/** – Tkinter-based graphical interface for price prediction.
- **predict_model.py** – Script to load the trained model and predict prices.
- **train_model.py** – Script to train the neural network model.

---

## **Features**
- Predicts apartment prices in USD.
- Cleans and preprocesses raw data (translations, filtering, and conversions).
- Provides various data visualizations (price distribution, heatmaps, scatter plots).
- Interactive **Tkinter GUI** for easy price prediction.
- GUI is available in **English**.
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
    # On Windows
    py -m venv venv    
    ./venv/Scripts/activate
    
    # On Linux/Mac
    python3 -m venv venv    
    source venv/bin/activate
3. **Install required libraries:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt

4. **How to use the project:**
   ```bash
   # Run the prediction script
   python Scr/predict_model.py

   # Train the model
   python Scr/train_model.py

   # Launch the GUI for price prediction
   python Scr/GUI/app.py



   #You need to use Jupyter Notebook to open and run data_processing and  data_visualisation files:
   
   bash
   jupyter notebook Notebooks/data_processing.ipynb
   jupyter notebook Notebooks/data_visualisation.ipynb


   #If you want to run them from VS Code, install the Jupyter extension and make sure you have Jupyter installed:

   bash
   pip install notebook jupyterlab