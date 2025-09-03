# Utility functions untuk proyek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import joblib
import os
from pathlib import Path

def load_data(file_path):
    """Memuat data dari file CSV"""
    print(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    return pd.read_csv(file_path)

def save_model(model, filename):
    """Menyimpan model ke file"""
    # Convert to Path object if it's a string
    if isinstance(filename, str):
        filename = Path(filename)
    
    # Ensure directory exists
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, filename)
    print(f"Model disimpan: {filename}")
    
def load_model(filename):
    """Memuat model dari file"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file tidak ditemukan: {filename}")
    return joblib.load(filename)

def evaluate_model(y_true, y_pred, model_name="Model"):
    """Evaluasi model dengan RMSE"""
    # Hitung RMSE secara manual untuk kompatibilitas dengan semua versi scikit-learn
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} RMSE: {rmse:.4f}")
    return rmse

def calculate_rmse(y_true, y_pred):
    """Menghitung RMSE secara manual"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()

def check_dataframe_info(df, name="DataFrame"):
    """Menampilkan info dataframe untuk debugging"""
    print(f"\n{name} Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    if df.isnull().sum().sum() > 0:
        print(f"Missing values:\n{df.isnull().sum().sort_values(ascending=False).head(10)}")
    else:
        print("No missing values")
    print(f"Data types:\n{df.dtypes.value_counts()}")

def check_sklearn_version():
    """Cek versi scikit-learn"""
    import sklearn
    print(f"Scikit-learn version: {sklearn.__version__}")
    return sklearn.__version__

def save_submission(df, file_path):
    """Menyimpan file submission dengan memastikan direktori ada"""
    # Convert to Path object if it's a string
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # Ensure directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(file_path, index=False)
    print(f"Submission disimpan di: {file_path}")
    return file_path