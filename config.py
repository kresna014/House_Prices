# Konfigurasi path dan parameter proyek
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Path ke data
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

# Path untuk menyimpan model dan submission
MODELS_DIR = BASE_DIR / "models" / "saved_models"
SUBMISSION_DIR = BASE_DIR / "submissions"

# Parameter model
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Fungsi untuk membuat direktori dengan aman
def create_directories():
    """Membuat semua direktori yang diperlukan"""
    directories = [DATA_DIR, MODELS_DIR, SUBMISSION_DIR]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Directory created/verified: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")

# Buat direktori saat config diimport
create_directories()

# Print konfigurasi untuk debugging
print("Konfigurasi loaded:")
print(f"BASE_DIR: {BASE_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"TRAIN_DATA_PATH: {TRAIN_DATA_PATH}")
print(f"TEST_DATA_PATH: {TEST_DATA_PATH}")
print(f"MODELS_DIR: {MODELS_DIR}")
print(f"SUBMISSION_DIR: {SUBMISSION_DIR}")
print(f"Directory exists - data: {DATA_DIR.exists()}")
print(f"Directory exists - models: {MODELS_DIR.exists()}")
print(f"Directory exists - submissions: {SUBMISSION_DIR.exists()}")