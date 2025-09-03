# Program utama untuk menjalankan proyek prediksi harga rumah - versi simplified
import pandas as pd
import numpy as np
from config import TRAIN_DATA_PATH, TEST_DATA_PATH, MODELS_DIR, SUBMISSION_DIR, RANDOM_STATE, TEST_SIZE
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.modeling import HousePriceModel
from src.utils import load_data, save_model, evaluate_model, plot_feature_importance, check_dataframe_info, calculate_rmse, check_sklearn_version, save_submission
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

def main():
    print("Memulai proyek prediksi harga rumah...")
    print("Menggunakan model Random Forest dan Gradient Boosting saja")
    
    # Cek versi scikit-learn
    check_sklearn_version()
    
    try:
        # 1. Load data
        print("1. Memuat data...")
        
        # Cek jika file data ada
        if not TRAIN_DATA_PATH.exists():
            print(f"File training data tidak ditemukan: {TRAIN_DATA_PATH}")
            print("Pastikan Anda telah mendownload train.csv dan test.csv dari Kaggle")
            print("dan menempatkannya di folder data/")
            return
        
        train_data = load_data(TRAIN_DATA_PATH)
        
        if not TEST_DATA_PATH.exists():
            print(f"File test data tidak ditemukan: {TEST_DATA_PATH}")
            print("Lanjutkan tanpa test data untuk prediksi...")
            test_data = None
        else:
            test_data = load_data(TEST_DATA_PATH)
            print(f"Test data shape: {test_data.shape}")
        
        print(f"Training data shape: {train_data.shape}")
        
        # Debug info
        check_dataframe_info(train_data, "Training Data")
        if test_data is not None:
            check_dataframe_info(test_data, "Test Data")
        
        # 2. Pisahkan features dan target
        target_col = 'SalePrice'
        
        # Pastikan target column ada di training data
        if target_col not in train_data.columns:
            print("Target column 'SalePrice' tidak ditemukan. Columns yang ada:")
            print(train_data.columns.tolist())
            return
        
        # Simpan ID untuk submission
        X = train_data.drop(columns=[target_col, 'Id'])
        y = train_data[target_col]
        
        # 3. Preprocessing data
        print("2. Preprocessing data...")
        data_processor = DataProcessor()
        feature_engineer = FeatureEngineer()
        
        # Apply feature engineering
        X_engineered = feature_engineer.apply_feature_engineering(X, target_col=target_col, is_training=True)
        print(f"Shape setelah feature engineering: {X_engineered.shape}")
        
        # Preprocess data
        X_processed = data_processor.preprocess_data(X_engineered, is_training=True)
        print(f"Shape setelah preprocessing: {X_processed.shape}")
        
        # 4. Split data
        print("3. Splitting data...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_processed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
        
        # 5. Training model
        print("4. Training model...")
        
        # Model 1: Random Forest
        print("\nTraining Random Forest...")
        rf_model = HousePriceModel(model_type='random_forest', random_state=RANDOM_STATE)
        rf_model.train(X_train, y_train)
        
        # Prediksi dan evaluasi
        y_pred_rf = rf_model.predict(X_val)
        rf_rmse = evaluate_model(y_val, y_pred_rf, "Random Forest")
        
        # Model 2: Gradient Boosting
        print("\nTraining Gradient Boosting...")
        gb_model = HousePriceModel(model_type='gradient_boosting', random_state=RANDOM_STATE)
        gb_model.train(X_train, y_train)
        
        y_pred_gb = gb_model.predict(X_val)
        gb_rmse = evaluate_model(y_val, y_pred_gb, "Gradient Boosting")
        
        # Pilih model terbaik berdasarkan RMSE
        if rf_rmse < gb_rmse:
            best_model = rf_model
            best_model_name = "Random Forest"
            best_pred = y_pred_rf
            best_rmse = rf_rmse
        else:
            best_model = gb_model
            best_model_name = "Gradient Boosting"
            best_pred = y_pred_gb
            best_rmse = gb_rmse
        
        print(f"\nModel terbaik: {best_model_name} (RMSE: {best_rmse:.4f})")
        
        # 6. Plot feature importance
        print("5. Plotting feature importance...")
        feature_importances = best_model.get_feature_importances()
        if feature_importances is not None:
            plot_feature_importance(best_model.model, X_train.columns)
        
        # 7. Preprocess test data (jika test data tersedia)
        if test_data is not None:
            print("6. Preprocessing test data...")
            test_ids = test_data['Id']
            X_test_raw = test_data.drop(columns=['Id'])
            
            # Apply feature engineering dan preprocessing yang sama seperti training
            X_test_engineered = feature_engineer.apply_feature_engineering(X_test_raw, is_training=False)
            X_test_processed = data_processor.preprocess_data(X_test_engineered, is_training=False)
            
            # Pastikan features sama dengan training data
            # Align test features dengan training features
            missing_cols = set(X_train.columns) - set(X_test_processed.columns)
            for col in missing_cols:
                X_test_processed[col] = 0
            
            extra_cols = set(X_test_processed.columns) - set(X_train.columns)
            X_test_processed = X_test_processed[X_train.columns]
            
            print(f"Test data shape setelah alignment: {X_test_processed.shape}")
            
            # 8. Membuat prediksi untuk test data
            print("7. Membuat prediksi untuk test data...")
            test_predictions = best_model.predict(X_test_processed)
            
            # 9. Menyimpan hasil
            print("8. Menyimpan hasil...")
            submission = pd.DataFrame({
                'Id': test_ids,
                'SalePrice': test_predictions
            })
            
            submission_path = SUBMISSION_DIR / 'submission.csv'
            save_submission(submission, submission_path)
            print(f"Preview submission:\n{submission.head()}")
        else:
            print("6. Test data tidak tersedia, melewati prediksi test data...")
        
        # 10. Menyimpan model
        print("9. Menyimpan model...")
        model_path = MODELS_DIR / 'best_model.pkl'
        save_model(best_model, model_path)
        print(f"Model disimpan di: {model_path}")
        
        # 11. Informasi tambahan
        print("\n" + "="*50)
        print("INFORMASI PROYEK:")
        print(f"Model terbaik: {best_model_name}")
        print(f"RMSE terbaik: {best_rmse:.4f}")
        print(f"Jumlah features: {X_train.shape[1]}")
        print(f"Jumlah samples training: {X_train.shape[0]}")
        if test_data is not None:
            print(f"File submission: {SUBMISSION_DIR / 'submission.csv'}")
        print("="*50)
        
        print("\nProyek selesai dengan sukses!")
        
    except Exception as e:
        print(f"Error terjadi: {str(e)}")
        print("Detail error:", type(e).__name__)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()