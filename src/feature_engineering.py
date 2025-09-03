# Modul untuk feature engineering
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.features_to_drop = []
        
    def create_new_features(self, df):
        """ membuat feature baru mungkin berguna """
        df = df.copy()
        
        # Total area rumah
        if all (col in df.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
            df['TotalArea'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
            
        # Ratio area hidup terhadap total area
        if all (col in df.columns for col in ['GrLivArea', 'TotalArea']):
            df['LivingAreaRatio'] = df['GrLivArea'] / df['TotalArea']
            
        # usia rumah
        if all (col in df.columns for col in ['YrSold', 'YearBuilt']):
            df['HouseAge'] = df['YrSold'] - df['YearBuilt']
            
        # apakah rumah di renovasi
        if 'YearRemodAdd' in df.columns and 'YearBuilt' in df.columns:
            df['Renovated'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
            
        # total bathroom 
        if all(col in df.columns for col in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
            df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
            
        return df
    
    def remove_outliers(self, df, target_col=None, threshold=3):
        """Menghapus outliers menggunakan z-score"""
        if target_col and target_col in df.columns:
            z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
            df = df[z_scores < threshold]
        
        return df
    
    def select_features(self, df, target_col=None, correlation_threshold=0.1):
        """Seleksi features berdasarkan correlation dengan target"""
        if target_col and target_col in df.columns:
            # Hitung correlation dengan target
            correlations = df.corr()[target_col].abs().sort_values(ascending=False)
            
            # Pilih features dengan correlation di atas threshold
            selected_features = correlations[correlations > correlation_threshold].index.tolist()
            
            # Simpan features yang akan di-drop
            self.features_to_drop = [col for col in df.columns if col not in selected_features and col != target_col]
            
            # Drop features yang tidak terpilih
            df = df[selected_features]
        
        return df
    
    def apply_feature_engineering(self, df, target_col=None, is_training=True):
        """Apply semua feature engineering steps"""
        df_engineered = df.copy()
        
        # Buat features baru
        df_engineered = self.create_new_features(df_engineered)
        
        # Handle outliers hanya pada data training
        if is_training and target_col:
            df_engineered = self.remove_outliers(df_engineered, target_col)
        
        # Feature selection hanya pada data training
        if is_training and target_col:
            df_engineered = self.select_features(df_engineered, target_col)
        elif not is_training and self.features_to_drop:
            # Drop features yang sama seperti pada training
            cols_to_drop = [col for col in self.features_to_drop if col in df_engineered.columns]
            df_engineered = df_engineered.drop(columns=cols_to_drop)
        
        return df_engineered