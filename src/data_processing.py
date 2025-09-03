# module for data processing
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.label_encoder = {}
        self.scaler = None
        
    def handle_missing_values(self, df, numeric_strategy='median', categorical_strategy='most_frequent'):
        """ Handle missing values in the dataset and categorical features."""
        # pisahkan numeric dan categorical features
        numeric_features = df.select_dtypes(include=[np.number]).columns
        categorical_features = df.select_dtypes(include=['object']).columns
        
        # handle numeric missing values
        if self.numeric_imputer is None:
            self.numeric_imputer = SimpleImputer(strategy=numeric_strategy)
            df[numeric_features] = self.numeric_imputer.fit_transform(df[numeric_features])
        else :
            df[numeric_features] = self.numeric_imputer.transform(df[numeric_features])
        
        # handle categorical missing values
        if len(categorical_features) > 0:
            if self.categorical_imputer is None:
                self.categorical_imputer = SimpleImputer(strategy=categorical_strategy)
                df[categorical_features] = self.categorical_imputer.fit_transform(df[categorical_features])
            else:
                df[categorical_features] = self.categorical_imputer.transform(df[categorical_features])
        return df
    
    def encode_categorical_features(self, df):
        """ Encode categorical features using label encoding."""
        categorical_features = df.select_dtypes(include=['object']).columns
        
        for feature in categorical_features:
            if feature not in self.label_encoder:
                self.label_encoder[feature] = LabelEncoder()
                # Handle unseen categories by fitting on all possible values
                df[feature] = df[feature].astype(str)
                self.label_encoder[feature].fit(df[feature])
                
            df[feature] = self.label_encoder[feature].transform(df[feature].astype(str))
        return df
    
    def scale_features(self, df, fit=True):
        """ Scale numeric features using standardization."""
        numeric_features = df.select_dtypes(include=[np.number]).columns
        
        if self.scaler is None:
            self.scaler = StandardScaler()
        if fit:
            df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
        else:
            df[numeric_features] = self.scaler.transform(df[numeric_features])
        return df

    def preprocess_data(self, df, is_training=True):
        """Preprocess data lengkap"""
        df_processed = df.copy()
        
        # Handle missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # Encode categorical features
        df_processed = self.encode_categorical_features(df_processed)
        
        # Scale features hanya jika training
        if is_training:
            df_processed = self.scale_features(df_processed, fit=True)
        else:
            df_processed = self.scale_features(df_processed, fit=False)
        
        return df_processed