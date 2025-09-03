# Modul untuk modeling - versi simplified tanpa xgboost/lightgbm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def rmse_scorer(y_true, y_pred):
    """Custom scorer untuk RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

class HousePriceModel:
    def __init__(self, model_type='random_forest', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.best_params = None
        
    def create_model(self, **params):
        """Membuat model berdasarkan type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                random_state=self.random_state,
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                n_jobs=-1  # Use all cores
            )
            
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                random_state=self.random_state,
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 3),
                subsample=params.get('subsample', 0.8)
            )
            
        else:
            # Fallback to Random Forest jika model type tidak dikenali
            print(f"Model type {self.model_type} tidak dikenali, menggunakan Random Forest")
            self.model = RandomForestRegressor(random_state=self.random_state, n_estimators=100)
        
        return self.model
    
    def tune_hyperparameters(self, X, y, param_grid, cv=5):
        """Tuning hyperparameters menggunakan GridSearch"""
        # Gunakan custom scorer untuk RMSE
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring=make_scorer(rmse_scorer, greater_is_better=False), 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return self.model
    
    def cross_validate(self, X, y, cv=5):
        """Cross validation"""
        cv_scores = cross_val_score(
            self.model, X, y, cv=cv, 
            scoring=make_scorer(rmse_scorer, greater_is_better=False)
        )
        
        print(f"CV RMSE Scores: {[-score for score in cv_scores]}")
        print(f"Mean CV RMSE: {np.mean(-cv_scores):.4f} (+/- {np.std(-cv_scores):.4f})")
        
        return cv_scores
    
    def train(self, X, y, tune=False, param_grid=None):
        """Train model"""
        if self.model is None:
            self.create_model()
        
        if tune and param_grid:
            self.tune_hyperparameters(X, y, param_grid)
        else:
            self.model.fit(X, y)
        
        return self.model
    
    def predict(self, X):
        """Membuat prediksi"""
        if self.model is None:
            raise ValueError("Model belum di-training")
        
        return self.model.predict(X)
    
    def get_feature_importances(self):
        """Mendapatkan feature importances"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            print("Model tidak memiliki feature_importances_ attribute")
            return None