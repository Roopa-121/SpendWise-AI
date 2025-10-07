#!/usr/bin/env python3
"""
Machine Learning Models Training Script
Trains Random Forest, XGBoost, LightGBM, and CatBoost on processed data
"""

import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Suppress warnings
warnings.filterwarnings('ignore')

# Import models and utilities
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLModelsTraining:
    """Advanced ML models training with hyperparameter optimization."""
    
    def __init__(self):
        """Initialize the training pipeline."""
        self.data_path = Path("data/processed")
        self.models_path = Path("models/ml")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = {}
        self.models = {}
        self.scalers = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load processed train, validation, and test datasets."""
        logger.info("Loading processed datasets...")
        
        train_data = pd.read_csv(self.data_path / "train_data.csv", parse_dates=['date'])
        val_data = pd.read_csv(self.data_path / "val_data.csv", parse_dates=['date'])
        test_data = pd.read_csv(self.data_path / "test_data.csv", parse_dates=['date'])
        
        logger.info(f"Train data: {train_data.shape}, Val data: {val_data.shape}, Test data: {test_data.shape}")
        return train_data, val_data, test_data
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features for ML models."""
        df = df.copy()
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear 
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Cyclical encoding for temporal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        
        # Lag features
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'expense_lag_{lag}'] = df['total_daily_expense'].shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 14, 30, 90]:
            df[f'expense_rolling_mean_{window}'] = df['total_daily_expense'].rolling(window=window, min_periods=1).mean()
            df[f'expense_rolling_std_{window}'] = df['total_daily_expense'].rolling(window=window, min_periods=1).std()
            df[f'expense_rolling_min_{window}'] = df['total_daily_expense'].rolling(window=window, min_periods=1).min()
            df[f'expense_rolling_max_{window}'] = df['total_daily_expense'].rolling(window=window, min_periods=1).max()
        
        # Category-wise features (treating each category as a feature)
        category_cols = [col for col in df.columns if col not in ['date', 'total_daily_expense']]
        
        for col in category_cols:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                # Lag features for each category
                df[f'{col}_lag_1'] = df[col].shift(1)
                df[f'{col}_lag_7'] = df[col].shift(7)
                
                # Rolling features for each category
                df[f'{col}_rolling_mean_7'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_rolling_mean_30'] = df[col].rolling(window=30, min_periods=1).mean()
        
        # Volatility features
        df['expense_volatility_7'] = df['total_daily_expense'].rolling(window=7, min_periods=1).std()
        df['expense_volatility_30'] = df['total_daily_expense'].rolling(window=30, min_periods=1).std()
        
        # Trend features
        df['expense_trend_7'] = df['total_daily_expense'] - df['expense_rolling_mean_7']
        df['expense_trend_30'] = df['total_daily_expense'] - df['expense_rolling_mean_30']
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns for modeling."""
        exclude_cols = ['date', 'total_daily_expense']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train Random Forest with hyperparameter tuning."""
        logger.info("Training Random Forest model with hyperparameter tuning...")
        
        # Parameter grid for tuning
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        # Time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Base model
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Random search
        rf_search = RandomizedSearchCV(
            rf, param_grid, n_iter=10, cv=tscv, 
            scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
        )
        
        rf_search.fit(X_train, y_train)
        best_rf = rf_search.best_estimator_
        
        # Predictions
        y_pred_train = best_rf.predict(X_train)
        y_pred_val = best_rf.predict(X_val)
        
        # Metrics
        results = self._calculate_metrics(
            y_train, y_pred_train, y_val, y_pred_val, 
            'Random Forest', best_rf, rf_search.best_params_
        )
        
        # Save model
        joblib.dump(best_rf, self.models_path / "random_forest.pkl")
        self.models['random_forest'] = best_rf
        
        return results
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train XGBoost with hyperparameter tuning."""
        logger.info("Training XGBoost model with hyperparameter tuning...")
        
        # Parameter grid for tuning
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        # Time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Base model
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        # Random search
        xgb_search = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=10, cv=tscv, 
            scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
        )
        
        xgb_search.fit(X_train, y_train)
        best_xgb = xgb_search.best_estimator_
        
        # Predictions
        y_pred_train = best_xgb.predict(X_train)
        y_pred_val = best_xgb.predict(X_val)
        
        # Metrics
        results = self._calculate_metrics(
            y_train, y_pred_train, y_val, y_pred_val, 
            'XGBoost', best_xgb, xgb_search.best_params_
        )
        
        # Save model
        joblib.dump(best_xgb, self.models_path / "xgboost.pkl")
        self.models['xgboost'] = best_xgb
        
        return results
    
    def _calculate_metrics(self, y_train: pd.Series, y_pred_train: np.ndarray, 
                          y_val: pd.Series, y_pred_val: np.ndarray, 
                          model_name: str, model: Any, best_params: Dict) -> Dict[str, Any]:
        """Calculate comprehensive metrics for model evaluation."""
        
        # Training metrics
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_mape = mean_absolute_percentage_error(y_train, y_pred_train) * 100
        
        # Training R² Score
        train_ss_res = np.sum((y_train - y_pred_train) ** 2)
        train_ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
        train_r2 = 1 - (train_ss_res / train_ss_tot) if train_ss_tot != 0 else 0
        
        # Training Directional Accuracy
        if len(y_train) > 1:
            train_true_direction = np.diff(y_train) > 0
            train_pred_direction = np.diff(y_pred_train) > 0
            train_directional_accuracy = np.mean(train_true_direction == train_pred_direction) * 100
        else:
            train_directional_accuracy = 0
        
        # Validation metrics
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        val_mape = mean_absolute_percentage_error(y_val, y_pred_val) * 100
        
        # Validation R² Score
        val_ss_res = np.sum((y_val - y_pred_val) ** 2)
        val_ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        val_r2 = 1 - (val_ss_res / val_ss_tot) if val_ss_tot != 0 else 0
        
        # Validation Directional Accuracy
        if len(y_val) > 1:
            val_true_direction = np.diff(y_val) > 0
            val_pred_direction = np.diff(y_pred_val) > 0
            val_directional_accuracy = np.mean(val_true_direction == val_pred_direction) * 100
        else:
            val_directional_accuracy = 0
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        
        results = {
            'model_name': model_name,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_mape': train_mape,
            'train_r2': train_r2,
            'train_directional_accuracy': train_directional_accuracy,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_mape': val_mape,
            'val_r2': val_r2,
            'val_directional_accuracy': val_directional_accuracy,
            'best_params': best_params,
            'feature_importance': feature_importance
        }
        
        logger.info(f"{model_name} - Val MAE: {val_mae:.2f}, Val RMSE: {val_rmse:.2f}, Val MAPE: {val_mape:.2f}%, Val R²: {val_r2:.3f}, Dir.Acc: {val_directional_accuracy:.1f}%")
        return results
    
    def save_results(self, results: List[Dict[str, Any]], feature_names: List[str]):
        """Save training results and feature importance."""
        # Save results DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.models_path / "ml_results.csv", index=False)
        
        # Save feature importance for each model
        for result in results:
            if result.get('feature_importance') is not None:
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': result['feature_importance']
                })
                importance_df = importance_df.sort_values('importance', ascending=False)
                importance_df.to_csv(
                    self.models_path / f"{result['model_name'].lower().replace(' ', '_')}_feature_importance.csv", 
                    index=False
                )
        
        # Create summary
        summary = {
            'training_date': datetime.now().isoformat(),
            'models_trained': len(results),
            'results': results
        }
        
        if len(results_df) > 0 and 'val_mae' in results_df.columns:
            summary.update({
                'best_model_by_mae': results_df.loc[results_df['val_mae'].idxmin(), 'model_name'],
                'best_model_by_rmse': results_df.loc[results_df['val_rmse'].idxmin(), 'model_name'],
                'best_model_by_mape': results_df.loc[results_df['val_mape'].idxmin(), 'model_name']
            })
        
        joblib.dump(summary, self.models_path / "ml_summary.pkl")
        logger.info(f"Results saved to {self.models_path}")
        
        return summary
    
    def run_ml_training(self):
        """Run complete ML training pipeline."""
        logger.info("="*60)
        logger.info("🤖 STARTING MACHINE LEARNING MODELS TRAINING")
        logger.info("="*60)
        
        # Load data
        train_data, val_data, test_data = self.load_data()
        
        # Engineer features
        logger.info("Engineering features...")
        train_features = self.engineer_features(train_data)
        val_features = self.engineer_features(val_data)
        
        # Get feature columns
        feature_cols = self.get_feature_columns(train_features)
        logger.info(f"Using {len(feature_cols)} features for training")
        
        # Prepare data
        X_train = train_features[feature_cols].fillna(0)
        y_train = train_features['total_daily_expense']
        X_val = val_features[feature_cols].fillna(0)
        y_val = val_features['total_daily_expense']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val), 
            columns=X_val.columns, 
            index=X_val.index
        )
        
        # Save scaler
        joblib.dump(scaler, self.models_path / "feature_scaler.pkl")
        self.scalers['feature_scaler'] = scaler
        
        # Train models
        results = []
        
        # Random Forest
        try:
            rf_results = self.train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)
            results.append(rf_results)
        except Exception as e:
            logger.error(f"Random Forest failed: {str(e)}")
            results.append({'model_name': 'Random Forest', 'error': str(e)})
        
        # XGBoost
        try:
            xgb_results = self.train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
            results.append(xgb_results)
        except Exception as e:
            logger.error(f"XGBoost failed: {str(e)}")
            results.append({'model_name': 'XGBoost', 'error': str(e)})
        
        # Save results
        summary = self.save_results(results, feature_cols)
        
        # Print summary
        logger.info("="*60)
        logger.info("📊 MACHINE LEARNING TRAINING RESULTS")
        logger.info("="*60)
        
        for result in results:
            if 'error' not in result:
                logger.info(f"{result['model_name']}:")
                logger.info(f"  Validation MAE: {result['val_mae']:.2f}")
                logger.info(f"  Validation RMSE: {result['val_rmse']:.2f}")
                logger.info(f"  Validation MAPE: {result['val_mape']:.2f}%")
                logger.info(f"  Best Parameters: {result['best_params']}")
                logger.info("-" * 40)
            else:
                logger.error(f"{result['model_name']}: {result['error']}")
        
        logger.info("✅ ML training completed!")
        return summary

def main():
    """Main function."""
    trainer = MLModelsTraining()
    trainer.run_ml_training()

if __name__ == "__main__":
    main()