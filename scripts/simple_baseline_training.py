#!/usr/bin/env python3
"""
Simple Baseline Models Training Script
Trains Linear Regression, ARIMA, and Prophet on processed data
"""

import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

# Suppress warnings
warnings.filterwarnings('ignore')

# Import models
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleBaselineTraining:
    """Simple baseline models training with processed data."""
    
    def __init__(self):
        """Initialize the training pipeline."""
        self.data_path = Path("data/processed")
        self.models_path = Path("models/baseline")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = {}
        self.models = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load processed train, validation, and test datasets."""
        logger.info("Loading processed datasets...")
        
        train_data = pd.read_csv(self.data_path / "train_data.csv", parse_dates=['date'])
        val_data = pd.read_csv(self.data_path / "val_data.csv", parse_dates=['date'])
        test_data = pd.read_csv(self.data_path / "test_data.csv", parse_dates=['date'])
        
        logger.info(f"Train data: {train_data.shape}, Val data: {val_data.shape}, Test data: {test_data.shape}")
        return train_data, val_data, test_data
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare basic features for modeling."""
        df = df.copy()
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Create lag features for total expense
        df['expense_lag_1'] = df['total_daily_expense'].shift(1)
        df['expense_lag_7'] = df['total_daily_expense'].shift(7)
        df['expense_rolling_7'] = df['total_daily_expense'].rolling(window=7, min_periods=1).mean()
        df['expense_rolling_30'] = df['total_daily_expense'].rolling(window=30, min_periods=1).mean()
        
        return df
    
    def train_linear_regression(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, Any]:
        """Train Linear Regression model."""
        logger.info("Training Linear Regression model...")
        
        # Prepare features
        train_features = self.prepare_features(train_data)
        val_features = self.prepare_features(val_data)
        
        # Select numerical features
        feature_cols = ['year', 'month', 'day', 'weekday', 'is_weekend', 'quarter', 'day_of_year']
        lag_cols = ['expense_lag_1', 'expense_lag_7', 'expense_rolling_7', 'expense_rolling_30']
        feature_cols.extend(lag_cols)
        
        X_train = train_features[feature_cols].fillna(0)
        y_train = train_features['total_daily_expense'].fillna(train_features['total_daily_expense'].median())
        
        X_val = val_features[feature_cols].fillna(0)
        y_val = val_features['total_daily_expense'].fillna(val_features['total_daily_expense'].median())
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        # Metrics
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
        
        # Save model
        joblib.dump(model, self.models_path / "linear_regression.pkl")
        
        results = {
            'model_name': 'Linear Regression',
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
            'feature_importance': dict(zip(feature_cols, model.coef_))
        }
        
        logger.info(f"Linear Regression - Val MAE: {val_mae:.2f}, Val RMSE: {val_rmse:.2f}, Val MAPE: {val_mape:.2f}%, Val R²: {val_r2:.3f}, Dir.Acc: {val_directional_accuracy:.1f}%")
        return results
    
    def train_prophet(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, Any]:
        """Train Prophet model."""
        logger.info("Training Prophet model...")
        
        # Prepare data for Prophet (data is already daily aggregated)
        prophet_train = train_data[['date', 'total_daily_expense']].copy()
        prophet_train.columns = ['ds', 'y']
        
        prophet_val = val_data[['date', 'total_daily_expense']].copy()
        prophet_val.columns = ['ds', 'y']
        
        # Train model
        model = Prophet(
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_train)
        
        # Predictions
        train_future = model.make_future_dataframe(periods=0)
        train_forecast = model.predict(train_future)
        
        val_future = model.make_future_dataframe(periods=len(prophet_val))
        val_forecast = model.predict(val_future)
        val_forecast = val_forecast.tail(len(prophet_val))
        
        # Metrics
        train_mae = mean_absolute_error(prophet_train['y'], train_forecast['yhat'])
        train_rmse = np.sqrt(mean_squared_error(prophet_train['y'], train_forecast['yhat']))
        train_mape = mean_absolute_percentage_error(prophet_train['y'], train_forecast['yhat']) * 100
        
        # Training R² Score
        train_y_true = np.array(prophet_train['y'])
        train_y_pred = np.array(train_forecast['yhat'])
        train_ss_res = np.sum((train_y_true - train_y_pred) ** 2)
        train_ss_tot = np.sum((train_y_true - np.mean(train_y_true)) ** 2)
        train_r2 = 1 - (train_ss_res / train_ss_tot) if train_ss_tot != 0 else 0
        
        # Training Directional Accuracy
        if len(train_y_true) > 1:
            train_true_direction = np.diff(train_y_true) > 0
            train_pred_direction = np.diff(train_y_pred) > 0
            train_directional_accuracy = np.mean(train_true_direction == train_pred_direction) * 100
        else:
            train_directional_accuracy = 0
        
        val_mae = mean_absolute_error(prophet_val['y'], val_forecast['yhat'])
        val_rmse = np.sqrt(mean_squared_error(prophet_val['y'], val_forecast['yhat']))
        val_mape = mean_absolute_percentage_error(prophet_val['y'], val_forecast['yhat']) * 100
        
        # Validation R² Score
        val_y_true = np.array(prophet_val['y'])
        val_y_pred = np.array(val_forecast['yhat'])
        val_ss_res = np.sum((val_y_true - val_y_pred) ** 2)
        val_ss_tot = np.sum((val_y_true - np.mean(val_y_true)) ** 2)
        val_r2 = 1 - (val_ss_res / val_ss_tot) if val_ss_tot != 0 else 0
        
        # Validation Directional Accuracy
        if len(val_y_true) > 1:
            val_true_direction = np.diff(val_y_true) > 0
            val_pred_direction = np.diff(val_y_pred) > 0
            val_directional_accuracy = np.mean(val_true_direction == val_pred_direction) * 100
        else:
            val_directional_accuracy = 0
        
        # Save model
        joblib.dump(model, self.models_path / "prophet.pkl")
        
        results = {
            'model_name': 'Prophet',
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_mape': train_mape,
            'train_r2': train_r2,
            'train_directional_accuracy': train_directional_accuracy,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_mape': val_mape,
            'val_r2': val_r2,
            'val_directional_accuracy': val_directional_accuracy
        }
        
        logger.info(f"Prophet - Val MAE: {val_mae:.2f}, Val RMSE: {val_rmse:.2f}, Val MAPE: {val_mape:.2f}%, Val R²: {val_r2:.3f}, Dir.Acc: {val_directional_accuracy:.1f}%")
        return results
    
    def train_arima(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> Dict[str, Any]:
        """Train ARIMA model."""
        logger.info("Training ARIMA model...")
        
        try:
            # Prepare data for ARIMA (data is already daily aggregated)
            arima_train = train_data.set_index('date')['total_daily_expense']
            arima_val = val_data.set_index('date')['total_daily_expense']
            
            # Train ARIMA model with simple parameters
            model = ARIMA(arima_train, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Predictions
            train_pred = fitted_model.fittedvalues
            val_pred = fitted_model.forecast(steps=len(arima_val))
            
            # Align predictions with actual values
            train_actual = arima_train.iloc[1:]  # ARIMA starts from second observation
            train_pred = train_pred.iloc[1:]
            
            # Metrics
            train_mae = mean_absolute_error(train_actual, train_pred)
            train_rmse = np.sqrt(mean_squared_error(train_actual, train_pred))
            train_mape = mean_absolute_percentage_error(train_actual, train_pred) * 100
            
            # Training R² Score
            train_y_true = np.array(train_actual)
            train_y_pred = np.array(train_pred)
            train_ss_res = np.sum((train_y_true - train_y_pred) ** 2)
            train_ss_tot = np.sum((train_y_true - np.mean(train_y_true)) ** 2)
            train_r2 = 1 - (train_ss_res / train_ss_tot) if train_ss_tot != 0 else 0
            
            # Training Directional Accuracy
            if len(train_y_true) > 1:
                train_true_direction = np.diff(train_y_true) > 0
                train_pred_direction = np.diff(train_y_pred) > 0
                train_directional_accuracy = np.mean(train_true_direction == train_pred_direction) * 100
            else:
                train_directional_accuracy = 0
            
            val_mae = mean_absolute_error(arima_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(arima_val, val_pred))
            val_mape = mean_absolute_percentage_error(arima_val, val_pred) * 100
            
            # Validation R² Score
            val_y_true = np.array(arima_val)
            val_y_pred = np.array(val_pred)
            val_ss_res = np.sum((val_y_true - val_y_pred) ** 2)
            val_ss_tot = np.sum((val_y_true - np.mean(val_y_true)) ** 2)
            val_r2 = 1 - (val_ss_res / val_ss_tot) if val_ss_tot != 0 else 0
            
            # Validation Directional Accuracy
            if len(val_y_true) > 1:
                val_true_direction = np.diff(val_y_true) > 0
                val_pred_direction = np.diff(val_y_pred) > 0
                val_directional_accuracy = np.mean(val_true_direction == val_pred_direction) * 100
            else:
                val_directional_accuracy = 0
            
            # Save model
            joblib.dump(fitted_model, self.models_path / "arima.pkl")
            
            results = {
                'model_name': 'ARIMA',
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
                'model_summary': str(fitted_model.summary())
            }
            
            logger.info(f"ARIMA - Val MAE: {val_mae:.2f}, Val RMSE: {val_rmse:.2f}, Val MAPE: {val_mape:.2f}%, Val R²: {val_r2:.3f}, Dir.Acc: {val_directional_accuracy:.1f}%")
            return results
            
        except Exception as e:
            logger.error(f"ARIMA training failed: {str(e)}")
            return {
                'model_name': 'ARIMA',
                'error': str(e),
                'train_mae': np.nan,
                'train_rmse': np.nan,
                'train_mape': np.nan,
                'val_mae': np.nan,
                'val_rmse': np.nan,
                'val_mape': np.nan
            }
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save training results."""
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.models_path / "baseline_results.csv", index=False)
        
        # Create summary
        summary = {
            'training_date': datetime.now().isoformat(),
            'models_trained': len(results),
            'best_model_by_mae': results_df.loc[results_df['val_mae'].idxmin(), 'model_name'],
            'best_model_by_rmse': results_df.loc[results_df['val_rmse'].idxmin(), 'model_name'],
            'best_model_by_mape': results_df.loc[results_df['val_mape'].idxmin(), 'model_name'],
            'results': results
        }
        
        joblib.dump(summary, self.models_path / "baseline_summary.pkl")
        logger.info(f"Results saved to {self.models_path}")
        
        return summary
    
    def run_baseline_training(self):
        """Run complete baseline training pipeline."""
        logger.info("="*60)
        logger.info("🚀 STARTING BASELINE MODELS TRAINING")
        logger.info("="*60)
        
        # Load data
        train_data, val_data, test_data = self.load_data()
        
        # Train models
        results = []
        
        # Linear Regression
        try:
            lr_results = self.train_linear_regression(train_data, val_data)
            results.append(lr_results)
        except Exception as e:
            logger.error(f"Linear Regression failed: {str(e)}")
            results.append({'model_name': 'Linear Regression', 'error': str(e)})
        
        # Prophet
        try:
            prophet_results = self.train_prophet(train_data, val_data)
            results.append(prophet_results)
        except Exception as e:
            logger.error(f"Prophet failed: {str(e)}")
            results.append({'model_name': 'Prophet', 'error': str(e)})
        
        # ARIMA
        arima_results = self.train_arima(train_data, val_data)
        results.append(arima_results)
        
        # Save results
        summary = self.save_results(results)
        
        # Print summary
        logger.info("="*60)
        logger.info("📊 BASELINE TRAINING RESULTS")
        logger.info("="*60)
        
        for result in results:
            if 'error' not in result:
                logger.info(f"{result['model_name']}:")
                logger.info(f"  Validation MAE: {result['val_mae']:.2f}")
                logger.info(f"  Validation RMSE: {result['val_rmse']:.2f}")
                logger.info(f"  Validation MAPE: {result['val_mape']:.2f}%")
                logger.info("-" * 40)
            else:
                logger.error(f"{result['model_name']}: {result['error']}")
        
        logger.info("✅ Baseline training completed!")
        return summary

def main():
    """Main function."""
    trainer = SimpleBaselineTraining()
    trainer.run_baseline_training()

if __name__ == "__main__":
    main()