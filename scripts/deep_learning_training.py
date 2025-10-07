#!/usr/bin/env python3
"""
Deep Learning Models Training Script
Implements LSTM, GRU, Bi-LSTM, CNN-1D models for time series forecasting
"""

import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import deep learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepLearningModelsTraining:
    """Advanced Deep Learning models for time series forecasting."""
    
    def __init__(self, sequence_length: int = 30):
        """Initialize the deep learning training pipeline."""
        self.data_path = Path("data/processed")
        self.models_path = Path("models/deep_learning")
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self.sequence_length = sequence_length
        self.results = {}
        self.models = {}
        self.scalers = {}
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load processed train, validation, and test datasets."""
        logger.info("Loading processed datasets...")
        
        train_data = pd.read_csv(self.data_path / "train_data.csv", parse_dates=['date'])
        val_data = pd.read_csv(self.data_path / "val_data.csv", parse_dates=['date'])
        test_data = pd.read_csv(self.data_path / "test_data.csv", parse_dates=['date'])
        
        logger.info(f"Train data: {train_data.shape}, Val data: {val_data.shape}, Test data: {test_data.shape}")
        return train_data, val_data, test_data
    
    def prepare_time_series_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare enhanced features for deep learning models."""
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # Ensure we have the target variable
        if 'total_daily_expense' not in df.columns:
            logger.error("Target variable 'total_daily_expense' not found in data")
            raise ValueError("Target variable missing")
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        
        # Lag features for expense prediction
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'expense_lag_{lag}'] = df['total_daily_expense'].shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 14, 30]:
            df[f'expense_rolling_mean_{window}'] = df['total_daily_expense'].rolling(window=window, min_periods=1).mean()
            df[f'expense_rolling_std_{window}'] = df['total_daily_expense'].rolling(window=window, min_periods=1).std()
        
        # Category features as additional inputs
        category_cols = [col for col in df.columns if col not in ['date', 'total_daily_expense'] and 
                        not col.startswith(('year', 'month', 'day', 'weekday', 'is_weekend', 'quarter', 
                                          'expense_lag', 'expense_rolling'))]
        
        # Add rolling statistics for main categories
        for col in category_cols[:5]:  # Limit to top 5 categories to avoid too many features
            if df[col].dtype in ['float64', 'int64']:
                df[f'{col}_rolling_mean_7'] = df[col].rolling(window=7, min_periods=1).mean()
        
        return df
    
    def create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series modeling."""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, 0])  # Assuming target is first column
        
        return np.array(X), np.array(y)
    
    def prepare_data_for_dl(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                           test_data: pd.DataFrame) -> Tuple:
        """Prepare data specifically for deep learning models."""
        logger.info("Preparing data for deep learning models...")
        
        # Prepare features
        train_features = self.prepare_time_series_data(train_data)
        val_features = self.prepare_time_series_data(val_data)
        test_features = self.prepare_time_series_data(test_data)
        
        # Select features for modeling
        feature_cols = [col for col in train_features.columns if col not in ['date']]
        
        # Prepare data arrays
        train_array = train_features[feature_cols].fillna(0).values
        val_array = val_features[feature_cols].fillna(0).values
        test_array = test_features[feature_cols].fillna(0).values
        
        # Scale the data
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_array)
        val_scaled = scaler.transform(val_array)
        test_scaled = scaler.transform(test_array)
        
        # Save scaler
        joblib.dump(scaler, self.models_path / "dl_scaler.pkl")
        self.scalers['dl_scaler'] = scaler
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_scaled, self.sequence_length)
        X_val, y_val = self.create_sequences(val_scaled, self.sequence_length)
        X_test, y_test = self.create_sequences(test_scaled, self.sequence_length)
        
        logger.info(f"Sequence data shapes: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    
    def build_lstm_model(self, input_shape: Tuple[int, int], lstm_units: int = 64) -> keras.Model:
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(lstm_units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_gru_model(self, input_shape: Tuple[int, int], gru_units: int = 64) -> keras.Model:
        """Build GRU model architecture."""
        model = Sequential([
            GRU(gru_units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(gru_units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_bilstm_model(self, input_shape: Tuple[int, int], lstm_units: int = 64) -> keras.Model:
        """Build Bidirectional LSTM model architecture."""
        model = Sequential([
            Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(lstm_units // 2, return_sequences=False)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_cnn_model(self, input_shape: Tuple[int, int], filters: int = 64) -> keras.Model:
        """Build CNN-1D model architecture."""
        model = Sequential([
            Conv1D(filters=filters, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Conv1D(filters=filters//2, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Flatten(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, model: keras.Model, model_name: str, 
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train a deep learning model with callbacks."""
        logger.info(f"Training {model_name} model...")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
            ModelCheckpoint(
                self.models_path / f"{model_name.lower().replace(' ', '_')}.h5",
                monitor='val_loss', save_best_only=True, save_weights_only=False
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Make predictions
        y_pred_train = model.predict(X_train, verbose=0)
        y_pred_val = model.predict(X_val, verbose=0)
        
        # Calculate metrics
        results = self._calculate_dl_metrics(
            y_train, y_pred_train.flatten(), 
            y_val, y_pred_val.flatten(), 
            model_name, history
        )
        
        # Store model
        self.models[model_name.lower().replace(' ', '_')] = model
        
        return results
    
    def _calculate_dl_metrics(self, y_train: np.ndarray, y_pred_train: np.ndarray,
                             y_val: np.ndarray, y_pred_val: np.ndarray,
                             model_name: str, history: Any) -> Dict[str, Any]:
        """Calculate comprehensive metrics for deep learning models."""
        
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
        
        # Training history
        min_val_loss = min(history.history['val_loss'])
        epochs_trained = len(history.history['loss'])
        
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
            'min_val_loss': min_val_loss,
            'epochs_trained': epochs_trained,
            'convergence': 'Good' if epochs_trained < 80 else 'Slow'
        }
        
        logger.info(f"{model_name} - Val MAE: {val_mae:.2f}, Val RMSE: {val_rmse:.2f}, Val MAPE: {val_mape:.2f}%, Val R²: {val_r2:.3f}, Dir.Acc: {val_directional_accuracy:.1f}%")
        return results
    
    def save_results(self, results: List[Dict[str, Any]]):
        """Save training results and models."""
        # Save results DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.models_path / "dl_results.csv", index=False)
        
        # Create summary
        summary = {
            'training_date': datetime.now().isoformat(),
            'sequence_length': self.sequence_length,
            'models_trained': len(results),
            'results': results
        }
        
        if len(results_df) > 0 and 'val_mae' in results_df.columns:
            summary.update({
                'best_model_by_mae': results_df.loc[results_df['val_mae'].idxmin(), 'model_name'],
                'best_model_by_rmse': results_df.loc[results_df['val_rmse'].idxmin(), 'model_name'],
                'best_model_by_mape': results_df.loc[results_df['val_mape'].idxmin(), 'model_name']
            })
        
        joblib.dump(summary, self.models_path / "dl_summary.pkl")
        logger.info(f"Results saved to {self.models_path}")
        
        return summary
    
    def run_dl_training(self):
        """Run complete deep learning training pipeline."""
        logger.info("="*70)
        logger.info("🧠 STARTING DEEP LEARNING MODELS TRAINING")
        logger.info("="*70)
        
        # Load and prepare data
        train_data, val_data, test_data = self.load_data()
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = self.prepare_data_for_dl(
            train_data, val_data, test_data
        )
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        logger.info(f"Input shape for models: {input_shape}")
        logger.info(f"Number of features: {len(feature_cols)}")
        
        results = []
        
        # Train LSTM Model
        try:
            lstm_model = self.build_lstm_model(input_shape)
            lstm_results = self.train_model(lstm_model, "LSTM", X_train, y_train, X_val, y_val)
            results.append(lstm_results)
        except Exception as e:
            logger.error(f"LSTM training failed: {str(e)}")
            results.append({'model_name': 'LSTM', 'error': str(e)})
        
        # Train GRU Model
        try:
            gru_model = self.build_gru_model(input_shape)
            gru_results = self.train_model(gru_model, "GRU", X_train, y_train, X_val, y_val)
            results.append(gru_results)
        except Exception as e:
            logger.error(f"GRU training failed: {str(e)}")
            results.append({'model_name': 'GRU', 'error': str(e)})
        
        # Train Bidirectional LSTM Model
        try:
            bilstm_model = self.build_bilstm_model(input_shape)
            bilstm_results = self.train_model(bilstm_model, "Bi-LSTM", X_train, y_train, X_val, y_val)
            results.append(bilstm_results)
        except Exception as e:
            logger.error(f"Bi-LSTM training failed: {str(e)}")
            results.append({'model_name': 'Bi-LSTM', 'error': str(e)})
        
        # Train CNN Model
        try:
            cnn_model = self.build_cnn_model(input_shape)
            cnn_results = self.train_model(cnn_model, "CNN-1D", X_train, y_train, X_val, y_val)
            results.append(cnn_results)
        except Exception as e:
            logger.error(f"CNN training failed: {str(e)}")
            results.append({'model_name': 'CNN-1D', 'error': str(e)})
        
        # Save results
        summary = self.save_results(results)
        
        # Print summary
        logger.info("="*70)
        logger.info("📊 DEEP LEARNING TRAINING RESULTS")
        logger.info("="*70)
        
        for result in results:
            if 'error' not in result:
                logger.info(f"{result['model_name']}:")
                logger.info(f"  Validation MAE: {result['val_mae']:.2f}")
                logger.info(f"  Validation RMSE: {result['val_rmse']:.2f}")
                logger.info(f"  Validation MAPE: {result['val_mape']:.2f}%")
                logger.info(f"  Epochs Trained: {result['epochs_trained']}")
                logger.info(f"  Convergence: {result['convergence']}")
                logger.info("-" * 50)
            else:
                logger.error(f"{result['model_name']}: {result['error']}")
        
        logger.info("✅ Deep Learning training completed!")
        
        # Compare with previous models
        self._compare_with_previous_models(results)
        
        return summary
    
    def _compare_with_previous_models(self, dl_results: List[Dict[str, Any]]):
        """Compare deep learning results with previous model results."""
        logger.info("="*70)
        logger.info("🏆 MODEL PERFORMANCE COMPARISON")
        logger.info("="*70)
        
        # Load previous results
        try:
            baseline_df = pd.read_csv("models/baseline/baseline_results.csv")
            ml_df = pd.read_csv("models/ml/ml_results.csv")
            
            # Best previous models
            best_baseline = baseline_df.loc[baseline_df['val_mae'].idxmin()]
            best_ml = ml_df.loc[ml_df['val_mae'].idxmin()]
            
            # Best DL model
            dl_df = pd.DataFrame([r for r in dl_results if 'error' not in r])
            if len(dl_df) > 0:
                best_dl = dl_df.loc[dl_df['val_mae'].idxmin()]
                
                logger.info("Best Model Performance Comparison:")
                logger.info(f"Baseline ({best_baseline['model_name']}): MAE {best_baseline['val_mae']:.2f}, MAPE {best_baseline['val_mape']:.2f}%")
                logger.info(f"ML ({best_ml['model_name']}): MAE {best_ml['val_mae']:.2f}, MAPE {best_ml['val_mape']:.2f}%")
                logger.info(f"DL ({best_dl['model_name']}): MAE {best_dl['val_mae']:.2f}, MAPE {best_dl['val_mape']:.2f}%")
                
                # Calculate improvements
                dl_vs_baseline = ((best_baseline['val_mae'] - best_dl['val_mae']) / best_baseline['val_mae']) * 100
                dl_vs_ml = ((best_ml['val_mae'] - best_dl['val_mae']) / best_ml['val_mae']) * 100
                
                logger.info(f"\nImprovements:")
                logger.info(f"DL vs Baseline: {dl_vs_baseline:.1f}% better")
                logger.info(f"DL vs ML: {dl_vs_ml:.1f}% {'better' if dl_vs_ml > 0 else 'worse'}")
                
        except Exception as e:
            logger.info(f"Could not load previous results for comparison: {str(e)}")

def main():
    """Main function."""
    trainer = DeepLearningModelsTraining(sequence_length=30)
    trainer.run_dl_training()

if __name__ == "__main__":
    main()