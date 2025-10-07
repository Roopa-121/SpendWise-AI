"""
BudgetWise AI - Advanced Transformer Models Training (Simplified)
Week 7: N-BEATS Implementation

This script implements N-BEATS (Neural Basis Expansion Analysis for Time Series) 
a state-of-the-art deep learning architecture specifically designed for time series forecasting.

Author: BudgetWise AI Team
Date: October 2025
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import logging
import joblib
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta

# PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Sklearn for metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transformer_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NBeatsBlock(nn.Module):
    """
    N-BEATS Block Implementation
    """
    def __init__(self, units, thetas_dim, device, backcast_length=30, forecast_length=1):
        super().__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.device = device
        
        # Stack of fully connected layers
        self.layers = nn.ModuleList([
            nn.Linear(backcast_length, units),
            nn.Linear(units, units),
            nn.Linear(units, units),
            nn.Linear(units, units),
        ])
        
        # Theta layers for backcast and forecast
        self.theta_b = nn.Linear(units, thetas_dim)
        self.theta_f = nn.Linear(units, thetas_dim)
        
        # Share weights for interpretability
        self.backcast_g = nn.Linear(thetas_dim, backcast_length)
        self.forecast_g = nn.Linear(thetas_dim, forecast_length)
        
    def forward(self, x):
        # Forward through stack
        for layer in self.layers:
            x = F.relu(layer(x))
            
        # Generate theta parameters
        theta_b = self.theta_b(x)
        theta_f = self.theta_f(x)
        
        # Generate backcast and forecast
        backcast = self.backcast_g(theta_b)
        forecast = self.forecast_g(theta_f)
        
        return backcast, forecast

class NBeatsNet(nn.Module):
    """
    Complete N-BEATS Network
    """
    def __init__(self, device, backcast_length=30, forecast_length=1, 
                 stack_types=['trend', 'seasonality', 'generic'], 
                 nb_blocks_per_stack=3, thetas_dim=[4, 8, 8], 
                 share_weights_in_stack=False, hidden_layer_units=128):
        super().__init__()
        
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.device = device
        
        # Create stacks
        self.stacks = nn.ModuleList()
        
        for stack_id, stack_type in enumerate(stack_types):
            stack = nn.ModuleList()
            for block_id in range(nb_blocks_per_stack):
                block = NBeatsBlock(
                    units=hidden_layer_units,
                    thetas_dim=thetas_dim[stack_id],
                    device=device,
                    backcast_length=backcast_length,
                    forecast_length=forecast_length
                )
                stack.append(block)
            self.stacks.append(stack)
            
    def forward(self, x):
        # Initialize forecast
        forecast = torch.zeros(x.size(0), self.forecast_length).to(self.device)
        
        for stack in self.stacks:
            for block in stack:
                backcast, block_forecast = block(x)
                x = x - backcast  # Residual connection  
                forecast = forecast + block_forecast
                
        return forecast

class TransformerModelsTraining:
    """
    Advanced Transformer Models Training Class
    """
    
    def __init__(self, data_dir: str = "data/processed", models_dir: str = "models/transformer"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model configurations
        self.sequence_length = 30
        self.prediction_length = 1
        
        # Results storage
        self.results = {}
        
    def load_data(self):
        """Load processed datasets"""
        logger.info("Loading processed datasets...")
        
        # Load train/val/test splits (same format as deep learning script)
        self.train_data = pd.read_csv(self.data_dir / "train_data.csv", parse_dates=['date'])
        self.val_data = pd.read_csv(self.data_dir / "val_data.csv", parse_dates=['date'])
        self.test_data = pd.read_csv(self.data_dir / "test_data.csv", parse_dates=['date'])
        
        # Separate features and target
        target_col = 'total_daily_expense'
        feature_cols = [col for col in self.train_data.columns if col not in ['date', target_col]]
        
        self.X_train = self.train_data[feature_cols]
        self.y_train = self.train_data[target_col]
        self.X_val = self.val_data[feature_cols]
        self.y_val = self.val_data[target_col]
        self.X_test = self.test_data[feature_cols]
        self.y_test = self.test_data[target_col]
        
        logger.info(f"Train data: {self.X_train.shape}, Val data: {self.X_val.shape}, Test data: {self.X_test.shape}")
        
    def prepare_nbeats_data(self):
        """Prepare data for N-BEATS"""
        logger.info("Preparing data for N-BEATS...")
        
        # Create new scaler for transformer data
        self.scaler = StandardScaler()
        combined_X = pd.concat([self.X_train, self.X_val], ignore_index=True)
        self.scaler.fit(combined_X)
        
        # Save scaler for future use
        joblib.dump(self.scaler, self.models_dir / "transformer_scaler.pkl")
            
        # Scale features
        X_train_scaled = self.scaler.transform(self.X_train)
        X_val_scaled = self.scaler.transform(self.X_val)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Create sequences for N-BEATS
        def create_sequences(X, y, seq_length):
            X_seq, y_seq = [], []
            for i in range(len(X) - seq_length):
                X_seq.append(X[i:(i + seq_length)].flatten())
                y_seq.append(y.iloc[i + seq_length] if hasattr(y, 'iloc') else y[i + seq_length])
            return np.array(X_seq), np.array(y_seq)
        
        # Create sequences
        self.X_train_seq, self.y_train_seq = create_sequences(X_train_scaled, self.y_train, self.sequence_length)
        self.X_val_seq, self.y_val_seq = create_sequences(X_val_scaled, self.y_val, self.sequence_length)
        self.X_test_seq, self.y_test_seq = create_sequences(X_test_scaled, self.y_test, self.sequence_length)
        
        logger.info(f"N-BEATS sequences: X_train: {self.X_train_seq.shape}, X_val: {self.X_val_seq.shape}")
    
    def train_nbeats_model(self):
        """Train N-BEATS model"""
        logger.info("Training N-BEATS model...")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(self.X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(self.y_train_seq).to(self.device)
        X_val_tensor = torch.FloatTensor(self.X_val_seq).to(self.device)
        y_val_tensor = torch.FloatTensor(self.y_val_seq).to(self.device)
        
        # Initialize model
        input_size = X_train_tensor.shape[1]
        model = NBeatsNet(
            device=self.device,
            backcast_length=input_size,
            forecast_length=1,
            stack_types=['trend', 'seasonality', 'generic'],
            nb_blocks_per_stack=3,
            thetas_dim=[4, 8, 8],
            hidden_layer_units=128
        ).to(self.device)
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        epochs = 100
        epoch = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            
            # Process in batches
            batch_size = 32
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase  
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, len(X_val_tensor), batch_size):
                    batch_X = X_val_tensor[i:i+batch_size]
                    batch_y = y_val_tensor[i:i+batch_size]
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / (len(X_train_tensor) // batch_size + 1)
            avg_val_loss = val_loss / (len(X_val_tensor) // batch_size + 1)
            
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.models_dir / "nbeats.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= 15:  # Early stopping
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(self.models_dir / "nbeats.pth"))
        model.eval()
        
        # Make predictions
        with torch.no_grad():
            y_pred = model(X_val_tensor).cpu().numpy().flatten()
            y_true = y_val_tensor.cpu().numpy()
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # R² Score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Directional Accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true.flatten()) > 0
            pred_direction = np.diff(y_pred.flatten()) > 0
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = 0
        
        # Store results
        self.results['NBEATS'] = {
            'val_mae': mae,
            'val_rmse': rmse,
            'val_mape': mape,
            'val_r2': r2,
            'val_directional_accuracy': directional_accuracy,
            'epochs_trained': epoch + 1,
            'convergence': 'Good' if epoch < 50 else 'Slow'
        }
        
        logger.info(f"N-BEATS - Val MAE: {mae:.2f}, Val RMSE: {rmse:.2f}, Val MAPE: {mape:.2f}%, Val R²: {r2:.3f}, Dir.Acc: {directional_accuracy:.1f}%")
    
    def save_results(self):
        """Save training results"""
        logger.info("Saving transformer models results...")
        
        # Save results to CSV
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        results_df.to_csv(self.models_dir / "transformer_results.csv")
        
        # Save detailed results
        with open(self.models_dir / "transformer_summary.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"Results saved to {self.models_dir}")
    
    def run_transformer_training(self):
        """Main training pipeline"""
        
        logger.info("=" * 70)
        logger.info("🤖 STARTING TRANSFORMER MODELS TRAINING")
        logger.info("=" * 70)
        
        # Load and prepare data
        self.load_data()
        self.prepare_nbeats_data()
        
        # Train models
        self.train_nbeats_model()
        
        # Save results
        self.save_results()
        
        # Display results
        logger.info("=" * 70)
        logger.info("📊 TRANSFORMER TRAINING RESULTS")
        logger.info("=" * 70)
        
        for model_name, metrics in self.results.items():
            logger.info(f"{model_name}:")
            logger.info(f"  Validation MAE: {metrics['val_mae']:.2f}")
            logger.info(f"  Validation RMSE: {metrics['val_rmse']:.2f}")
            logger.info(f"  Validation MAPE: {metrics['val_mape']:.2f}%")
            logger.info(f"  Epochs Trained: {metrics['epochs_trained']}")
            logger.info(f"  Convergence: {metrics['convergence']}")
            logger.info("-" * 50)
        
        logger.info("✅ Transformer training completed!")
        
        # Compare with previous best models
        self.compare_with_previous_models()
    
    def compare_with_previous_models(self):
        """Compare transformer results with previous models"""
        logger.info("=" * 70)
        logger.info("🏆 MODEL PERFORMANCE COMPARISON")
        logger.info("=" * 70)
        
        try:
            # Load previous results
            baseline_results = pd.read_csv("models/baseline/baseline_results.csv", index_col=0)
            ml_results = pd.read_csv("models/ml/ml_results.csv", index_col=0)
            dl_results = pd.read_csv("models/deep_learning/dl_results.csv", index_col=0)
            
            # Get best models from each category
            best_baseline = baseline_results.loc[baseline_results['val_mae'].idxmin()]
            best_ml = ml_results.loc[ml_results['val_mae'].idxmin()]  
            best_dl = dl_results.loc[dl_results['val_mae'].idxmin()]
            
            # Get best transformer model
            transformer_results = pd.DataFrame.from_dict(self.results, orient='index')
            best_transformer = transformer_results.loc[transformer_results['val_mae'].idxmin()]
            
            logger.info("Best Model Performance Comparison:")
            logger.info(f"Baseline ({best_baseline.name}): MAE {best_baseline['val_mae']:.2f}, MAPE {best_baseline['val_mape']:.2f}%")
            logger.info(f"ML ({best_ml.name}): MAE {best_ml['val_mae']:.2f}, MAPE {best_ml['val_mape']:.2f}%")
            logger.info(f"DL ({best_dl.name}): MAE {best_dl['val_mae']:.2f}, MAPE {best_dl['val_mape']:.2f}%")
            logger.info(f"Transformer ({best_transformer.name}): MAE {best_transformer['val_mae']:.2f}, MAPE {best_transformer['val_mape']:.2f}%")
            
            # Calculate improvements
            baseline_mae_val = best_baseline['val_mae']
            transformer_mae_val = best_transformer['val_mae']
            
            if transformer_mae_val < baseline_mae_val:
                improvement = ((baseline_mae_val - transformer_mae_val) / baseline_mae_val) * 100
                logger.info(f"\nImprovements:")
                logger.info(f"Transformer vs Baseline: {improvement:.1f}% better")
            else:
                regression = ((transformer_mae_val - baseline_mae_val) / baseline_mae_val) * 100
                logger.info(f"\nTransformer vs Baseline: {regression:.1f}% worse")
                
        except Exception as e:
            logger.warning(f"Could not compare with previous models: {str(e)}")

def main():
    """Main execution function"""
    trainer = TransformerModelsTraining()
    trainer.run_transformer_training()

if __name__ == "__main__":
    main()