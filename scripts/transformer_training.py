"""
BudgetWise AI - Advanced Transformer Models Training
Week 7: Temporal Fusion Transformer (TFT) and N-BEATS Implementation

This script implements state-of-the-art transformer architectures:
1. Temporal Fusion Transformer (TFT) - Attention-based model with interpretability
2. N-BEATS - Neural Basis Expansion Analysis for Time Series

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

# PyTorch and forecasting libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# PyTorch Forecasting for TFT
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss

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
        
    def prepare_tft_data(self):
        """Prepare data for Temporal Fusion Transformer"""
        logger.info("Preparing data for TFT...")
        
        # Use original data with dates for TFT
        def create_tft_dataset(data, split_name):
            df = data.copy()
            df['amount'] = df['total_daily_expense']
            df['time_idx'] = range(len(df))
            df['group'] = 0  # Single time series
            
            # Convert categorical columns
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_columns:
                if col not in ['date']:
                    df[col] = pd.Categorical(df[col]).codes
                    
            return df
        
        # Create datasets
        self.train_tft = create_tft_dataset(self.train_data, 'train')
        self.val_tft = create_tft_dataset(self.val_data, 'val')
        self.test_tft = create_tft_dataset(self.test_data, 'test')
        
        # Combine for time series dataset
        self.data_tft = pd.concat([self.train_tft, self.val_tft], ignore_index=True)
        
        logger.info(f"TFT data prepared: {self.data_tft.shape}")
        
    def prepare_nbeats_data(self):
        """Prepare data for N-BEATS"""
        logger.info("Preparing data for N-BEATS...")
        
        # Use same scaler as deep learning models
        scaler_path = self.models_dir.parent / "deep_learning" / "dl_scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = StandardScaler()
            combined_X = pd.concat([self.X_train, self.X_val], ignore_index=True)
            self.scaler.fit(combined_X)
            
        # Scale features
        X_train_scaled = self.scaler.transform(self.X_train)
        X_val_scaled = self.scaler.transform(self.X_val)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Create sequences for N-BEATS
        def create_sequences(X, y, seq_length):
            X_seq, y_seq = [], []
            for i in range(len(X) - seq_length):
                X_seq.append(X[i:(i + seq_length)].flatten())
                y_seq.append(y.iloc[i + seq_length])
            return np.array(X_seq), np.array(y_seq)
        
        # Create sequences
        self.X_train_seq, self.y_train_seq = create_sequences(X_train_scaled, self.y_train, self.sequence_length)
        self.X_val_seq, self.y_val_seq = create_sequences(X_val_scaled, self.y_val, self.sequence_length)
        self.X_test_seq, self.y_test_seq = create_sequences(X_test_scaled, self.y_test, self.sequence_length)
        
        # Convert to numpy arrays if they're pandas Series
        if hasattr(self.y_train_seq, 'values'):
            self.y_train_seq = self.y_train_seq.values
        if hasattr(self.y_val_seq, 'values'):
            self.y_val_seq = self.y_val_seq.values
        if hasattr(self.y_test_seq, 'values'):
            self.y_test_seq = self.y_test_seq.values
        
        logger.info(f"N-BEATS sequences: X_train: {self.X_train_seq.shape}, X_val: {self.X_val_seq.shape}")
        
    def train_tft_model(self):
        """Train Temporal Fusion Transformer"""
        logger.info("Training Temporal Fusion Transformer...")
        
        try:
            # Define time series dataset
            max_encoder_length = self.sequence_length
            max_prediction_length = self.prediction_length
            
            # Get feature columns (exclude target and metadata)
            time_varying_known_reals = []
            time_varying_unknown_reals = ['amount']
            static_categoricals = []
            static_reals = []
            
            # Create TimeSeriesDataSet
            training = TimeSeriesDataSet(
                self.data_tft[:len(self.train_tft)],
                time_idx="time_idx",
                target="amount", 
                group_ids=["group"],
                min_encoder_length=max_encoder_length // 2,
                max_encoder_length=max_encoder_length,
                min_prediction_length=1,
                max_prediction_length=max_prediction_length,
                static_categoricals=static_categoricals,
                static_reals=static_reals,
                time_varying_known_reals=time_varying_known_reals,
                time_varying_unknown_reals=time_varying_unknown_reals,
                target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )
            
            # Create validation dataset
            validation = TimeSeriesDataSet.from_dataset(training, self.data_tft, predict=True, stop_randomization=True)
            
            # Create dataloaders
            train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
            val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
            
            # Define model
            tft = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=0.03,
                hidden_size=64,
                attention_head_size=4,
                dropout=0.1,
                hidden_continuous_size=16,
                output_size=7,  # Quantiles
                loss=QuantileLoss(),
                log_interval=10,
                reduce_on_plateau_patience=4,
            )
            
            # Setup trainer
            trainer = pl.Trainer(
                max_epochs=50,
                accelerator="cpu",  # Use CPU for compatibility
                enable_model_summary=True,
                gradient_clip_val=0.1,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=10, verbose=True),
                    ModelCheckpoint(
                        dirpath=self.models_dir,
                        filename="tft-{epoch:02d}-{val_loss:.3f}",
                        monitor="val_loss",
                        save_top_k=1,
                        mode="min"
                    )
                ],
                logger=TensorBoardLogger("lightning_logs", name="tft"),
            )
            
            # Train model
            trainer.fit(tft, train_dataloader, val_dataloader)
            
            # Make predictions
            predictions = tft.predict(val_dataloader, return_y=True)
            
            # Calculate metrics
            y_true = predictions[1].numpy().flatten()
            y_pred = predictions[0].numpy().mean(axis=-1).flatten()  # Use mean of quantiles
            
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
            
            # Save model
            torch.save(tft.state_dict(), self.models_dir / "tft.pth")
            
            # Store results
            self.results['TFT'] = {
                'val_mae': mae,
                'val_rmse': rmse,
                'val_mape': mape,
                'epochs_trained': trainer.current_epoch,
                'convergence': 'Good' if trainer.current_epoch < 30 else 'Slow'
            }
            
            logger.info(f"TFT - Val MAE: {mae:.2f}, Val RMSE: {rmse:.2f}, Val MAPE: {mape:.2f}%")
            
        except Exception as e:
            logger.error(f"TFT training failed: {str(e)}")
            # Fallback to simplified metrics
            self.results['TFT'] = {
                'val_mae': float('inf'),
                'val_rmse': float('inf'), 
                'val_mape': float('inf'),
                'epochs_trained': 0,
                'convergence': 'Failed'
            }
    
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
            
            avg_train_loss = train_loss / (len(X_train_tensor) // batch_size)
            avg_val_loss = val_loss / (len(X_val_tensor) // batch_size)
            
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
        
        # Store results
        self.results['NBEATS'] = {
            'val_mae': mae,
            'val_rmse': rmse,
            'val_mape': mape,
            'epochs_trained': epoch + 1,
            'convergence': 'Good' if epoch < 50 else 'Slow'
        }
        
        logger.info(f"N-BEATS - Val MAE: {mae:.2f}, Val RMSE: {rmse:.2f}, Val MAPE: {mape:.2f}%")
    
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
        self.prepare_tft_data()
        self.prepare_nbeats_data()
        
        # Train models
        self.train_tft_model()
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
            baseline_mae = float(best_baseline['val_mae'])
            transformer_mae = float(best_transformer['val_mae'])
            
            if transformer_mae < baseline_mae:
                improvement = ((baseline_mae - transformer_mae) / baseline_mae) * 100
                logger.info(f"\nImprovements:")
                logger.info(f"Transformer vs Baseline: {improvement:.1f}% better")
            else:
                regression = ((transformer_mae - baseline_mae) / baseline_mae) * 100
                logger.info(f"\nTransformer vs Baseline: {regression:.1f}% worse")
                
        except Exception as e:
            logger.warning(f"Could not compare with previous models: {str(e)}")

def main():
    """Main execution function"""
    trainer = TransformerModelsTraining()
    trainer.run_transformer_training()

if __name__ == "__main__":
    main()