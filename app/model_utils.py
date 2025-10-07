"""
Model utilities module for SpendWise AI application.
Contains model loading, results processing, and sample data generation.
"""

import pandas as pd
from pathlib import Path
import streamlit as st

from .config import MODEL_RESULT_PATHS

class ModelManager:
    """Handles all model-related operations for the SpendWise AI application"""

    def __init__(self, root_dir=None):
        """Initialize the ModelManager"""
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).parent.parent
        self.model_results = {}
        self.models_path = None

    def find_models_path(self):
        """Find the models directory"""
        possible_paths = [
            self.root_dir / "models",
            Path("../models"),
            Path("models"),
            Path("./models")
        ]

        for path in possible_paths:
            if path.exists():
                self.models_path = path
                return path
        return None

    def create_sample_model_results(self):
        """Create sample model results for demo when real results aren't available"""
        # Create realistic sample results based on actual performance
        baseline_results = pd.DataFrame({
            'MAE': [682726, 1245892, 1567234],
            'MAPE': [521.26, 952.48, 1200.15],
            'R2': [-4.21, -8.52, -11.00]
        }, index=['ARIMA', 'Prophet', 'Linear Regression'])

        ml_results = pd.DataFrame({
            'MAE': [27137, 29847, 35621],
            'MAPE': [14.53, 15.89, 18.94],
            'R2': [0.85, 0.84, 0.81]
        }, index=['XGBoost', 'Random Forest', 'Decision Tree'])

        dl_results = pd.DataFrame({
            'MAE': [158945, 162334, 171823],
            'MAPE': [128.67, 131.21, 139.56],
            'R2': [0.27, 0.25, 0.21]
        }, index=['LSTM', 'GRU', 'CNN-1D'])

        transformer_results = pd.DataFrame({
            'MAE': [158409],
            'MAPE': [127.11],
            'R2': [0.28]
        }, index=['N-BEATS'])

        return {
            'Baseline': baseline_results,
            'Machine Learning': ml_results,
            'Deep Learning': dl_results,
            'Transformer': transformer_results
        }

    def load_model_results(self):
        """Load trained model results with fallback for cloud deployment"""
        self.model_results = {}
        loaded_categories = 0
        total_categories = len(MODEL_RESULT_PATHS)

        # Try to load model results
        for category, file_path in MODEL_RESULT_PATHS.items():
            loaded = False
            # Try multiple possible paths
            possible_paths = []

            # Add models_path if it exists
            if self.models_path is not None:
                possible_paths.append(self.models_path / file_path)

            # Add other possible paths
            possible_paths.extend([
                Path("../models") / file_path,
                Path("models") / file_path,
                Path("./models") / file_path
            ])

            for path in possible_paths:
                try:
                    if path.exists():
                        self.model_results[category] = pd.read_csv(path, index_col=0)
                        loaded = True
                        loaded_categories += 1
                        break
                except:
                    continue

            if not loaded:
                # Use sample results if real ones not found
                sample_results = self.create_sample_model_results()
                if category in sample_results:
                    self.model_results[category] = sample_results[category]

        # If no real model results found, use all sample results
        if loaded_categories == 0:
            st.warning("⚠️ No trained model results found. Using sample results for demonstration.")
            st.info("💡 **For full functionality**: Train models locally using the provided scripts in `/scripts/` directory.")
            self.model_results = self.create_sample_model_results()
        elif loaded_categories < total_categories:
            st.info(f"ℹ️ Loaded {loaded_categories}/{total_categories} model result files. Using sample data for missing results.")
            # Fill in missing categories with sample data
            sample_results = self.create_sample_model_results()
            for category, results in sample_results.items():
                if category not in self.model_results:
                    self.model_results[category] = results

    def get_all_model_results(self):
        """Compile all model results into a single dataframe for comparison"""
        all_results = []

        for category, results_df in self.model_results.items():
            for model_name, row in results_df.iterrows():
                # Handle different file structures
                if 'model_name' in results_df.columns:
                    # ML/DL results have model_name column
                    model_display_name = row.get('model_name', model_name)
                else:
                    # Baseline/Transformer results use index as model name
                    model_display_name = model_name

                all_results.append({
                    'Category': category,
                    'Model': model_display_name,
                    'MAE': row.get('val_mae', row.get('MAE', float('inf'))),
                    'RMSE': row.get('val_rmse', row.get('RMSE', float('inf'))),
                    'MAPE': row.get('val_mape', row.get('MAPE', float('inf'))),
                    'R²': row.get('val_r2', row.get('R2', row.get('R²', 0))),
                    'Directional_Accuracy': row.get('val_directional_accuracy', row.get('Directional_Accuracy', 0))
                })

        results_df = pd.DataFrame(all_results)

        # Filter out only completely invalid values
        results_df = results_df[results_df['MAE'] != float('inf')]
        results_df = results_df[~results_df['MAE'].isna()]
        results_df = results_df[~results_df['MAPE'].isna()]

        return results_df

    def get_best_model_info(self):
        """Get information about the best performing model"""
        results_df = self.get_all_model_results()

        if len(results_df) == 0:
            return None

        # Best model identification
        best_model_idx = results_df['MAE'].idxmin()
        best_model = results_df.loc[best_model_idx]

        return {
            'name': best_model['Model'],
            'category': best_model['Category'],
            'mae': best_model['MAE'],
            'rmse': best_model['RMSE'],
            'mape': best_model['MAPE'],
            'r2': best_model['R²'],
            'directional_accuracy': best_model['Directional_Accuracy']
        }

    def get_model_summary_stats(self):
        """Get summary statistics about loaded models"""
        results_df = self.get_all_model_results()

        if len(results_df) == 0:
            return None

        return {
            'total_models': len(results_df),
            'categories': len(self.model_results),
            'best_mae': results_df['MAE'].min(),
            'best_mape': results_df['MAPE'].min(),
            'best_r2': results_df['R²'].max(),
            'best_directional_accuracy': results_df['Directional_Accuracy'].max(),
            'category_names': list(self.model_results.keys())
        }

    def check_extreme_mape_models(self):
        """Check for models with extremely high MAPE values"""
        results_df = self.get_all_model_results()
        extreme_mape_models = results_df[results_df['MAPE'] > 500]

        if len(extreme_mape_models) > 0:
            return len(extreme_mape_models)
        return 0
