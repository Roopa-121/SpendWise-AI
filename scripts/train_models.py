"""
Main Training Orchestrator for BudgetWise Forecasting System
Coordinates training of all model types and evaluation.
"""

import sys
import os
from pathlib import Path
import logging
import argparse
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import all model modules
from src.data_preprocessing import AdvancedDataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models.baseline_models import BaselineModels
from src.models.ml_models import MLModels
from src.models.deep_learning_models import DeepLearningModels
from src.evaluation.model_evaluator import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingOrchestrator:
    """
    Main orchestrator for the complete model training pipeline.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize orchestrator with configuration."""
        self.config_path = config_path
        self.data_processor = AdvancedDataPreprocessor(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.baseline_models = BaselineModels(config_path)
        self.ml_models = MLModels(config_path)
        self.dl_models = DeepLearningModels(config_path)
        self.evaluator = ModelEvaluator(config_path)
        
    def run_data_preprocessing(self, force_reprocess: bool = False) -> bool:
        """
        Run data preprocessing pipeline.
        
        Args:
            force_reprocess: Whether to force reprocessing even if files exist
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("="*60)
            logger.info("🔄 STARTING DATA PREPROCESSING")
            logger.info("="*60)
            
            # Check if processed data already exists
            processed_path = Path("data/processed/")
            if processed_path.exists() and not force_reprocess:
                existing_files = list(processed_path.glob("*.csv"))
                if existing_files:
                    logger.info(f"Found {len(existing_files)} existing processed files. Use --force-reprocess to reprocess.")
                    return True
            
            # Run preprocessing
            self.data_processor.run_complete_preprocessing_pipeline()
            logger.info("✅ Data preprocessing completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data preprocessing failed: {str(e)}")
            return False
    
    def run_feature_engineering(self, force_reprocess: bool = False) -> bool:
        """
        Run feature engineering pipeline.
        
        Args:
            force_reprocess: Whether to force reprocessing even if files exist
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("="*60)
            logger.info("🔧 STARTING FEATURE ENGINEERING")
            logger.info("="*60)
            
            # Check if feature files already exist
            features_path = Path("data/features/")
            if features_path.exists() and not force_reprocess:
                existing_files = list(features_path.glob("*_features.csv"))
                if len(existing_files) >= 3:  # train, val, test
                    logger.info(f"Found {len(existing_files)} existing feature files. Use --force-reprocess to reprocess.")
                    return True
            
            # Run feature engineering
            self.feature_engineer.run_feature_engineering_pipeline()
            logger.info("✅ Feature engineering completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Feature engineering failed: {str(e)}")
            return False
    
    def train_baseline_models(self) -> bool:
        """
        Train all baseline models.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("="*60)
            logger.info("📊 TRAINING BASELINE MODELS")
            logger.info("="*60)
            
            self.baseline_models.train_all_baseline_models()
            logger.info("✅ Baseline models training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Baseline models training failed: {str(e)}")
            return False
    
    def train_ml_models(self, tune_hyperparameters: bool = True) -> bool:
        """
        Train all machine learning models.
        
        Args:
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("="*60)
            logger.info("🤖 TRAINING MACHINE LEARNING MODELS")
            logger.info("="*60)
            
            self.ml_models.train_all_ml_models(tune_hyperparameters=tune_hyperparameters)
            logger.info("✅ ML models training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ ML models training failed: {str(e)}")
            return False
    
    def train_deep_learning_models(self) -> bool:
        """
        Train all deep learning models.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("="*60)
            logger.info("🧠 TRAINING DEEP LEARNING MODELS")
            logger.info("="*60)
            
            self.dl_models.train_all_dl_models()
            logger.info("✅ Deep learning models training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deep learning models training failed: {str(e)}")
            return False
    
    def run_evaluation(self) -> str:
        """
        Run comprehensive model evaluation.
        
        Returns:
            Path to evaluation report
        """
        try:
            logger.info("="*60)
            logger.info("📋 RUNNING MODEL EVALUATION")
            logger.info("="*60)
            
            # Quick summary first
            self.evaluator.print_quick_summary()
            
            # Generate detailed report
            report_path = self.evaluator.generate_detailed_report()
            logger.info(f"✅ Evaluation completed! Report: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {str(e)}")
            return ""
    
    def run_full_pipeline(self, 
                         force_reprocess: bool = False,
                         skip_baseline: bool = False,
                         skip_ml: bool = False,
                         skip_dl: bool = False,
                         tune_hyperparameters: bool = True) -> Dict[str, bool]:
        """
        Run the complete training pipeline.
        
        Args:
            force_reprocess: Whether to force data reprocessing
            skip_baseline: Whether to skip baseline models
            skip_ml: Whether to skip ML models
            skip_dl: Whether to skip deep learning models
            tune_hyperparameters: Whether to tune hyperparameters for ML models
            
        Returns:
            Dictionary with success status for each step
        """
        logger.info("🚀 STARTING FULL BUDGETWISE FORECASTING PIPELINE")
        logger.info("="*80)
        
        results = {}
        
        # Step 1: Data Preprocessing
        results['data_preprocessing'] = self.run_data_preprocessing(force_reprocess)
        if not results['data_preprocessing']:
            logger.error("❌ Pipeline stopped due to data preprocessing failure")
            return results
        
        # Step 2: Feature Engineering
        results['feature_engineering'] = self.run_feature_engineering(force_reprocess)
        if not results['feature_engineering']:
            logger.error("❌ Pipeline stopped due to feature engineering failure")
            return results
        
        # Step 3: Train Baseline Models
        if not skip_baseline:
            results['baseline_models'] = self.train_baseline_models()
        else:
            logger.info("⏭️  Skipping baseline models training")
            results['baseline_models'] = True
        
        # Step 4: Train ML Models
        if not skip_ml:
            results['ml_models'] = self.train_ml_models(tune_hyperparameters)
        else:
            logger.info("⏭️  Skipping ML models training")
            results['ml_models'] = True
        
        # Step 5: Train Deep Learning Models
        if not skip_dl:
            results['deep_learning_models'] = self.train_deep_learning_models()
        else:
            logger.info("⏭️  Skipping deep learning models training")
            results['deep_learning_models'] = True
        
        # Step 6: Evaluation
        report_path = self.run_evaluation()
        results['evaluation'] = bool(report_path)
        results['report_path'] = report_path
        
        # Final summary
        self._print_pipeline_summary(results)
        
        return results
    
    def _print_pipeline_summary(self, results: Dict[str, Any]) -> None:
        """Print pipeline execution summary."""
        
        print("\n" + "="*80)
        print("🎯 BUDGETWISE FORECASTING PIPELINE SUMMARY")
        print("="*80)
        
        steps = [
            ('Data Preprocessing', 'data_preprocessing'),
            ('Feature Engineering', 'feature_engineering'),
            ('Baseline Models', 'baseline_models'),
            ('ML Models', 'ml_models'),
            ('Deep Learning Models', 'deep_learning_models'),
            ('Model Evaluation', 'evaluation')
        ]
        
        for step_name, step_key in steps:
            if step_key in results:
                status = "✅ SUCCESS" if results[step_key] else "❌ FAILED"
                print(f"{step_name:<25}: {status}")
        
        if 'report_path' in results and results['report_path']:
            print(f"\n📋 Evaluation Report: {results['report_path']}")
        
        # Overall status
        overall_success = all(results.get(key, False) for key, _ in steps if key in results)
        overall_status = "🎉 PIPELINE COMPLETED SUCCESSFULLY!" if overall_success else "⚠️  PIPELINE COMPLETED WITH ISSUES"
        
        print(f"\n{overall_status}")
        print("="*80)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="BudgetWise Forecasting Training Pipeline")
    
    parser.add_argument("--config", default="config/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--force-reprocess", action="store_true",
                       help="Force reprocessing of data and features")
    parser.add_argument("--skip-baseline", action="store_true",
                       help="Skip baseline models training")
    parser.add_argument("--skip-ml", action="store_true",
                       help="Skip machine learning models training")
    parser.add_argument("--skip-dl", action="store_true",
                       help="Skip deep learning models training")
    parser.add_argument("--no-tune", action="store_true",
                       help="Skip hyperparameter tuning for ML models")
    parser.add_argument("--step", choices=[
        'preprocess', 'features', 'baseline', 'ml', 'dl', 'evaluate'
    ], help="Run only a specific step")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(args.config)
    
    # Run specific step or full pipeline
    if args.step:
        if args.step == 'preprocess':
            orchestrator.run_data_preprocessing(args.force_reprocess)
        elif args.step == 'features':
            orchestrator.run_feature_engineering(args.force_reprocess)
        elif args.step == 'baseline':
            orchestrator.train_baseline_models()
        elif args.step == 'ml':
            orchestrator.train_ml_models(not args.no_tune)
        elif args.step == 'dl':
            orchestrator.train_deep_learning_models()
        elif args.step == 'evaluate':
            orchestrator.run_evaluation()
    else:
        # Run full pipeline
        orchestrator.run_full_pipeline(
            force_reprocess=args.force_reprocess,
            skip_baseline=args.skip_baseline,
            skip_ml=args.skip_ml,
            skip_dl=args.skip_dl,
            tune_hyperparameters=not args.no_tune
        )


if __name__ == "__main__":
    main()