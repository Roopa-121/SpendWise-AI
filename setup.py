"""
BudgetWise Forecasting - Setup and Quick Start Script
Automates the initial setup and provides guided quick start options.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import shutil

def print_banner():
    """Print welcome banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║            🏦 BudgetWise Forecasting System 🏦               ║
    ║                                                              ║
    ║        AI-Powered Personal Expense Forecasting &            ║
    ║                Budget Optimization                           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def create_directories():
    """Create necessary project directories."""
    print("📁 Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/features",
        "models",
        "reports",
        "logs",
        "notebooks/exploratory",
        "notebooks/experiments",
        "tests",
        "app/static",
        "app/templates"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Created {directory}")
    
    print("✅ All directories created successfully!")

def install_dependencies(install_all=False):
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    
    # Basic requirements
    basic_deps = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "streamlit>=1.25.0",
        "pyyaml>=6.0",
        "joblib>=1.1.0"
    ]
    
    # Advanced ML/DL dependencies (optional)
    advanced_deps = [
        "xgboost>=1.6.0",
        "lightgbm>=3.3.0",
        "tensorflow>=2.10.0",
        "torch>=1.12.0",
        "prophet>=1.1.0",
        "statsmodels>=0.13.0"
    ]
    
    try:
        # Install basic dependencies
        print("   Installing basic dependencies...")
        for dep in basic_deps:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                          check=True, capture_output=True)
        
        print("✅ Basic dependencies installed successfully!")
        
        if install_all:
            print("   Installing advanced ML/DL dependencies...")
            for dep in advanced_deps:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                  check=True, capture_output=True)
                    print(f"   ✓ Installed {dep}")
                except subprocess.CalledProcessError as e:
                    print(f"   ⚠️  Failed to install {dep}: {e}")
            
            print("✅ Advanced dependencies installation completed!")
        else:
            print("ℹ️  Use --install-all flag to install advanced ML/DL dependencies")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def setup_sample_data():
    """Set up sample data if no data exists."""
    raw_data_path = Path("data/raw")
    
    # Check if user has their own data
    existing_files = list(raw_data_path.glob("*.csv"))
    
    if existing_files:
        print(f"✅ Found existing data files: {[f.name for f in existing_files]}")
        return True
    
    print("📊 Setting up sample data...")
    
    # Check if we have the original dataset
    original_files = [
        "budgetwise_finance_dataset.csv",
        "../budgetwise_finance_dataset.csv"
    ]
    
    source_file = None
    for file_path in original_files:
        if Path(file_path).exists():
            source_file = Path(file_path)
            break
    
    if source_file:
        # Copy to raw data directory
        dest_file = raw_data_path / "financial_data.csv"
        shutil.copy2(source_file, dest_file)
        print(f"✅ Copied sample data to {dest_file}")
        return True
    else:
        print("⚠️  No sample data found. Please add your financial data to data/raw/")
        print("   Supported format: CSV with columns like 'date', 'amount', 'category', 'description'")
        return False

def run_initial_setup():
    """Run initial data processing setup."""
    print("🔄 Running initial data processing...")
    
    try:
        # Import and run data preprocessing
        sys.path.append(str(Path.cwd()))
        
        from src.data_preprocessing import DataPreprocessor
        from src.feature_engineering import FeatureEngineer
        
        # Initialize components
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineer()
        
        # Run preprocessing
        print("   Processing raw data...")
        preprocessor.run_preprocessing_pipeline()
        
        # Run feature engineering
        print("   Engineering features...")
        feature_engineer.run_feature_engineering_pipeline()
        
        print("✅ Initial data processing completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in initial setup: {e}")
        print("   You can run this manually later with: python train_models.py --step preprocess")
        return False

def create_quick_start_script():
    """Create a quick start script."""
    script_content = """#!/usr/bin/env python3
'''
BudgetWise Forecasting - Quick Start Script
'''

import subprocess
import sys
from pathlib import Path

def main():
    print("🚀 BudgetWise Forecasting Quick Start")
    print("====================================")
    
    print("\\n1. Training all models (this may take a while)...")
    subprocess.run([sys.executable, "train_models.py"], check=False)
    
    print("\\n2. Starting Streamlit dashboard...")
    print("   Dashboard will open in your browser at http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"], check=False)

if __name__ == "__main__":
    main()
"""
    
    script_path = Path("quick_start.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':
        script_path.chmod(0o755)
    
    print(f"✅ Created quick start script: {script_path}")

def display_next_steps():
    """Display next steps for the user."""
    next_steps = """
    
    🎉 Setup Complete! Here are your next steps:
    
    📊 QUICK START (Recommended):
    ┌─────────────────────────────────────────────────────────────┐
    │  python quick_start.py                                      │
    └─────────────────────────────────────────────────────────────┘
    
    📝 MANUAL STEPS:
    
    1️⃣  Add your financial data:
       • Place CSV files in data/raw/ directory
       • Required columns: date, amount, category, description
    
    2️⃣  Train forecasting models:
       • Full pipeline: python train_models.py
       • Individual steps: python train_models.py --step [preprocess|features|baseline|ml|dl]
    
    3️⃣  Launch the dashboard:
       • streamlit run app/streamlit_app.py
       • Opens at http://localhost:8501
    
    🔧 CONFIGURATION:
       • Edit config/config.yaml to customize settings
       • Adjust model parameters, data paths, etc.
    
    📚 DOCUMENTATION:
       • README.md - Complete project documentation
       • notebooks/ - Jupyter notebooks for exploration
    
    ❓ NEED HELP?
       • Check logs/ directory for detailed logs
       • Review reports/ for model evaluation results
    """
    
    print(next_steps)

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="BudgetWise Forecasting Setup")
    parser.add_argument("--install-all", action="store_true", 
                       help="Install all dependencies including advanced ML/DL libraries")
    parser.add_argument("--skip-deps", action="store_true",
                       help="Skip dependency installation")
    parser.add_argument("--skip-data", action="store_true",
                       help="Skip initial data processing")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies(args.install_all):
            print("⚠️  Dependency installation had issues. You may need to install manually.")
    
    # Setup sample data
    setup_sample_data()
    
    # Create quick start script
    create_quick_start_script()
    
    # Run initial setup
    if not args.skip_data:
        run_initial_setup()
    
    # Display next steps
    display_next_steps()
    
    print("🎉 BudgetWise Forecasting setup completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())