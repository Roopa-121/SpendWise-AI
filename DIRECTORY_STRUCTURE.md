 - Directory Structure

This document outlines the organized directory structure of the BudgetWise AI project.

## 🏗️ Root Directory Structure

```
BudgetWise-AI-based-Expense-Forecasting-Tool/
├── 📄 README.md                     # Main project documentation
├── 📄 requirements.txt              # Python dependencies
├── 📄 setup.py                      # Project setup script
├── 📄 launch_app.py                 # Application launcher
├── 📄 SECURITY.md                   # Security documentation
├── 📄 CONTRIBUTORS.md               # Project contributors and attribution
├── 📄 DIRECTORY_STRUCTURE.md        # This file
├── 🚫 .gitignore                    # Git ignore rules
├── 🔧 .git/                         # Git repository data
├── 🔧 .streamlit/                   # Streamlit configuration
├── 🔧 .venv/                        # Virtual environment
│
├── 📊 **DATA PIPELINE**
│   ├── 📁 data/                     # Data storage (15 items)
│   │   ├── raw/                     # Original datasets
│   │   ├── processed/               # Cleaned & processed data
│   │   └── features/                # Feature-engineered data
│   │
│   ├── 📁 src/                      # Source code (7 items)
│   │   ├── data_preprocessing.py    # Data cleaning pipeline
│   │   ├── feature_engineering.py  # Feature creation
│   │   └── [5 other source files]
│   │
│   └── 📁 scripts/                  # Training scripts (9 items)
│       ├── train_models.py          # Main training pipeline
│       ├── ml_training.py           # ML model training
│       ├── deep_learning_training.py # DL model training
│       ├── transformer_training.py  # Transformer training
│       └── [5 other scripts]
│
├── 🤖 **MODELS & CONFIG**
│   ├── 📁 models/                   # Trained models (23 items)
│   │   ├── baseline/                # Statistical models
│   │   ├── ml/                      # Machine learning models
│   │   ├── deep_learning/           # Neural network models
│   │   └── transformer/             # Transformer models
│   │
│   └── 📁 config/                   # Configuration files (1 item)
│       └── config.yaml              # Main configuration
│
├── 🌐 **APPLICATION**
│   ├── 📁 app/                      # Streamlit application (7 items)
│   │   ├── budgetwise_app.py        # Main application
│   │   ├── USER_MANUAL.md           # User guide
│   │   ├── DEPLOYMENT_GUIDE.md      # Deployment instructions
│   │   └── [4 other files]
│   │
│   └── 📁 notebooks/                # Jupyter notebooks (1 item)
│       └── data_Preprocessing.ipynb # EDA notebook
│
├── 📋 **DOCUMENTATION**
│   ├── 📁 docs/                     # Documentation (14 items)
│   │   ├── project_specs/           # Project specifications
│   │   ├── technical_reports/       # Technical documentation
│   │   └── deployment/              # Deployment guides
│   │
│   └── 📁 reports/                  # Generated reports
│
├── 🔧 **UTILITIES & TESTING**
│   ├── 📁 utils/                    # Utility scripts (13 items)
│   │   ├── analyze_amounts.py       # Data analysis utilities
│   │   ├── check_all_models.py      # Model verification
│   │   └── [11 other utilities]
│   │
│   └── 📁 tests/                    # Test suite
│
├── 📝 **LOGS & TEMPORARY FILES**
│   ├── 📁 logs/                     # Log files
│   │   ├── training.log             # Training logs
│   │   └── transformer_training.log # Transformer logs
│   │
│   ├── 📁 temp/                     # Temporary files (1 item)
│   │   └── sample_expense_data.csv  # Sample data
│   │
│   └── 📁 myvenv/                   # Virtual environment
```

## 📂 Directory Descriptions

### **📊 Data Pipeline**
- **`data/`**: All datasets (raw, processed, features)
- **`src/`**: Core data processing modules
- **`scripts/`**: Model training and pipeline scripts

### **🤖 Models & Configuration**
- **`models/`**: Trained model artifacts organized by type
- **`config/`**: Configuration files for the entire system

### **🌐 Application**
- **`app/`**: Streamlit web application with user guides
- **`notebooks/`**: Jupyter notebooks for analysis and experimentation

### **📋 Documentation**
- **`docs/`**: All documentation organized by category
- **`reports/`**: Generated reports and benchmarks

### **🔧 Utilities & Testing**
- **`utils/`**: Analysis scripts, verification tools, and utilities
- **`tests/`**: Test suite for quality assurance

### **📝 Support Files**
- **`logs/`**: Training and application logs
- **`temp/`**: Temporary files and sample data
- **`myvenv/`**: Python virtual environment

## 🎯 Benefits of This Organization

### **✅ Improved Navigation**
- Clear separation of concerns
- Logical grouping of related files
- Easy to find specific components

### **✅ Better Maintainability**
- Modular structure supports easier updates
- Clear documentation hierarchy
- Separated utilities from core code

### **✅ Production Ready**
- Clean root directory
- Proper separation of temporary and permanent files
- Organized deployment documentation

## 🚀 Quick Navigation

| **Need** | **Go To** |
|----------|-----------|
| Run the app | `python launch_app.py` |
| View models | `models/` directory |
| Read docs | `docs/` directory |
| Check logs | `logs/` directory |
| Find utilities | `utils/` directory |
| View data | `data/` directory |

---

**Last Updated**: October 2025
**Structure Version**: 3.0 (Accurate)
**Author**: moah0911