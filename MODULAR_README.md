# 🚀 BudgetWise AI - Modular Structure Documentation

## 📁 Project Structure

The BudgetWise AI application has been refactored into a clean, modular architecture for better maintainability and ease of use.

```
BudgetWise-AI-based-Expense-Forecasting-Tool-main/
├── app/                          # Main application package
│   ├── __init__.py              # Package initialization
│   ├── app.py                   # Main application class and navigation
│   ├── config.py                # Configuration, constants, and CSS styles
│   ├── data_utils.py            # Data loading, processing, and validation
│   ├── model_utils.py           # Model management and sample data generation
│   └── ui_components.py         # UI page creation and component methods
├── launch_app.py                # Updated launcher for modular structure
└── [other project files...]
```

## 🏗️ Architecture Overview

### **Core Modules**

1. **`config.py`** - Centralized configuration
   - CSS styles and themes
   - Constants and configuration settings
   - Navigation and UI helper functions

2. **`data_utils.py`** - Data management
   - Data loading from multiple sources
   - Data validation and processing
   - Sample data generation
   - File upload handling

3. **`model_utils.py`** - Model management
   - Model result loading and processing
   - Performance comparison utilities
   - Sample model results for demo

4. **`ui_components.py`** - User interface
   - Page creation methods (dashboard, models, forecasts, etc.)
   - Chart and visualization components
   - Form handling and user interactions

5. **`app.py`** - Main application
   - Core `BudgetWiseApp` class
   - Navigation and page routing
   - Application initialization

## 🚀 Quick Start

### **Installation & Setup**

1. **Navigate to project directory:**
   ```bash
   cd BudgetWise-AI-based-Expense-Forecasting-Tool-main
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application:**
   ```bash
   python launch_app.py
   ```

### **Using the Modular Structure**

#### **Import the main application:**
```python
from app import BudgetWiseApp

# Create and run the application
app = BudgetWiseApp()
app.run()
```

#### **Import specific modules:**
```python
# Data management
from app.data_utils import DataManager
data_manager = DataManager()

# Model utilities
from app.model_utils import ModelManager
model_manager = ModelManager()

# UI components
from app.ui_components import UIComponents
ui = UIComponents(data_manager, model_manager)
```

#### **Access configuration:**
```python
from app.config import PLOTLY_CONFIG, CSS_STYLES, APP_CONFIG

# Use configuration in your code
st.markdown(CSS_STYLES, unsafe_allow_html=True)
fig.update_layout(**PLOTLY_CONFIG)
```

## 📚 Module Documentation

### **DataManager Class (`data_utils.py`)**

**Key Methods:**
- `load_data()` - Load data from files or generate sample data
- `validate_uploaded_data(df)` - Validate CSV uploads
- `process_uploaded_data(df)` - Process and transform uploaded data
- `get_outlier_filtered_data()` - Remove outliers for visualization
- `set_uploaded_data(df)` - Set custom data as active dataset

**Usage:**
```python
from app.data_utils import DataManager

dm = DataManager()
dm.load_data()
filtered_data = dm.get_outlier_filtered_data()
```

### **ModelManager Class (`model_utils.py`)**

**Key Methods:**
- `load_model_results()` - Load trained model performance data
- `get_all_model_results()` - Get compiled results for comparison
- `get_best_model_info()` - Identify top performing model
- `create_sample_model_results()` - Generate demo model results

**Usage:**
```python
from app.model_utils import ModelManager

mm = ModelManager()
mm.load_model_results()
results_df = mm.get_all_model_results()
best_model = mm.get_best_model_info()
```

### **UIComponents Class (`ui_components.py`)**

**Key Methods:**
- `create_main_dashboard()` - Main dashboard with metrics and charts
- `create_model_comparison()` - Model performance comparison page
- `create_prediction_interface()` - Forecasting interface
- `create_insights_page()` - AI insights and recommendations
- `create_data_upload_interface()` - File upload and processing
- `create_about_page()` - About us and company information

**Usage:**
```python
from app.ui_components import UIComponents

ui = UIComponents(data_manager, model_manager)
ui.apply_css_styles()
ui.create_main_dashboard()
```

### **BudgetWiseApp Class (`app.py`)**

**Key Methods:**
- `run()` - Launch the complete Streamlit application
- Initialization handles data loading and model setup

**Usage:**
```python
from app import BudgetWiseApp

app = BudgetWiseApp()
app.run()  # Launches the full application
```

## 🔧 Configuration

### **Customization Options**

#### **Modify CSS Styles:**
Edit `config.py` to customize the appearance:
```python
# In config.py
CSS_STYLES = """
/* Your custom styles here */
:root {
    --primary-color: #your-color;
    /* ... other variables */
}
"""
```

#### **Update Model Paths:**
```python
# In config.py
MODEL_RESULT_PATHS = {
    'Baseline': 'your/path/baseline_results.csv',
    'Machine Learning': 'your/path/ml_results.csv',
    # ... other paths
}
```

#### **Add Data Sources:**
```python
# In config.py
DATA_FILE_PATHS = [
    "your/custom/data.csv",
    # ... other paths
]
```

## 🧪 Testing & Development

### **Run Individual Modules:**
```python
# Test data utilities
python -c "
from app.data_utils import DataManager
dm = DataManager()
dm.load_data()
print(f'Data loaded: {len(dm.all_data)} records')
"
```

### **Debug Mode:**
```python
# Run with debug output
python -c "
import streamlit as st
from app import BudgetWiseApp

# Enable debug mode
st.set_option('client.showErrorDetails', True)
app = BudgetWiseApp()
app.run()
"
```

## 🚀 Deployment

### **Production Deployment:**
```bash
# For production, use the modular structure
python -m streamlit run app/app.py --server.port 8502 --server.headless true
```

### **Docker Deployment:**
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8502
CMD ["streamlit", "run", "app/app.py", "--server.port", "8502"]
```

## 📈 Benefits of Modular Structure

### **✅ Maintainability**
- **Separation of Concerns**: Each module has a specific responsibility
- **Easy Updates**: Modify individual components without affecting others
- **Clear Dependencies**: Import relationships are explicit

### **✅ Reusability**
- **Component Reuse**: UI components can be used across different pages
- **Shared Utilities**: Data and model utilities can be imported anywhere
- **Configuration Centralization**: All settings in one place

### **✅ Testability**
- **Unit Testing**: Test individual modules in isolation
- **Mocking**: Easy to mock dependencies for testing
- **Debugging**: Easier to isolate and fix issues

### **✅ Scalability**
- **Feature Addition**: Add new modules without disrupting existing code
- **Team Development**: Multiple developers can work on different modules
- **Performance**: Load only required components

## 🔍 Troubleshooting

### **Common Issues**

1. **Import Errors:**
   ```bash
   # Ensure you're in the correct directory
   cd BudgetWise-AI-based-Expense-Forecasting-Tool-main
   python -c "from app import BudgetWiseApp"
   ```

2. **Missing Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Files Not Found:**
   - Check that data files exist in the expected locations
   - The app will automatically fall back to sample data if files are missing

4. **Streamlit Issues:**
   ```bash
   # Clear Streamlit cache
   python -c "import streamlit as st; st.cache_data.clear()"
   ```

## 📞 Support

For issues or questions about the modular structure:
1. Check the troubleshooting section above
2. Review the individual module documentation
3. Test components in isolation before full application runs

---

**© 2025 SpendWise AI** | Modular Architecture for Scalable Financial Intelligence
