"""
Configuration module for SpendWise AI application.
Contains constants, styling, and configuration settings.
"""

import plotly.graph_objects as go

# Unified Plotly display configuration
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
}

# Modern CSS for enhanced user experience
CSS_STYLES = """
<style>
    /* Modern Color Palette & Typography */
    :root {
        --primary-color: #6366f1;
        --primary-dark: #4f46e5;
        --secondary-color: #06b6d4;
        --accent-color: #f59e0b;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --background-light: #f0f4f8;  /* Changed from #f8fafc to a softer blue-gray */
        --background-card: #ffffff;
        --background-page: #e8f4f8;  /* New: Soft blue page background */
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
        --border-color: #e2e8f0;
        --shadow-light: 0 1px 3px rgba(0, 0, 0, 0.1);
        --shadow-medium: 0 4px 6px rgba(0, 0, 0, 0.07);
        --shadow-heavy: 0 10px 15px rgba(0, 0, 0, 0.1);
    }

    /* Global Styles */
    * {
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Page Background & Container */
    .main .block-container {
        background: rgba(232, 244, 248, 0.9) !important;
        max-width: 1400px;
        padding: 2rem 3rem;
        margin: 0 auto;
    }

    /* Apply background image to the entire app */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 30%, #d1ecf1 70%, #bee3f8 100%) !important;
        position: relative;
        overflow-x: hidden;
    }

    /* Create money-themed background pattern with currency symbols */
    [data-testid="stAppViewContainer"]::after {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background:
            radial-gradient(circle at 20% 30%, rgba(34, 197, 94, 0.25) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(59, 130, 246, 0.2) 0%, transparent 50%),
            radial-gradient(circle at 60% 10%, rgba(168, 85, 247, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 10% 90%, rgba(245, 158, 11, 0.18) 0%, transparent 50%),
            radial-gradient(circle at 90% 40%, rgba(239, 68, 68, 0.12) 0%, transparent 50%),
            linear-gradient(45deg, rgba(34, 197, 94, 0.08) 25%, transparent 25%, transparent 75%, rgba(34, 197, 94, 0.08) 75%),
            linear-gradient(-45deg, rgba(59, 130, 246, 0.06) 25%, transparent 25%, transparent 75%, rgba(59, 130, 246, 0.06) 75%);
        background-size: 250px 250px, 300px 300px, 200px 200px, 280px 280px, 220px 220px, 80px 80px, 60px 60px;
        background-position: 0 0, 75px 75px, 150px 25px, 25px 125px, 125px 175px, 0 0, 40px 40px;
        background-repeat: repeat;
        z-index: -1;
        filter: blur(0.8px);
    }

    /* Add visible money symbols */
    [data-testid="stAppViewContainer"]::before {
        content: '💰 💵 💎 💳 💸 💰 💵 💎 💳 💸 💰 💵 💎 💳 💸';
        position: fixed;
        top: 5%;
        left: 2%;
        width: 96%;
        height: 90%;
        font-size: 36px;
        line-height: 100px;
        color: rgba(34, 197, 94, 0.2);
        text-align: center;
        white-space: pre-wrap;
        overflow: hidden;
        z-index: -1;
        pointer-events: none;
        filter: blur(0.2px);
        opacity: 0.8;
    }

    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--shadow-heavy);
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #ffffff, #f8fafc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Navigation Bar */
    .nav-container {
        background: var(--background-card);
        border-radius: 15px;
        padding: 1rem 2rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-medium);
        border: 1px solid var(--border-color);
    }

    .nav-tabs {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        justify-content: center;
    }

    .nav-tab {
        background: none;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        cursor: pointer;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-secondary);
        text-decoration: none;
    }

    .nav-tab:hover {
        background: var(--background-light);
        color: var(--primary-color);
        transform: translateY(-2px);
    }

    .nav-tab.active {
        background: var(--primary-color);
        color: white;
        box-shadow: var(--shadow-medium);
    }

    /* Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .metric-card {
        background: var(--background-card);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: var(--shadow-light);
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        text-align: center;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-medium);
    }

    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        opacity: 0.8;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-delta {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    .metric-delta.positive {
        color: var(--success-color);
    }

    .metric-delta.negative {
        color: var(--danger-color);
    }

    /* Content Cards */
    .content-card {
        background: rgba(255, 255, 255, 0.95) !important;  /* Enhanced opacity for better readability */
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-light);
        border: 1px solid var(--border-color);
        backdrop-filter: blur(10px);
    }

    .content-card h3 {
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Insight Cards */
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .insight-card {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: var(--shadow-medium);
    }

    .insight-card h4 {
        margin-bottom: 1rem;
        font-weight: 700;
    }

    .insight-card p {
        opacity: 0.95;
        line-height: 1.6;
    }

    /* Button Styles */
    .btn-primary {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
        border: none;
        padding: 0.875rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-medium);
    }

    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-heavy);
    }

    /* Form Elements */
    .form-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .form-card {
        background: var(--background-light);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }

    /* Chart Containers */
    .chart-container {
        background: var(--background-card);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-light);
        border: 1px solid var(--border-color);
    }

    /* Loading Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }

        .hero-title {
            font-size: 2.5rem;
        }

        .nav-tabs {
            flex-direction: column;
            align-items: stretch;
        }

        .metric-grid {
            grid-template-columns: 1fr;
        }

        .insight-grid {
            grid-template-columns: 1fr;
        }

        .form-grid {
            grid-template-columns: 1fr;
        }
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(232, 244, 248, 0.8);  /* Updated to work with image background */
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }
</style>
"""

# Model result paths configuration
MODEL_RESULT_PATHS = {
    'Baseline': 'baseline/baseline_results.csv',
    'Machine Learning': 'ml/ml_results.csv',
    'Deep Learning': 'deep_learning/dl_results.csv',
    'Transformer': 'transformer/transformer_results.csv'
}

# Data file paths to try in order
DATA_FILE_PATHS = [
    "../budgetwise_finance_dataset.csv",
    "budgetwise_finance_dataset.csv",
    "../data/budgetwise_finance_dataset.csv",
    "data/budgetwise_finance_dataset.csv",
    "../sample_expense_data.csv",
    "sample_expense_data.csv"
]

# Category mapping for data processing
CATEGORY_MAPPING = {
    'Food': 'Food & Dining',
    'Transportation': 'Travel',
    'Entertainment': 'Entertainment',
    'Healthcare': 'Healthcare',
    'Shopping': 'Others',
    'Utilities': 'Bills & Utilities',
    'Education': 'Education',
    'Income': 'Income',
    'Savings': 'Savings'
}

# Expected expense categories
EXPENSE_CATEGORIES = [
    'Bills & Utilities', 'Education', 'Entertainment', 'Food & Dining',
    'Healthcare', 'Income', 'Others', 'Savings', 'Travel'
]

# App configuration
APP_CONFIG = {
    'title': 'SpendWise AI - Smart Expense Forecasting',
    'icon': '💰',
    'layout': 'wide',
    'sidebar_state': 'expanded'
}

# Authentication and copyright settings
AUTH_SETTINGS = {
    'verify_authenticity': None,  # Will be set dynamically
    'create_copyright_notice': None,  # Will be set dynamically
    'project_signature': 'BW-AI-2025-v1.0'
}

def get_hero_html():
    """Return the hero section HTML"""
    return """
    <div class="hero-section">
        <h1 class="hero-title">💰 SpendWise AI</h1>
        <p class="hero-subtitle">Smart Expense Forecasting & Budget Optimization</p>
    </div>
    """

def get_navigation_html():
    """Return the navigation section HTML"""
    return '<div class="nav-container">'

def get_navigation_tabs():
    """Return navigation tab definitions"""
    return [
        ("🏠 Overview", "nav_overview", "Dashboard and key metrics"),
        ("📊 Models", "nav_models", "AI model performance comparison"),
        ("🔮 Forecasts", "nav_forecasts", "Predict future expenses"),
        ("💡 Insights", "nav_insights", "Smart spending insights"),
        ("📤 Upload", "nav_upload", "Upload your own data"),
        ("ℹ️ About Us", "nav_about", "Learn more about SpendWise AI")
    ]
