"""
Main application module for SpendWise AI.
Refactored into modular format for better maintainability and ease of use.
"""

import streamlit as st
from pathlib import Path
import sys

# Add src directory to path for auth_signature import
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from auth_signature import verify_authenticity, create_copyright_notice, PROJECT_SIGNATURE
except ImportError:
    # Fallback if auth_signature is not available
    def verify_authenticity():
        return {'is_authentic': True}
    def create_copyright_notice():
        return "© 2025 SpendWise AI"
    PROJECT_SIGNATURE = "BW-AI-2025-v1.0"

from .config import APP_CONFIG
from .data_utils import DataManager
from .model_utils import ModelManager
from .ui_components import UIComponents

class BudgetWiseApp:
    """Main SpendWise AI Application Class - Smart Expense Forecasting & Budget Optimization"""

    def __init__(self):
        """Initialize the BudgetWise application"""
        # Initialize managers
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        self.ui_components = UIComponents(self.data_manager, self.model_manager)

        # Load data and models
        self.data_manager.load_data()
        self.model_manager.load_model_results()

    def run(self):
        """Run the main application"""
        # Set page config FIRST - must be the very first Streamlit command
        st.set_page_config(**APP_CONFIG)

        # Apply CSS styles
        self.ui_components.apply_css_styles()

        # Verify authenticity
        if not verify_authenticity().get('is_authentic', True):
            st.error("❌ Application authenticity verification failed.")
            st.stop()

        # Create hero section
        self.ui_components.create_hero_section()

        # Create navigation
        self.ui_components.create_navigation()

        # Initialize page state
        if 'page' not in st.session_state:
            st.session_state.page = "overview"

        # Main content based on selected page
        if st.session_state.page == "overview":
            self.ui_components.create_main_dashboard()
        elif st.session_state.page == "models":
            self.ui_components.create_model_comparison()
        elif st.session_state.page == "forecasts":
            self.ui_components.create_prediction_interface()
        elif st.session_state.page == "insights":
            self.ui_components.create_insights_page()
        elif st.session_state.page == "upload":
            self.ui_components.create_data_upload_interface()
        elif st.session_state.page == "about":
            self.ui_components.create_about_page()

def main():
    """Main application function with modern UI"""
    try:
        app = BudgetWiseApp()
        app.run()
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        st.stop()

if __name__ == "__main__":
    main()
