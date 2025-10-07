"""
Data utilities module for SpendWise AI application.
Contains data loading, processing, validation, and aggregation functions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
import random

from .config import DATA_FILE_PATHS, CATEGORY_MAPPING, EXPENSE_CATEGORIES

class DataManager:
    """Handles all data-related operations for the SpendWise AI application"""

    def __init__(self, root_dir=None):
        """Initialize the DataManager"""
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).parent.parent
        self.all_data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.uploaded_data = None
        self.use_uploaded_data = False

    def find_data_path(self):
        """Find the first available data path"""
        possible_data_paths = [
            self.root_dir / "data" / "processed",
            Path("../data/processed"),
            Path("data/processed"),
            Path("./data/processed"),
            Path(".")
        ]

        for path in possible_data_paths:
            if path.exists() and (path / "train_data.csv").exists():
                return path
        return None

    def find_models_path(self):
        """Find the first available models path"""
        possible_models_paths = [
            self.root_dir / "models",
            Path("../models"),
            Path("models"),
            Path("./models")
        ]

        for path in possible_models_paths:
            if path.exists():
                return path
        return None

    def create_sample_data(self):
        """Create sample data for demo purposes when real data isn't available"""
        # Generate realistic sample expense data matching the expected structure
        start_date = datetime.now() - timedelta(days=365)
        dates = [start_date + timedelta(days=i) for i in range(365)]

        sample_data = []

        for date in dates:
            # Create daily aggregated expense record
            daily_record = {'date': date}

            # Initialize all categories with 0
            for cat in EXPENSE_CATEGORIES:
                daily_record[cat] = 0.0

            # Generate random expenses for 2-4 categories per day
            active_categories = random.sample(EXPENSE_CATEGORIES, random.randint(2, 4))
            daily_total = 0

            for cat in active_categories:
                if cat == 'Income':
                    amount = random.uniform(0, 5000)  # Higher income amounts
                elif cat == 'Savings':
                    amount = random.uniform(0, 2000)  # Savings amounts
                elif cat == 'Bills & Utilities':
                    amount = random.uniform(50, 300)  # Utility bills
                elif cat == 'Food & Dining':
                    amount = random.uniform(20, 150)  # Food expenses
                elif cat == 'Healthcare':
                    amount = random.uniform(0, 500)   # Healthcare costs
                elif cat == 'Travel':
                    amount = random.uniform(0, 800)   # Travel expenses
                elif cat == 'Entertainment':
                    amount = random.uniform(10, 200)  # Entertainment
                elif cat == 'Education':
                    amount = random.uniform(0, 400)   # Education costs
                else:  # Others
                    amount = random.uniform(5, 300)   # Other expenses

                daily_record[cat] = round(amount, 2)
                daily_total += amount

            # Calculate total daily expense
            daily_record['total_daily_expense'] = round(daily_total, 2)
            sample_data.append(daily_record)

        df = pd.DataFrame(sample_data)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def aggregate_transaction_data(self, raw_data):
        """Aggregate transaction-level data to daily totals"""
        # Ensure date column is datetime
        raw_data['date'] = pd.to_datetime(raw_data['date'])

        # Apply category mapping if category column exists
        if 'category' in raw_data.columns:
            raw_data['category'] = raw_data['category'].map(CATEGORY_MAPPING).fillna('Others')

        # Group by date and category, sum amounts
        if 'category' in raw_data.columns and 'amount' in raw_data.columns:
            daily_agg = raw_data.groupby(['date', 'category'])['amount'].sum().reset_index()

            # Pivot to get categories as columns
            daily_pivot = daily_agg.pivot(index='date', columns='category', values='amount').fillna(0.0)

            # Ensure all expected columns are present
            for col in EXPENSE_CATEGORIES:
                if col not in daily_pivot.columns:
                    daily_pivot[col] = 0.0

            # Reset index to make date a column
            daily_pivot = daily_pivot.reset_index()

            # Calculate total daily expense
            expense_cols = [col for col in EXPENSE_CATEGORIES if col not in ['Income']]
            daily_pivot['total_daily_expense'] = daily_pivot[expense_cols].sum(axis=1)

        else:
            # If no category/amount columns, just group by date and sum all numeric columns
            numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                daily_pivot = raw_data.groupby('date')[numeric_cols].sum().reset_index()
                daily_pivot['total_daily_expense'] = daily_pivot[numeric_cols].sum(axis=1)
            else:
                # Fallback - create minimal structure
                daily_pivot = raw_data.groupby('date').size().reset_index(name='total_daily_expense')

        return daily_pivot

    def get_outlier_filtered_data(self, column='total_daily_expense', method='iqr'):
        """Filter outliers for better visualization while keeping original data intact"""
        data = self.all_data.copy()

        if method == 'iqr':
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Cap extreme values instead of removing them
            data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)

        elif method == 'percentile':
            # Use 1st and 99th percentiles as bounds
            lower_bound = data[column].quantile(0.01)
            upper_bound = data[column].quantile(0.99)
            data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)

        return data

    def load_data(self):
        """Load processed data with fallback for cloud deployment"""
        try:
            # First try to load the split datasets (train/val/test)
            data_path = self.find_data_path()
            if data_path and (data_path / "train_data.csv").exists():
                self.train_data = pd.read_csv(data_path / "train_data.csv", parse_dates=['date'])
                self.val_data = pd.read_csv(data_path / "val_data.csv", parse_dates=['date'])
                self.test_data = pd.read_csv(data_path / "test_data.csv", parse_dates=['date'])

                # Combine all data for analysis
                self.all_data = pd.concat([self.train_data, self.val_data, self.test_data], ignore_index=True)
                self.all_data = self.all_data.sort_values('date').reset_index(drop=True)
                return

            # Try to load the original dataset or sample data
            for file_path in DATA_FILE_PATHS:
                try:
                    raw_data = pd.read_csv(file_path, parse_dates=['date'])
                    raw_data = raw_data.sort_values('date').reset_index(drop=True)

                    # Check if this is already aggregated daily data (has total_daily_expense column)
                    if 'total_daily_expense' in raw_data.columns:
                        self.all_data = raw_data
                    else:
                        # Transform transaction-level data to daily aggregated data
                        self.all_data = self.aggregate_transaction_data(raw_data)

                    # Create train/val/test splits for compatibility
                    total_len = len(self.all_data)
                    train_end = int(total_len * 0.7)
                    val_end = int(total_len * 0.85)

                    self.train_data = self.all_data[:train_end].copy()
                    self.val_data = self.all_data[train_end:val_end].copy()
                    self.test_data = self.all_data[val_end:].copy()

                    st.info(f"📊 Loaded data from {file_path}. Functionality may be limited without preprocessed data.")
                    return
                except Exception as e:
                    continue

            # If no data files found, create sample data
            st.warning("⚠️ No data files found. Using sample data for demonstration.")
            st.info("💡 **For full functionality**: Ensure `budgetwise_finance_dataset.csv` is in the repository root or run data preprocessing locally.")

            self.all_data = self.create_sample_data()

            # Create train/val/test splits for compatibility
            total_len = len(self.all_data)
            train_end = int(total_len * 0.7)
            val_end = int(total_len * 0.85)

            self.train_data = self.all_data[:train_end].copy()
            self.val_data = self.all_data[train_end:val_end].copy()
            self.test_data = self.all_data[val_end:].copy()

        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("🔧 **Troubleshooting**: Check that data files exist and are accessible.")
            # Create minimal sample data as final fallback
            self.all_data = self.create_sample_data()
            self.train_data = self.all_data.copy()
            self.val_data = pd.DataFrame()
            self.test_data = pd.DataFrame()

    def validate_uploaded_data(self, df):
        """Validate uploaded CSV data format"""
        errors = []
        valid = True

        # Check if DataFrame is empty
        if df.empty:
            errors.append("File is empty")
            valid = False
            return {'valid': valid, 'errors': errors}

        # Check for required columns
        required_cols = ['date', 'amount']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")
            valid = False

        # Validate date column
        if 'date' in df.columns:
            try:
                pd.to_datetime(df['date'])
            except Exception as e:
                errors.append(f"Date column format issue: {str(e)}")
                valid = False

        # Validate amount column
        if 'amount' in df.columns:
            try:
                pd.to_numeric(df['amount'])
            except Exception as e:
                errors.append(f"Amount column must contain numeric values: {str(e)}")
                valid = False

        return {'valid': valid, 'errors': errors}

    def process_uploaded_data(self, df):
        """Process uploaded data to match the expected format"""
        try:
            # Make a copy to avoid modifying original
            processed_df = df.copy()

            # Ensure date is datetime
            processed_df['date'] = pd.to_datetime(processed_df['date'])

            # Ensure amount is numeric
            processed_df['amount'] = pd.to_numeric(processed_df['amount'])

            # Sort by date
            processed_df = processed_df.sort_values('date').reset_index(drop=True)

            # Aggregate to daily totals (similar to existing data processing)
            processed_df = self.aggregate_transaction_data(processed_df)

            return processed_df

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return None

    def set_uploaded_data(self, df):
        """Set uploaded data as the active dataset"""
        if df is not None:
            self.uploaded_data = df
            self.use_uploaded_data = True

            # Update all_data to use uploaded data
            self.all_data = df.copy()

            # Recalculate splits for compatibility
            self.train_data = df.copy()
            self.val_data = pd.DataFrame()
            self.test_data = pd.DataFrame()

    def reset_to_sample_data(self):
        """Reset to sample data"""
        self.use_uploaded_data = False
        self.all_data = self.create_sample_data()

        # Create train/val/test splits for compatibility
        total_len = len(self.all_data)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.85)

        self.train_data = self.all_data[:train_end].copy()
        self.val_data = self.all_data[train_end:val_end].copy()
        self.test_data = self.all_data[val_end:].copy()

    def get_data_info(self):
        """Get information about current data state"""
        return {
            'has_data': self.all_data is not None and not self.all_data.empty,
            'data_source': 'uploaded' if self.use_uploaded_data else 'sample',
            'total_records': len(self.all_data) if self.all_data is not None else 0,
            'date_range': {
                'start': self.all_data['date'].min() if self.all_data is not None else None,
                'end': self.all_data['date'].max() if self.all_data is not None else None
            } if self.all_data is not None else None,
            'total_expense': self.all_data['total_daily_expense'].sum() if self.all_data is not None else 0
        }
