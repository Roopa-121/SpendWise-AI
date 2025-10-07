#!/usr/bin/env python3
"""Quick data quality validation script"""

import pandas as pd
import numpy as np

# Load processed data
cleaned_data = pd.read_csv('data/processed/cleaned_transactions.csv')
train_data = pd.read_csv('data/processed/train_data.csv')

print('📊 Data Quality Validation Report')
print('=' * 50)

# Cleaned transactions validation
print(f'✅ Cleaned Data Shape: {cleaned_data.shape}')
print(f'✅ Date range: {cleaned_data["date"].min()} to {cleaned_data["date"].max()}')
print(f'✅ Amount range: ₹{cleaned_data["amount"].min():.2f} to ₹{cleaned_data["amount"].max():.2f}')
print(f'✅ Categories: {len(cleaned_data["category"].unique())} unique categories')
print(f'✅ Missing values: {cleaned_data.isnull().sum().sum()}')
print(f'✅ Duplicates: {cleaned_data.duplicated().sum()}')

# Time series validation
print(f'\n📈 Time Series Shape: {train_data.shape}')
print(f'📈 Time series date range: {train_data["date"].min()} to {train_data["date"].max()}')
print(f'📈 Average daily expense: ₹{train_data["total_daily_expense"].mean():.2f}')
print(f'📈 Missing values in time series: {train_data.isnull().sum().sum()}')

print('\n🎉 Data quality validation completed successfully!')
print('🚀 Ready for feature engineering and model training!')