"""
Quick test to check cleaned data files
"""
import pandas as pd
from pathlib import Path

# Test cleaned data loading
data_path = Path("../data/processed")

try:
    train_cleaned = pd.read_csv(data_path / "train_data_cleaned.csv")
    print("✅ Cleaned train data loaded successfully")
    print(f"Shape: {train_cleaned.shape}")
    print(f"Columns: {list(train_cleaned.columns)}")
    print(f"Total daily expense stats:")
    print(f"  Min: ${train_cleaned['total_daily_expense'].min():.2f}")
    print(f"  Max: ${train_cleaned['total_daily_expense'].max():.2f}")
    print(f"  Mean: ${train_cleaned['total_daily_expense'].mean():.2f}")
    print(f"  Median: ${train_cleaned['total_daily_expense'].median():.2f}")
    
except Exception as e:
    print(f"❌ Error loading cleaned data: {e}")
    
    # Try original data
    try:
        train_original = pd.read_csv(data_path / "train_data.csv")
        print("✅ Original train data loaded successfully")
        print(f"Shape: {train_original.shape}")
        print(f"Columns: {list(train_original.columns)}")
        print(f"Total daily expense stats:")
        print(f"  Min: ${train_original['total_daily_expense'].min():.2f}")
        print(f"  Max: ${train_original['total_daily_expense'].max():.2f}")
        print(f"  Mean: ${train_original['total_daily_expense'].mean():.2f}")
        print(f"  Median: ${train_original['total_daily_expense'].median():.2f}")
    except Exception as e2:
        print(f"❌ Error loading original data: {e2}")