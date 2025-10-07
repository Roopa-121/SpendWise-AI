"""
Quick patch for BudgetWise AI - Fix Outlier Issues
This directly replaces the original data files with cleaned versions
"""

import pandas as pd
import shutil
from pathlib import Path

def patch_data_files():
    """Replace original data files with cleaned versions"""
    processed_path = Path("../data/processed")
    
    # Check if cleaned files exist
    cleaned_files = [
        "train_data_cleaned.csv",
        "val_data_cleaned.csv", 
        "test_data_cleaned.csv"
    ]
    
    original_files = [
        "train_data.csv",
        "val_data.csv",
        "test_data.csv"
    ]
    
    print("🔧 Patching BudgetWise AI data files...")
    
    for cleaned_file, original_file in zip(cleaned_files, original_files):
        cleaned_path = processed_path / cleaned_file
        original_path = processed_path / original_file
        
        if cleaned_path.exists():
            # Backup original
            backup_path = processed_path / f"{original_file}.backup"
            if original_path.exists():
                shutil.copy2(original_path, backup_path)
                print(f"✅ Backed up {original_file} to {original_file}.backup")
            
            # Replace with cleaned version
            shutil.copy2(cleaned_path, original_path)
            print(f"✅ Replaced {original_file} with cleaned version")
            
            # Verify the replacement
            df = pd.read_csv(original_path)
            max_expense = df['total_daily_expense'].max()
            mean_expense = df['total_daily_expense'].mean()
            print(f"   📊 New {original_file}: Max=${max_expense:.2f}, Mean=${mean_expense:.2f}")
        else:
            print(f"❌ Cleaned file {cleaned_file} not found")
    
    print("\n🎉 Data patching complete!")
    print("   Your app will now use the cleaned data with realistic expense values")
    print("   Max daily expense is now capped at $3,000")

if __name__ == "__main__":
    patch_data_files()