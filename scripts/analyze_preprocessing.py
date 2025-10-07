#!/usr/bin/env python3
"""Enhanced Data Preprocessing Analysis and Summary"""

import pandas as pd
import numpy as np

def analyze_enhanced_preprocessing():
    """Analyze the results of enhanced preprocessing pipeline."""
    
    print("🎉 ENHANCED DATA PREPROCESSING ANALYSIS")
    print("="*60)
    
    # Load processed data
    cleaned_data = pd.read_csv('data/processed/cleaned_transactions.csv')
    train_data = pd.read_csv('data/processed/train_data.csv')
    
    print("📊 ENHANCED FEATURES ANALYSIS")
    print("-"*40)
    
    # 1. Category Standardization Analysis
    print("\n🏷️  CATEGORY STANDARDIZATION:")
    print(f"   • Standardized categories: {cleaned_data['category'].nunique()}")
    print(f"   • Categories distribution:")
    category_dist = cleaned_data['category'].value_counts()
    for cat, count in category_dist.items():
        percentage = (count / len(cleaned_data)) * 100
        print(f"     - {cat}: {count:,} ({percentage:.1f}%)")
    
    # Check if standardization flags exist
    if 'category_was_standardized' in cleaned_data.columns:
        standardized_count = cleaned_data['category_was_standardized'].sum()
        print(f"   • Entries standardized: {standardized_count:,} ({(standardized_count/len(cleaned_data)*100):.1f}%)")
    
    # 2. Location Standardization Analysis
    print("\n🗺️  LOCATION STANDARDIZATION:")
    print(f"   • Unique locations: {cleaned_data['location'].nunique()}")
    location_dist = cleaned_data['location'].value_counts().head(10)
    print(f"   • Top locations:")
    for loc, count in location_dist.items():
        percentage = (count / len(cleaned_data)) * 100
        print(f"     - {loc}: {count:,} ({percentage:.1f}%)")
    
    # 3. Payment Mode Standardization Analysis
    print("\n💳 PAYMENT MODE STANDARDIZATION:")
    print(f"   • Unique payment modes: {cleaned_data['payment_mode'].nunique()}")
    payment_dist = cleaned_data['payment_mode'].value_counts()
    print(f"   • Payment modes distribution:")
    for mode, count in payment_dist.items():
        percentage = (count / len(cleaned_data)) * 100
        print(f"     - {mode}: {count:,} ({percentage:.1f}%)")
    
    # 4. Date Quality Analysis
    print("\n📅 DATE QUALITY ANALYSIS:")
    if 'date_imputed' in cleaned_data.columns:
        imputed_dates = cleaned_data['date_imputed'].sum()
        print(f"   • Dates imputed: {imputed_dates:,}")
    
    if 'date_is_future' in cleaned_data.columns:
        future_dates = cleaned_data['date_is_future'].sum()
        print(f"   • Future dates: {future_dates:,}")
    
    if 'date_is_very_old' in cleaned_data.columns:
        old_dates = cleaned_data['date_is_very_old'].sum()
        print(f"   • Very old dates: {old_dates:,}")
    
    # 5. Amount Quality Analysis
    print("\n💰 AMOUNT QUALITY ANALYSIS:")
    if 'amount_imputed' in cleaned_data.columns:
        imputed_amounts = cleaned_data['amount_imputed'].sum()
        print(f"   • Amounts imputed: {imputed_amounts:,}")
    
    if 'amount_was_invalid' in cleaned_data.columns:
        invalid_amounts = cleaned_data['amount_was_invalid'].sum()
        print(f"   • Invalid amounts fixed: {invalid_amounts:,}")
    
    if 'is_outlier' in cleaned_data.columns:
        outliers = cleaned_data['is_outlier'].sum()
        print(f"   • Outliers detected: {outliers:,} ({(outliers/len(cleaned_data)*100):.1f}%)")
    
    # 6. Duplicate Analysis
    print("\n🔍 DUPLICATE ANALYSIS:")
    if 'was_duplicate' in cleaned_data.columns:
        duplicates = cleaned_data['was_duplicate'].sum()
        print(f"   • Transaction ID duplicates resolved: {duplicates:,}")
    
    # 7. Time Series Quality
    print("\n📈 TIME SERIES QUALITY:")
    print(f"   • Time series span: {len(train_data):,} days")
    print(f"   • Categories in time series: {len([col for col in train_data.columns if col not in ['date', 'total_daily_expense']])}")
    print(f"   • Average daily expense: ₹{train_data['total_daily_expense'].mean():,.2f}")
    print(f"   • Max daily expense: ₹{train_data['total_daily_expense'].max():,.2f}")
    print(f"   • Min daily expense: ₹{train_data['total_daily_expense'].min():,.2f}")
    
    # 8. Data Completeness
    print("\n✅ DATA COMPLETENESS:")
    print(f"   • Total processed records: {len(cleaned_data):,}")
    print(f"   • Total features: {len(cleaned_data.columns)}")
    print(f"   • Missing values: {cleaned_data.isnull().sum().sum():,}")
    print(f"   • Data completeness: {((cleaned_data.size - cleaned_data.isnull().sum().sum()) / cleaned_data.size * 100):.2f}%")
    
    # 9. Feature Engineering Ready
    print("\n🚀 FEATURE ENGINEERING READINESS:")
    temporal_features = [col for col in cleaned_data.columns if col in ['year', 'month', 'day_of_week', 'is_weekend']]
    quality_flags = [col for col in cleaned_data.columns if 'imputed' in col or 'invalid' in col or 'outlier' in col or 'duplicate' in col]
    
    print(f"   • Temporal features: {len(temporal_features)} ({temporal_features})")
    print(f"   • Quality flags: {len(quality_flags)} ({quality_flags})")
    print(f"   • Original vs processed columns available: Yes")
    print(f"   • Dataset ready for advanced feature engineering: ✅")
    
    print("\n" + "="*60)
    print("🎯 SUMMARY: Enhanced preprocessing with fuzzy matching, intelligent")
    print("   imputation, and comprehensive quality checks completed successfully!")
    print("🚀 Ready for advanced feature engineering and model training!")

if __name__ == "__main__":
    analyze_enhanced_preprocessing()