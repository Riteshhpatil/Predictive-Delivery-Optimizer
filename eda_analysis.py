"""
Exploratory Data Analysis Script
Run this to understand the data before building the model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modules.data_loader import DataLoader

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
sns.set_style('whitegrid')

# Load data
loader = DataLoader('data/raw/')
loader.load_all_datasets()

print("\n" + "="*80)
print("📊 NEXGEN LOGISTICS - DATA ANALYSIS REPORT")
print("="*80 + "\n")

# 1. Dataset Overview
print("1️⃣ DATASET OVERVIEW")
print("-" * 80)
summary_df = loader.get_data_summary()
print(summary_df.to_string(index=False))

# 2. Orders Dataset Analysis
print("\n\n2️⃣ ORDERS DATASET ANALYSIS")
print("-" * 80)
print(f"Total Orders: {len(loader.orders)}")
print(f"\nColumn Names:\n{loader.orders.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(loader.orders.head(3))

# Check for date columns and convert
date_columns = [col for col in loader.orders.columns if 'date' in col.lower()]
print(f"\nDate columns found: {date_columns}")

# 3. Delivery Performance Analysis
print("\n\n3️⃣ DELIVERY PERFORMANCE ANALYSIS")
print("-" * 80)
print(f"Total Delivery Records: {len(loader.delivery_performance)}")
print(f"\nColumns:\n{loader.delivery_performance.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(loader.delivery_performance.head(3))

# Check delivery status distribution
if 'Delivery_Status' in loader.delivery_performance.columns:
    print("\n📦 Delivery Status Distribution:")
    status_dist = loader.delivery_performance['Delivery_Status'].value_counts()
    print(status_dist)
    print(f"\nDelay Rate: {(status_dist.get('Delayed', 0) / len(loader.delivery_performance) * 100):.2f}%")

# 4. Missing Data Analysis
print("\n\n4️⃣ MISSING DATA ANALYSIS")
print("-" * 80)

def analyze_missing_data(df, name):
    """Analyze missing data in a dataset"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing_Count': missing.values,
        'Missing_Percent': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) > 0:
        print(f"\n{name}:")
        print(missing_df.to_string(index=False))
    else:
        print(f"\n{name}: No missing values ✅")

analyze_missing_data(loader.orders, "Orders")
analyze_missing_data(loader.delivery_performance, "Delivery Performance")
analyze_missing_data(loader.routes, "Routes & Distance")

# 5. Key Relationships
print("\n\n5️⃣ KEY RELATIONSHIPS BETWEEN DATASETS")
print("-" * 80)

# Check for common columns (join keys)
print("\nPotential Join Keys:")
print(f"Orders - Order ID columns: {[col for col in loader.orders.columns if 'order' in col.lower() and 'id' in col.lower()]}")
print(f"Delivery Performance - Order ID columns: {[col for col in loader.delivery_performance.columns if 'order' in col.lower() and 'id' in col.lower()]}")

# 6. Quick Statistics
print("\n\n6️⃣ QUICK STATISTICS")
print("-" * 80)

# Order value analysis
if 'Order_Value' in loader.orders.columns:
    print(f"\n💰 Order Value Statistics:")
    print(loader.orders['Order_Value'].describe())

# Customer ratings analysis
if 'Customer_Rating' in loader.delivery_performance.columns:
    print(f"\n⭐ Customer Rating Statistics:")
    print(loader.delivery_performance['Customer_Rating'].describe())
    print(f"Average Rating: {loader.delivery_performance['Customer_Rating'].mean():.2f}/5.0")

print("\n" + "="*80)
print("✅ EDA ANALYSIS COMPLETE!")
print("="*80)
