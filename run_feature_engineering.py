"""
Feature Engineering Execution Script
"""

import pandas as pd
from modules.data_loader import DataLoader
from modules.feature_engineering import FeatureEngineer

# Load data
print("="*80)
print("🚀 NEXGEN LOGISTICS - FEATURE ENGINEERING PIPELINE")
print("="*80)

loader = DataLoader('data/raw/')
loader.load_all_datasets()

# Initialize feature engineer
engineer = FeatureEngineer(loader)

# Execute feature engineering pipeline
engineer.create_master_dataset()
engineer.engineer_delay_features()
engineer.engineer_operational_features()
engineer.engineer_time_features()
engineer.add_aggregated_features()

# Get final dataset
final_df = engineer.get_feature_summary()

# Save processed dataset
output_path = 'data/processed/master_dataset.csv'
final_df.to_csv(output_path, index=False)
print(f"\n💾 Master dataset saved to: {output_path}")

# Display sample features
print("\n📋 SAMPLE OF ENGINEERED FEATURES:")
print("-" * 80)
feature_cols = ['Order_ID', 'Is_Delayed', 'Delay_Days', 'Route_Complexity_Score', 
                'Carrier_Historical_Delay_Rate', 'Is_Express', 'Has_Weather_Impact']
display_cols = [col for col in feature_cols if col in final_df.columns]
print(final_df[display_cols].head(10))

print("\n" + "="*80)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("="*80)
