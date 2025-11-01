"""
Feature Engineering Module
Creates predictive features from raw data
"""

import pandas as pd
import numpy as np
from datetime import datetime

class FeatureEngineer:
    def __init__(self, loader):
        """Initialize with DataLoader instance"""
        self.loader = loader
        
    def create_master_dataset(self):
        """Merge all datasets into a single analytical dataset"""
        print("\n🔧 Starting Feature Engineering...")
        
        # Start with orders as base
        master = self.loader.orders.copy()
        print(f"✓ Base dataset: {len(master)} orders")
        
        # Merge delivery performance (LEFT JOIN - some orders still in transit)
        master = master.merge(
            self.loader.delivery_performance,
            on='Order_ID',
            how='left'
        )
        print(f"✓ After merging delivery_performance: {len(master)} rows")
        
        # Merge route/distance data
        master = master.merge(
            self.loader.routes,
            on='Order_ID',
            how='left'
        )
        print(f"✓ After merging routes_distance: {len(master)} rows")
        
        # Merge cost breakdown
        master = master.merge(
            self.loader.costs,
            on='Order_ID',
            how='left'
        )
        print(f"✓ After merging cost_breakdown: {len(master)} rows")
        
        self.master_df = master
        print(f"\n✅ Master dataset created: {master.shape[0]} rows × {master.shape[1]} columns")
        return self
    
    def engineer_delay_features(self):
        """Create delay-related features"""
        print("\n🎯 Engineering delay prediction features...")
        
        df = self.master_df.copy()
        
        # 1. TARGET VARIABLE: Binary delay flag
        # On-Time = 0, Slightly-Delayed or Severely-Delayed = 1
        df['Is_Delayed'] = df['Delivery_Status'].apply(
            lambda x: 0 if x == 'On-Time' else 1 if pd.notna(x) else np.nan
        )
        
        # 2. Delay severity (for analysis)
        df['Delay_Severity'] = df['Delivery_Status'].map({
            'On-Time': 0,
            'Slightly-Delayed': 1,
            'Severely-Delayed': 2
        })
        
        # 3. Actual delay in days
        df['Delay_Days'] = df['Actual_Delivery_Days'] - df['Promised_Delivery_Days']
        df['Delay_Days'] = df['Delay_Days'].fillna(0)
        
        # 4. Delay percentage
        df['Delay_Percentage'] = (df['Delay_Days'] / df['Promised_Delivery_Days']) * 100
        df['Delay_Percentage'] = df['Delay_Percentage'].fillna(0)
        
        print(f"✓ Created delay features: Is_Delayed, Delay_Severity, Delay_Days, Delay_Percentage")
        
        self.master_df = df
        return self
    
    def engineer_operational_features(self):
        """Create operational and risk features"""
        print("\n⚙️ Engineering operational features...")
        
        df = self.master_df.copy()
        
        # 1. Route complexity score
        if 'Distance_KM' in df.columns and 'Traffic_Delay_Hours' in df.columns:
            df['Route_Complexity_Score'] = (
                df['Distance_KM'] / 100 +  # Normalize distance
                df['Traffic_Delay_Hours'].fillna(0) * 2  # Traffic is 2x weight
            )
        
        # 2. Cost efficiency ratio
        if 'Delivery_Cost_INR' in df.columns and 'Order_Value_INR' in df.columns:
            df['Cost_to_Value_Ratio'] = df['Delivery_Cost_INR'] / (df['Order_Value_INR'] + 1)  # +1 to avoid division by zero
        
        # 3. Fuel efficiency indicator
        if 'Fuel_Consumption_Liters' in df.columns and 'Distance_KM' in df.columns:
            df['Fuel_Efficiency_KMPL'] = df['Distance_KM'] / (df['Fuel_Consumption_Liters'] + 0.1)
        
        # 4. Has weather impact flag
        if 'Weather_Impact' in df.columns:
            df['Has_Weather_Impact'] = df['Weather_Impact'].notna().astype(int)
        
        # 5. Has special handling flag
        if 'Special_Handling' in df.columns:
            df['Has_Special_Handling'] = df['Special_Handling'].notna().astype(int)
        
        # 6. Quality issue flag
        if 'Quality_Issue' in df.columns:
            df['Has_Quality_Issue'] = (df['Quality_Issue'] != 'Perfect').astype(int)
        
        # 7. Express delivery flag
        df['Is_Express'] = (df['Priority'] == 'Express').astype(int)
        df['Is_Economy'] = (df['Priority'] == 'Economy').astype(int)
        
        # 8. High value order flag (top 25%)
        if 'Order_Value_INR' in df.columns:
            value_threshold = df['Order_Value_INR'].quantile(0.75)
            df['Is_High_Value_Order'] = (df['Order_Value_INR'] > value_threshold).astype(int)
        
        print(f"✓ Created {8} operational features")
        
        self.master_df = df
        return self
    
    def engineer_time_features(self):
        """Create time-based features"""
        print("\n📅 Engineering time features...")
        
        df = self.master_df.copy()
        
        # Convert Order_Date to datetime
        df['Order_Date'] = pd.to_datetime(df['Order_Date'])
        
        # Extract time components
        df['Order_Day_of_Week'] = df['Order_Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['Order_Month'] = df['Order_Date'].dt.month
        df['Order_Day'] = df['Order_Date'].dt.day
        df['Is_Weekend_Order'] = (df['Order_Day_of_Week'] >= 5).astype(int)
        df['Is_Month_End'] = (df['Order_Day'] >= 25).astype(int)
        
        print(f"✓ Created 5 time-based features")
        
        self.master_df = df
        return self
    
    def add_aggregated_features(self):
        """Add carrier and route performance aggregations"""
        print("\n📈 Adding aggregated performance features...")
        
        df = self.master_df.copy()
        
        # Carrier performance metrics (only for completed deliveries)
        completed_df = df[df['Delivery_Status'].notna()].copy()
        
        if 'Carrier' in df.columns and len(completed_df) > 0:
            # Average delay rate by carrier
            carrier_delay_rate = completed_df.groupby('Carrier')['Is_Delayed'].mean().to_dict()
            df['Carrier_Historical_Delay_Rate'] = df['Carrier'].map(carrier_delay_rate)
            
            # Average rating by carrier
            carrier_rating = completed_df.groupby('Carrier')['Customer_Rating'].mean().to_dict()
            df['Carrier_Avg_Rating'] = df['Carrier'].map(carrier_rating)
        
        # Route performance (Origin-Destination pairs)
        if 'Origin' in df.columns and 'Destination' in df.columns and len(completed_df) > 0:
            completed_df['Route'] = completed_df['Origin'] + '_to_' + completed_df['Destination']
            df['Route'] = df['Origin'] + '_to_' + df['Destination']
            
            route_delay_rate = completed_df.groupby('Route')['Is_Delayed'].mean().to_dict()
            df['Route_Historical_Delay_Rate'] = df['Route'].map(route_delay_rate)
        
        # Product category delay patterns
        if 'Product_Category' in df.columns and len(completed_df) > 0:
            category_delay_rate = completed_df.groupby('Product_Category')['Is_Delayed'].mean().to_dict()
            df['Category_Historical_Delay_Rate'] = df['Product_Category'].map(category_delay_rate)
        
        print(f"✓ Created aggregated performance features")
        
        self.master_df = df
        return self
    
    def get_feature_summary(self):
        """Get summary of engineered features"""
        completed = self.master_df[self.master_df['Is_Delayed'].notna()]
        
        print("\n" + "="*80)
        print("📊 FEATURE ENGINEERING SUMMARY")
        print("="*80)
        print(f"\nTotal Records: {len(self.master_df)}")
        print(f"Records with Delay Data (Training Set): {len(completed)}")
        print(f"Records without Delay Data (To Predict): {len(self.master_df) - len(completed)}")
        
        if len(completed) > 0:
            print(f"\n🎯 TARGET VARIABLE DISTRIBUTION:")
            print(f"   - On-Time Deliveries: {(completed['Is_Delayed']==0).sum()} ({(completed['Is_Delayed']==0).sum()/len(completed)*100:.1f}%)")
            print(f"   - Delayed Deliveries: {(completed['Is_Delayed']==1).sum()} ({(completed['Is_Delayed']==1).sum()/len(completed)*100:.1f}%)")
            
            print(f"\n📏 KEY METRICS:")
            print(f"   - Average Delay (when delayed): {completed[completed['Is_Delayed']==1]['Delay_Days'].mean():.2f} days")
            print(f"   - Max Delay: {completed['Delay_Days'].max():.0f} days")
            print(f"   - Avg Customer Rating: {completed['Customer_Rating'].mean():.2f}/5.0")
        
        print(f"\n✅ Total Features Created: {len(self.master_df.columns)}")
        print("="*80)
        
        return self.master_df
