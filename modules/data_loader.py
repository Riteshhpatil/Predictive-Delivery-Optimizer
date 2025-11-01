"""
Data Loader Module
Handles loading and initial processing of all datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self, data_path='/Users/riteshpatil/Desktop/predictive_delivery_optimizer/data/raw/'):
        self.data_path = Path(data_path)
        
    def load_all_datasets(self):
        """Load all 7 CSV files"""
        print("📦 Loading datasets...")
        
        # Load each dataset
        self.orders = pd.read_csv(self.data_path / 'orders.csv')
        self.delivery_performance = pd.read_csv(self.data_path / 'delivery_performance.csv')
        self.routes = pd.read_csv(self.data_path / 'routes_distance.csv')
        self.vehicles = pd.read_csv(self.data_path / 'vehicle_fleet.csv')
        self.warehouse = pd.read_csv(self.data_path / 'warehouse_inventory.csv')
        self.feedback = pd.read_csv(self.data_path / 'customer_feedback.csv')
        self.costs = pd.read_csv(self.data_path / 'cost_breakdown.csv')
        
        print("✅ All datasets loaded successfully!")
        return self
    
    def get_data_summary(self):
        """Generate summary statistics for all datasets"""
        datasets = {
            'orders': self.orders,
            'delivery_performance': self.delivery_performance,
            'routes_distance': self.routes,
            'vehicle_fleet': self.vehicles,
            'warehouse_inventory': self.warehouse,
            'customer_feedback': self.feedback,
            'cost_breakdown': self.costs
        }
        
        summary = []
        for name, df in datasets.items():
            summary.append({
                'Dataset': name,
                'Rows': len(df),
                'Columns': len(df.columns),
                'Missing (%)': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
                'Memory (KB)': round(df.memory_usage(deep=True).sum() / 1024, 2)
            })
        
        return pd.DataFrame(summary)
