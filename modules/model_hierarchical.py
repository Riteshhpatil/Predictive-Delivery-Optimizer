"""
Hierarchical Classification Models
Two-stage approach: Stage 1 (Any Delay?) → Stage 2 (Severity?)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class HierarchicalDelayPredictor:
    """
    Two-Stage Hierarchical Model:
    Stage 1: Predict if delivery will be delayed (binary: On-Time vs Any-Delay)
    Stage 2: For delayed orders, predict severity (Slightly vs Severely Delayed)
    
    This mimics real-world decision making and improves accuracy by specializing each stage.
    """
    
    def __init__(self, master_df):
        """Initialize with master dataset"""
        self.master_df = master_df
        self.stage1_model = None  # Binary: On-Time vs Delayed
        self.stage2_model = None  # Multi-class: Slight vs Severe
        self.scaler_stage1 = None
        self.scaler_stage2 = None
        self.feature_names_stage1 = None
        self.feature_names_stage2 = None
        self.label_encoders = {}
        
    def prepare_hierarchical_data(self):
        """
        Prepare data for two-stage hierarchical classification
        
        Returns:
        --------
        stage1_data : tuple (X, y)
            Data for Stage 1: Binary classification (On-Time=0, Any-Delay=1)
        stage2_data : tuple (X, y)
            Data for Stage 2: Delayed orders only (Slight=0, Severe=1)
        """
        print("\n🔧 Preparing Hierarchical Training Data...")
        
        df = self.master_df[self.master_df['Delivery_Status'].notna()].copy()
        print(f"✓ Total samples: {len(df)}")
        
        # STAGE 1 DATA: Binary classification (On-Time vs Any Delay)
        print("\n📊 STAGE 1 DATA: On-Time vs Any-Delay")
        
        # Stage 1 features: General operational features
        self.feature_names_stage1 = [
            'Route_Historical_Delay_Rate',
            'Carrier_Historical_Delay_Rate',
            'Carrier_Avg_Rating',
            'Promised_Delivery_Days',
            'Distance_KM',
            'Priority',
            'Customer_Segment',
            'Is_Express',
            'Has_Weather_Impact',
            'Has_Special_Handling',
            'Is_Weekend_Order',
            'Order_Day_of_Week'
        ]
        
        # Filter existing columns
        self.feature_names_stage1 = [f for f in self.feature_names_stage1 if f in df.columns]
        
        X_stage1 = df[self.feature_names_stage1].copy()
        
        # Encode categoricals
        categorical_cols = X_stage1.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X_stage1[col] = X_stage1[col].fillna('Unknown')
            X_stage1[col] = le.fit_transform(X_stage1[col].astype(str))
            self.label_encoders[f'stage1_{col}'] = le
        
        # Fill missing
        X_stage1 = X_stage1.fillna(X_stage1.median())
        
        # Stage 1 target: 0=On-Time, 1=Any Delay
        y_stage1 = (df['Delivery_Status'] != 'On-Time').astype(int)
        
        print(f"   - Features: {len(self.feature_names_stage1)}")
        print(f"   - On-Time: {sum(y_stage1==0)} samples")
        print(f"   - Any Delay: {sum(y_stage1==1)} samples")
        
        # STAGE 2 DATA: Severity classification (only for delayed orders)
        print("\n📊 STAGE 2 DATA: Slightly-Delayed vs Severely-Delayed")
        
        delayed_df = df[df['Delivery_Status'] != 'On-Time'].copy()
        print(f"   - Total delayed orders: {len(delayed_df)}")
        
        if len(delayed_df) > 20:  # Need enough samples for Stage 2
            # Stage 2 features: More detailed operational features
            self.feature_names_stage2 = [
                'Distance_KM',
                'Traffic_Delay_Hours',
                'Toll_Charges',
                'Fuel_Consumption_L',
                'Promised_Delivery_Days',
                'Route_Historical_Delay_Rate',
                'Carrier_Historical_Delay_Rate',
                'Has_Weather_Impact',
                'Is_Express',
                'Order_Value_INR'
            ]
            
            self.feature_names_stage2 = [f for f in self.feature_names_stage2 if f in delayed_df.columns]
            
            X_stage2 = delayed_df[self.feature_names_stage2].copy()
            
            # Encode categoricals for stage 2
            categorical_cols2 = X_stage2.select_dtypes(include=['object']).columns
            for col in categorical_cols2:
                le = LabelEncoder()
                X_stage2[col] = X_stage2[col].fillna('Unknown')
                X_stage2[col] = le.fit_transform(X_stage2[col].astype(str))
                self.label_encoders[f'stage2_{col}'] = le
            
            X_stage2 = X_stage2.fillna(X_stage2.median())
            
            # Stage 2 target: 0=Slightly-Delayed, 1=Severely-Delayed
            y_stage2 = (delayed_df['Delivery_Status'] == 'Severely-Delayed').astype(int)
            
            print(f"   - Features: {len(self.feature_names_stage2)}")
            print(f"   - Slightly-Delayed: {sum(y_stage2==0)} samples")
            print(f"   - Severely-Delayed: {sum(y_stage2==1)} samples")
        else:
            X_stage2, y_stage2 = None, None
            print("   ⚠️  Not enough delayed samples for Stage 2 training")
        
        return (X_stage1, y_stage1), (X_stage2, y_stage2)
    
    def train_stage1(self, X, y, test_size=0.25, random_state=42):
        """
        Train Stage 1: Binary classifier (On-Time vs Any-Delay)
        Uses Logistic Regression for interpretability
        """
        print("\n" + "="*80)
        print("🎯 TRAINING STAGE 1: On-Time vs Any-Delay (Logistic Regression)")
        print("="*80)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✓ Train: {len(X_train)} | Test: {len(X_test)}")
        print(f"✓ Train: On-Time={sum(y_train==0)}, Delayed={sum(y_train==1)}")
        
        # Scale
        self.scaler_stage1 = StandardScaler()
        X_train_scaled = self.scaler_stage1.fit_transform(X_train)
        X_test_scaled = self.scaler_stage1.transform(X_test)
        
        # Train with cross-validation to find best C
        print("\n🔬 Cross-validating regularization strength...")
        best_c, best_score = 0.1, 0
        
        for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
            model = LogisticRegression(C=C, max_iter=2000, random_state=random_state, 
                                      class_weight='balanced', solver='lbfgs')
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                best_c = C
        
        print(f"✓ Best C: {best_c} (CV Accuracy: {best_score:.3f})")
        
        # Train final model
        self.stage1_model = LogisticRegression(
            C=best_c, max_iter=2000, random_state=random_state,
            class_weight='balanced', solver='lbfgs'
        )
        self.stage1_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = self.stage1_model.predict(X_train_scaled)
        y_pred_test = self.stage1_model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\n📊 Stage 1 Performance:")
        print(f"   Training Accuracy: {train_acc:.3f}")
        print(f"   Test Accuracy: {test_acc:.3f}")
        print(f"   Overfitting Gap: {train_acc - test_acc:.3f}")
        
        print("\n" + classification_report(y_test, y_pred_test, 
                                          target_names=['On-Time', 'Any-Delay']))
        
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"\nConfusion Matrix:")
        print(f"             Predicted")
        print(f"            On-Time  Delayed")
        print(f"Actual On    {cm[0,0]:3d}      {cm[0,1]:3d}")
        print(f"       Del   {cm[1,0]:3d}      {cm[1,1]:3d}")
        
        return X_test, y_test, y_pred_test
    
    def train_stage2(self, X, y, test_size=0.25, random_state=42):
        """
        Train Stage 2: Severity classifier using SIMPLE Logistic Regression
        Better for very small datasets (<60 samples)
        """
        if X is None or len(X) < 20:
            print("\n⚠️  Skipping Stage 2: Insufficient delayed samples")
            return None, None, None
        
        print("\n" + "="*80)
        print("🎯 TRAINING STAGE 2: Slight vs Severe (Logistic Regression)")
        print("="*80)
        print(f"   💡 Using simple model: only {len(X)} samples")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✓ Train: {len(X_train)} | Test: {len(X_test)}")
        print(f"✓ Train: Slight={sum(y_train==0)}, Severe={sum(y_train==1)}")
        
        # Initialize scaler for Stage 2
        self.scaler_stage2 = StandardScaler()
        X_train_scaled = self.scaler_stage2.fit_transform(X_train)
        X_test_scaled = self.scaler_stage2.transform(X_test)
        
        # Find best regularization with cross-validation
        print("\n🔬 Testing regularization strengths...")
        best_c, best_score = 0.01, 0
        
        # Use 3-fold CV (small data)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        
        for C in [0.001, 0.01, 0.05, 0.1, 0.5]:
            model = LogisticRegression(
                C=C, 
                penalty='l2',
                max_iter=2000,
                random_state=random_state,
                class_weight={0: 1, 1: 2},  # Weight severe delays 2x
                solver='lbfgs'
            )
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
            
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                best_c = C
        
        print(f"✓ Best C: {best_c} (CV Accuracy: {best_score:.3f})")
        
        # Train final Stage 2 model
        self.stage2_model = LogisticRegression(
            C=best_c,
            penalty='l2',
            max_iter=2000,
            random_state=random_state,
            class_weight={0: 1, 1: 2},
            solver='lbfgs'
        )
        
        self.stage2_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = self.stage2_model.predict(X_train_scaled)
        y_pred_test = self.stage2_model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\n📊 Stage 2 Performance:")
        print(f"   Training Accuracy: {train_acc:.3f}")
        print(f"   Test Accuracy: {test_acc:.3f}")
        print(f"   Overfitting Gap: {train_acc - test_acc:.3f}")
        
        if train_acc - test_acc < 0.15:
            print("   ✅ Good generalization!")
        elif train_acc - test_acc < 0.25:
            print("   ⚠️  Acceptable generalization")
        else:
            print("   ❌ Still overfitting")
        
        print("\n" + classification_report(y_test, y_pred_test,
                                          target_names=['Slightly-Delayed', 'Severely-Delayed'],
                                          zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"\nConfusion Matrix:")
        print(f"               Predicted")
        print(f"            Slight  Severe")
        print(f"Actual Sli   {cm[0,0]:3d}     {cm[0,1]:3d}")
        print(f"       Sev   {cm[1,0]:3d}     {cm[1,1]:3d}")
        
        return X_test, y_test, y_pred_test
    def predict_severity_rules(self, X_delayed):
        """
        Rule-based severity prediction (no ML)
        Best for datasets <60 samples where ML overfits
        """
        predictions = []
        
        for idx, row in X_delayed.iterrows():
            severity_score = 0
            
            # Rule 1: High route delay rate → likely severe
            if row.get('Route_Historical_Delay_Rate', 0) > 0.6:
                severity_score += 2
            
            # Rule 2: High carrier delay rate → likely severe
            if row.get('Carrier_Historical_Delay_Rate', 0) > 0.6:
                severity_score += 2
            
            # Rule 3: Very long distance → likely severe
            if row.get('Distance_KM', 0) > 800:
                severity_score += 1
            
            # Rule 4: High traffic delay → likely severe
            if row.get('Traffic_Delay_Hours', 0) > 3:
                severity_score += 2
            
            # Rule 5: Weather + distance → severe
            if row.get('Has_Weather_Impact', 0) == 1 and row.get('Distance_KM', 0) > 500:
                severity_score += 1
            
            # Rule 6: Express with high delay history → severe
            if row.get('Is_Express', 0) == 1 and row.get('Carrier_Historical_Delay_Rate', 0) > 0.5:
                severity_score += 1
            
            # Threshold: score >= 4 → Severely-Delayed
            predictions.append(1 if severity_score >= 4 else 0)
        
        return np.array(predictions)
       
    def train_stage2_rules(self, X, y, test_size=0.25, random_state=42):
        """
        Evaluate rule-based Stage 2 (no training needed)
        """
        if X is None or len(X) < 20:
            print("\n⚠️  Skipping Stage 2: Insufficient delayed samples")
            return None, None, None
        
        print("\n" + "="*80)
        print("🎯 STAGE 2: Rule-Based Severity Prediction (No ML)")
        print("="*80)
        print(f"   💡 Using domain rules: {len(X)} samples")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✓ Train: {len(X_train)} | Test: {len(X_test)}")
        print(f"✓ Test: Slight={sum(y_test==0)}, Severe={sum(y_test==1)}")
        
        # Apply rules to test set
        y_pred_test = self.predict_severity_rules(X_test)
        
        # Also evaluate on train for comparison
        y_pred_train = self.predict_severity_rules(X_train)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\n📊 Rule-Based Performance:")
        print(f"   Training Accuracy: {train_acc:.3f}")
        print(f"   Test Accuracy: {test_acc:.3f}")
        print(f"   Gap: {abs(train_acc - test_acc):.3f}")
        print("   ✅ No overfitting with rules!")
        
        print("\n" + classification_report(y_test, y_pred_test,
                                          target_names=['Slightly-Delayed', 'Severely-Delayed'],
                                          zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"\nConfusion Matrix:")
        print(f"               Predicted")
        print(f"            Slight  Severe")
        print(f"Actual Sli   {cm[0,0]:3d}     {cm[0,1]:3d}")
        print(f"       Sev   {cm[1,0]:3d}     {cm[1,1]:3d}")
        
        # Store flag that we're using rules
        self.stage2_model = 'RULES'
        
        return X_test, y_test, y_pred_test

    def evaluate_hierarchical_system(self, X_full, y_full_3class):
        """
        Evaluate the complete hierarchical system end-to-end
        
        Parameters:
        -----------
        X_full : DataFrame
            Full feature set with both stage 1 and stage 2 features
        y_full_3class : Series
            3-class target: 0=On-Time, 1=Slightly-Delayed, 2=Severely-Delayed
        """
        print("\n" + "="*80)
        print("🎯 HIERARCHICAL SYSTEM: END-TO-END EVALUATION")
        print("="*80)
        
        # Stage 1 predictions
        X_stage1 = X_full[self.feature_names_stage1].fillna(X_full[self.feature_names_stage1].median())
        X_stage1_scaled = self.scaler_stage1.transform(X_stage1)
        stage1_pred = self.stage1_model.predict(X_stage1_scaled)
        
        # Initialize final predictions with Stage 1 results
        # 0 = On-Time, 1 = Slightly-Delayed (default for delayed), 2 = Severely-Delayed
        final_predictions = np.where(stage1_pred == 0, 0, 1)  # Start with on-time or slight delay
        
        # Stage 2 predictions (only for delayed orders)
        if self.stage2_model is not None:
            delayed_mask = (stage1_pred == 1)
            if delayed_mask.sum() > 0:
                X_delayed = X_full[delayed_mask][self.feature_names_stage2]
                X_delayed = X_delayed.fillna(X_delayed.median())
                stage2_pred = self.stage2_model.predict(X_delayed)
                
                # Update: if Stage 2 predicts severe (1), change to 2
                final_predictions[delayed_mask] = np.where(stage2_pred == 1, 2, 1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_full_3class, final_predictions)
        
        print(f"\n📊 Overall Hierarchical System Accuracy: {accuracy:.3f}")
        print("\n" + classification_report(y_full_3class, final_predictions,
                                          target_names=['On-Time', 'Slightly-Delayed', 'Severely-Delayed']))
        
        cm = confusion_matrix(y_full_3class, final_predictions)
        print(f"\nFull Confusion Matrix:")
        print(f"                 Predicted")
        print(f"             On-Time  Slight  Severe")
        print(f"Actual On      {cm[0,0]:3d}     {cm[0,1]:3d}     {cm[0,2]:3d}")
        print(f"       Sli     {cm[1,0]:3d}     {cm[1,1]:3d}     {cm[1,2]:3d}")
        print(f"       Sev     {cm[2,0]:3d}     {cm[2,1]:3d}     {cm[2,2]:3d}")
        
        # Compare to binary delay detection
        binary_actual = (y_full_3class > 0).astype(int)
        binary_pred = (final_predictions > 0).astype(int)
        binary_acc = accuracy_score(binary_actual, binary_pred)
        
        print(f"\n💡 Binary Delay Detection Accuracy: {binary_acc:.3f}")
        print(f"   (Simplified: On-Time vs Any-Delay)")
        
        return accuracy, final_predictions
    
    def save_model(self, filepath='models/hierarchical_model.pkl'):
        """Save hierarchical model"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_package = {
            'stage1_model': self.stage1_model,
            'stage2_model': self.stage2_model,
            'scaler_stage1': self.scaler_stage1,
            'scaler_stage2': self.scaler_stage2,
            'feature_names_stage1': self.feature_names_stage1,
            'feature_names_stage2': self.feature_names_stage2,
            'label_encoders': self.label_encoders
        }
        
        joblib.dump(model_package, filepath)
        print(f"\n💾 Hierarchical model saved to: {filepath}")
