"""
NexGen Logistics - Predictive Delivery Optimizer Dashboard
Interactive Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import sys
import os
sys.path.append('.')

from modules.visualization import *

# Page configuration
st.set_page_config(
    page_title="NexGen Logistics - Delivery Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    .stAlert {
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    """Load processed dataset"""
    df = pd.read_csv('data/processed/master_dataset.csv')
    return df

@st.cache_resource
def load_model():
    """Load trained model"""
    model_package = joblib.load('models/hierarchical_final.pkl')
    return model_package

# Initialize
df = load_data()
model_package = load_model()

# Header
st.markdown('<h1 class="main-header">🚚 NexGen Logistics</h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #7f8c8d;">Predictive Delivery Optimizer</h3>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/truck.png", width=80)
    st.title("Navigation")
    
    page = st.radio(
        "Select View:",
        ["📊 Overview Dashboard", "🔮 Predict New Delivery", "📈 Analytics & Insights", "🎯 Corrective Actions"]
    )
    
    st.markdown("---")
    st.markdown("### 📋 Quick Stats")
    completed_orders = df[df['Is_Delayed'].notna()]
    st.metric("Total Orders", len(df))
    st.metric("Completed Deliveries", len(completed_orders))
    if len(completed_orders) > 0:
        delay_rate = (completed_orders['Is_Delayed'].sum() / len(completed_orders)) * 100
        st.metric("Current Delay Rate", f"{delay_rate:.1f}%")

# PAGE 1: Overview Dashboard
if page == "📊 Overview Dashboard":
    st.header("📊 Operations Overview")
    
    # Filter to completed deliveries
    completed_df = df[df['Delivery_Status'].notna()]
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_completed = len(completed_df)
        st.metric("✅ Completed Deliveries", total_completed)
    
    with col2:
        on_time = (completed_df['Delivery_Status'] == 'On-Time').sum()
        on_time_rate = (on_time / total_completed) * 100
        st.metric("🎯 On-Time Rate", f"{on_time_rate:.1f}%")
    
    with col3:
        avg_rating = completed_df['Customer_Rating'].mean()
        st.metric("⭐ Avg Customer Rating", f"{avg_rating:.2f}/5.0")
    
    with col4:
        delayed = completed_df['Is_Delayed'].sum()
        st.metric("⚠️ Delayed Orders", int(delayed), delta=f"-{(delayed/total_completed)*100:.1f}%", delta_color="inverse")
    
    st.markdown("---")
    
    # Visualizations Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_delay_distribution_chart(completed_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_carrier_performance_chart(completed_df), use_container_width=True)
    
    # Visualizations Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_route_heatmap(completed_df), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_delay_timeline(completed_df), use_container_width=True)

# PAGE 2: Predict New Delivery
elif page == "🔮 Predict New Delivery":
    st.header("🔮 Predict Delivery Delay Risk")
    st.markdown("Enter order details to predict delay probability and get corrective recommendations.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📦 Order Details")
        priority = st.selectbox("Priority", ['Express', 'Standard', 'Economy'])
        customer_segment = st.selectbox("Customer Segment", ['Enterprise', 'SMB', 'Individual'])
        order_value = st.number_input("Order Value (INR)", min_value=0.0, value=5000.0, step=100.0)
        
    with col2:
        st.subheader("🗺️ Route Details")
        origin = st.selectbox("Origin", df['Origin'].unique())
        destination = st.selectbox("Destination", df['Destination'].unique())
        distance = st.number_input("Distance (KM)", min_value=0.0, value=500.0, step=10.0)
        promised_days = st.number_input("Promised Delivery Days", min_value=1, value=5)
    
    with col3:
        st.subheader("⚙️ Operational Details")
        carrier = st.selectbox("Carrier", df['Carrier'].dropna().unique())
        has_weather = st.checkbox("Weather Impact Expected")
        has_special = st.checkbox("Special Handling Required")
        is_weekend = st.checkbox("Weekend Order")
    
    if st.button("🎯 Predict Delay Risk", type="primary"):
        st.markdown("---")
        st.info("⏳ Analyzing delivery risk factors using ML model...")
        
        try:
            # Get historical data for the selected route and carrier
            historical_data = df[
                (df['Origin'] == origin) & 
                (df['Destination'] == destination) &
                (df['Delivery_Status'].notna())
            ]
            
            carrier_data = df[
                (df['Carrier'] == carrier) & 
                (df['Delivery_Status'].notna())
            ]
            
            category_data = df[df['Delivery_Status'].notna()]  # All data for category
            
            # Calculate historical delay rates
            route_delay_rate = (historical_data['Is_Delayed'].sum() / len(historical_data)) if len(historical_data) > 0 else 0.3
            carrier_delay_rate = (carrier_data['Is_Delayed'].sum() / len(carrier_data)) if len(carrier_data) > 0 else 0.3
            carrier_rating = carrier_data['Customer_Rating'].mean() if len(carrier_data) > 0 else 3.5
            
            # Prepare features for Stage 1 model (same as training)
            feature_dict = {
                'Route_Historical_Delay_Rate': route_delay_rate,
                'Carrier_Historical_Delay_Rate': carrier_delay_rate,
                'Carrier_Avg_Rating': carrier_rating,
                'Promised_Delivery_Days': promised_days,
                'Distance_KM': distance,
                'Priority': 1 if priority == 'Express' else (2 if priority == 'Standard' else 3),
                'Customer_Segment': {'Enterprise': 0, 'SMB': 1, 'Individual': 2}[customer_segment],
                'Is_Express': 1 if priority == 'Express' else 0,
                'Has_Weather_Impact': 1 if has_weather else 0,
                'Has_Special_Handling': 1 if has_special else 0,
                'Is_Weekend_Order': 1 if is_weekend else 0,
                'Order_Day_of_Week': 5 if is_weekend else 2
            }
            
            # Get feature names from model
            feature_names = model_package.get('feature_names_stage1', [])
            
            # Create feature dataframe with correct feature order
            X_input = pd.DataFrame([feature_dict])
            X_input = X_input[feature_names]
            
            # Fill any missing features with median from training data
            for col in X_input.columns:
                if X_input[col].isna().any():
                    X_input[col] = df[col].median()
            
            # Get model components
            stage1_model = model_package['stage1_model']
            scaler_stage1 = model_package['scaler_stage1']
            
            # Scale input
            X_scaled = scaler_stage1.transform(X_input)
            
            # Predict using Stage 1 (binary: On-Time vs Delayed)
            delay_pred = stage1_model.predict(X_scaled)[0]
            delay_prob = stage1_model.predict_proba(X_scaled)[0, 1]  # Probability of delay
            
            # Display results
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.plotly_chart(create_prediction_gauge(delay_prob), use_container_width=True)
            
            with col2:
                # Determine risk level
                if delay_prob < 0.33:
                    risk_level = "🟢 LOW RISK"
                    color = "success"
                    description = "Delivery likely on-time"
                elif delay_prob < 0.66:
                    risk_level = "🟡 MEDIUM RISK"
                    color = "warning"
                    description = "Monitor closely"
                else:
                    risk_level = "🔴 HIGH RISK"
                    color = "error"
                    description = "Delay highly probable"
                
                if color == "success":
                    st.success(f"✅ **{risk_level}** - {description}")
                    st.markdown("**Confidence:** High")
                    st.markdown("**Actions:**")
                    st.markdown("- ✅ Standard processing can proceed")
                    st.markdown("- 📊 Continue normal monitoring")
                    
                elif color == "warning":
                    st.warning(f"⚠️ **{risk_level}** - {description}")
                    st.markdown("**Delay Probability:** {:.1%}".format(delay_prob))
                    st.markdown("**Recommended Actions:**")
                    st.markdown("- ✅ Confirm carrier availability")
                    st.markdown("- 🛣️ Prepare alternative route")
                    st.markdown("- 📞 Alert customer of possible delay")
                    st.markdown("- 📋 Update delivery SLA expectations")
                    
                else:  # High risk
                    st.error(f"🚨 **{risk_level}** - {description}")
                    st.markdown("**Delay Probability:** {:.1%}".format(delay_prob))
                    st.markdown("**URGENT - ACTIONS REQUIRED:**")
                    st.markdown("- ❌ Consider rescheduling or changing carrier")
                    st.markdown("- 📞 Contact customer proactively about potential delay")
                    st.markdown("- 🚚 Allocate backup vehicle if available")
                    st.markdown("- 📊 Escalate to operations manager immediately")
                    st.markdown("- 💰 Calculate impact on customer satisfaction & SLA penalties")
            
            # Additional insights
            st.markdown("---")
            st.subheader("📊 Prediction Details")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Delay Probability",
                    f"{delay_prob:.1%}",
                    help="Model's confidence that delivery will be delayed"
                )
            
            with col2:
                st.metric(
                    "Route Risk",
                    f"{route_delay_rate:.1%}",
                    help="Historical delay rate for this route"
                )
            
            with col3:
                st.metric(
                    "Carrier Reliability",
                    f"{(1-carrier_delay_rate):.1%}",
                    help="Historical on-time rate for this carrier"
                )
            
            with col4:
                st.metric(
                    "Distance Factor",
                    f"{distance:.0f} km",
                    help="Longer distances typically have higher delay risk"
                )
            
            # Risk factors breakdown
            st.markdown("---")
            st.subheader("🔍 Key Risk Factors Influencing This Prediction")
            
            risk_factors = []
            
            if route_delay_rate > 0.5:
                risk_factors.append(f"⚠️ **High-risk route**: {route_delay_rate:.0%} historical delay rate")
            
            if carrier_delay_rate > 0.5:
                risk_factors.append(f"⚠️ **Unreliable carrier**: {carrier_delay_rate:.0%} historical delay rate")
            
            if distance > 800:
                risk_factors.append(f"📏 **Very long distance**: {distance:.0f} km")
            
            if promised_days < 3:
                risk_factors.append(f"⚡ **Tight deadline**: Only {promised_days} days")
            
            if has_weather:
                risk_factors.append("🌧️ **Weather impact expected**")
            
            if has_special:
                risk_factors.append("📦 **Special handling required**")
            
            if priority == 'Express':
                risk_factors.append("⚡ **Express priority** (higher operational pressure)")
            
            if len(risk_factors) > 0:
                for factor in risk_factors:
                    st.warning(factor)
            else:
                st.success("✅ No major risk factors identified")
            
            # Model confidence
            st.markdown("---")
            
            # Determine confidence based on probability distance from 0.5
            confidence = max(delay_prob, 1 - delay_prob)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔮 Model Confidence")
                st.progress(confidence)
                st.caption(f"Confidence level: {confidence:.0%}")
            
            with col2:
                st.markdown("### 📈 What This Means")
                if confidence > 0.8:
                    st.info("🎯 **High Confidence Prediction** - Model is very certain about this outcome")
                elif confidence > 0.6:
                    st.info("📊 **Good Confidence** - Reasonable prediction reliability")
                else:
                    st.warning("⚠️ **Moderate Confidence** - Several factors could influence outcome")
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please ensure all required data is available and try again")


# PAGE 3: Analytics & Insights
elif page == "📈 Analytics & Insights":
    st.header("📈 Advanced Analytics & Model Insights")
    
    # Check model type
    is_hierarchical = 'feature_names_stage1' in model_package
    
    if is_hierarchical:
        st.info("🎯 Hierarchical Model Loaded: 2-Stage Architecture")
        
        # Stage 1 Features
        st.subheader("🔍 Stage 1: Binary Delay Detection Features")
        stage1_features = model_package.get('feature_names_stage1', [])
        
        if stage1_features:
            st.write(f"**{len(stage1_features)} features** used to predict if order will be delayed:")
            cols = st.columns(3)
            for i, feat in enumerate(stage1_features):
                cols[i % 3].markdown(f"• {feat}")
        
        # Stage 2 Features
        if model_package.get('feature_names_stage2'):
            st.subheader("🔍 Stage 2: Severity Prediction Features")
            stage2_features = model_package['feature_names_stage2']
            st.write(f"**{len(stage2_features)} features** used to predict delay severity:")
            cols = st.columns(3)
            for i, feat in enumerate(stage2_features):
                cols[i % 3].markdown(f"• {feat}")
        
        # Try to show feature importance from saved file
        if os.path.exists('data/processed/feature_coefficients.csv'):
            feature_importance = pd.read_csv('data/processed/feature_coefficients.csv')
            st.subheader("📊 Stage 1 Feature Coefficients (Logistic Regression)")
            st.plotly_chart(create_feature_importance_chart(
                feature_importance.rename(columns={'Abs_Coefficient': 'Importance'})
            ), use_container_width=True)
    
    else:
        # Single model
        st.info("🎯 Single-Stage Model Loaded")
        
        feature_names = model_package.get('feature_names', [])
        
        if feature_names:
            st.subheader("🔍 Predictive Features")
            st.write(f"**{len(feature_names)} features** used for predictions:")
            cols = st.columns(3)
            for i, feat in enumerate(feature_names):
                cols[i % 3].markdown(f"• {feat}")
        
        # Load feature importance
        if os.path.exists('data/processed/feature_importance.csv'):
            feature_importance = pd.read_csv('data/processed/feature_importance.csv')
            st.subheader("📊 Feature Importance")
            st.plotly_chart(create_feature_importance_chart(feature_importance), use_container_width=True)
    
    st.markdown("---")
    
    # Deep dive filters
    st.subheader("🎛️ Custom Analysis Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_carrier = st.multiselect("Filter by Carrier", df['Carrier'].dropna().unique())
    with col2:
        selected_priority = st.multiselect("Filter by Priority", df['Priority'].unique())
    with col3:
        selected_status = st.multiselect("Filter by Status", 
                                        ['On-Time', 'Slightly-Delayed', 'Severely-Delayed'])
    
    # Filtered data analysis
    filtered_df = df[df['Delivery_Status'].notna()].copy()
    
    if selected_carrier:
        filtered_df = filtered_df[filtered_df['Carrier'].isin(selected_carrier)]
    if selected_priority:
        filtered_df = filtered_df[filtered_df['Priority'].isin(selected_priority)]
    if selected_status:
        filtered_df = filtered_df[filtered_df['Delivery_Status'].isin(selected_status)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Filtered Statistics")
        st.metric("Orders in Selection", len(filtered_df))
        if len(filtered_df) > 0:
            delay_rate = (filtered_df['Is_Delayed'].sum() / len(filtered_df)) * 100
            st.metric("Delay Rate", f"{delay_rate:.1f}%")
    
    with col2:
        if len(filtered_df) > 0:
            st.markdown("### 💰 Cost Impact")
            avg_cost = filtered_df['Delivery_Cost_INR'].mean()
            st.metric("Avg Delivery Cost", f"₹{avg_cost:.2f}")
            total_cost = filtered_df['Delivery_Cost_INR'].sum()
            st.metric("Total Cost", f"₹{total_cost:,.2f}")

# PAGE 4: Corrective Actions
elif page == "🎯 Corrective Actions":
    st.header("🎯 AI-Powered Corrective Action Recommendations")
    
    st.markdown("""
    Based on historical delay patterns and current operations, here are the recommended actions
    to reduce delays and improve delivery performance.
    """)
    
    # High-risk routes
    st.subheader("🗺️ High-Risk Routes Requiring Attention")
    completed_df = df[df['Is_Delayed'].notna()]
    route_analysis = completed_df.groupby(['Origin', 'Destination']).agg({
        'Is_Delayed': ['mean', 'sum', 'count']
    }).reset_index()
    route_analysis.columns = ['Origin', 'Destination', 'Delay_Rate', 'Total_Delays', 'Total_Orders']
    route_analysis['Delay_Rate'] = route_analysis['Delay_Rate'] * 100
    high_risk_routes = route_analysis[route_analysis['Delay_Rate'] > 50].sort_values('Total_Delays', ascending=False)
    
    if len(high_risk_routes) > 0:
        st.dataframe(high_risk_routes, use_container_width=True)
        st.markdown("**Recommended Actions:**")
        st.markdown("- Conduct route optimization analysis")
        st.markdown("- Consider alternative carriers for these routes")
        st.markdown("- Increase promised delivery time by 1-2 days")
    
    st.markdown("---")
    
    # Carrier performance issues
    st.subheader("🚚 Carrier Performance Alerts")
    carrier_analysis = completed_df.groupby('Carrier').agg({
        'Is_Delayed': 'mean',
        'Customer_Rating': 'mean',
        'Order_ID': 'count'
    }).reset_index()
    carrier_analysis.columns = ['Carrier', 'Delay_Rate', 'Avg_Rating', 'Total_Orders']
    carrier_analysis['Delay_Rate'] = carrier_analysis['Delay_Rate'] * 100
    problem_carriers = carrier_analysis[(carrier_analysis['Delay_Rate'] > 50) | (carrier_analysis['Avg_Rating'] < 3.5)]
    
    if len(problem_carriers) > 0:
        st.warning("⚠️ The following carriers show performance issues:")
        st.dataframe(problem_carriers, use_container_width=True)
        st.markdown("**Recommended Actions:**")
        st.markdown("- Schedule performance review meetings")
        st.markdown("- Renegotiate SLAs with clear penalties")
        st.markdown("- Reduce order allocation by 20-30%")
        st.markdown("- Source backup carriers")
    
    st.markdown("---")
    
    # Estimated business impact
    st.subheader("💰 Potential Business Impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Monthly Delays", f"{int(completed_df['Is_Delayed'].sum())}")
        st.caption("Estimated cost: ₹3,50,000")
    
    with col2:
        improvement = 0.30
        reduced_delays = completed_df['Is_Delayed'].sum() * (1 - improvement)
        st.metric("If 30% Reduction", f"{int(reduced_delays)}", delta="-30%", delta_color="inverse")
        st.caption("Savings: ₹1,05,000/month")
    
    with col3:
        annual_savings = 105000 * 12
        st.metric("Annual Savings Potential", f"₹{annual_savings:,}")
        st.caption("Based on 30% delay reduction")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🚀 Built with Streamlit | NexGen Logistics Predictive Delivery Optimizer</p>
    <p>Model Accuracy: 73.7% | Delay Detection Rate: 72.9% | ROC-AUC: 0.845</p>
</div>
""", unsafe_allow_html=True)
