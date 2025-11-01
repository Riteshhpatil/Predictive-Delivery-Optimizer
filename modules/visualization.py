"""
Visualization Module for Streamlit Dashboard
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_delay_distribution_chart(df):
    """Create delay status distribution pie chart"""
    status_counts = df['Delivery_Status'].value_counts()
    
    fig = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title='Delivery Status Distribution',
        color_discrete_map={
            'On-Time': '#2ecc71',
            'Slightly-Delayed': '#f39c12',
            'Severely-Delayed': '#e74c3c'
        }
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_carrier_performance_chart(df):
    """Create carrier performance comparison"""
    carrier_stats = df.groupby('Carrier').agg({
        'Is_Delayed': 'mean',
        'Customer_Rating': 'mean',
        'Order_ID': 'count'
    }).reset_index()
    carrier_stats.columns = ['Carrier', 'Delay_Rate', 'Avg_Rating', 'Total_Orders']
    carrier_stats['Delay_Rate'] = carrier_stats['Delay_Rate'] * 100
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=carrier_stats['Carrier'],
        y=carrier_stats['Delay_Rate'],
        name='Delay Rate (%)',
        marker_color='#e74c3c',
        yaxis='y',
        text=carrier_stats['Delay_Rate'].round(1),
        textposition='auto'
    ))
    
    fig.add_trace(go.Scatter(
        x=carrier_stats['Carrier'],
        y=carrier_stats['Avg_Rating'],
        name='Avg Rating',
        marker_color='#3498db',
        yaxis='y2',
        mode='lines+markers',
        line=dict(width=3)
    ))
    
    fig.update_layout(
        title='Carrier Performance: Delay Rate vs Customer Rating',
        xaxis=dict(title='Carrier'),
        yaxis=dict(title='Delay Rate (%)', side='left'),
        yaxis2=dict(title='Average Rating (1-5)', side='right', overlaying='y', range=[0, 5]),
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_route_heatmap(df):
    """Create route delay heatmap"""
    route_delays = df.groupby(['Origin', 'Destination'])['Is_Delayed'].mean().reset_index()
    route_delays['Delay_Rate'] = route_delays['Is_Delayed'] * 100
    
    pivot_table = route_delays.pivot(index='Origin', columns='Destination', values='Delay_Rate')
    
    fig = px.imshow(
        pivot_table,
        labels=dict(x="Destination", y="Origin", color="Delay Rate (%)"),
        title='Route Delay Heatmap: Origin → Destination',
        color_continuous_scale='RdYlGn_r',
        aspect="auto"
    )
    
    return fig

def create_feature_importance_chart(importance_df, top_n=10):
    """Create feature importance horizontal bar chart"""
    top_features = importance_df.head(top_n).sort_values('Importance')
    
    fig = px.bar(
        top_features,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top {top_n} Features Driving Delay Predictions',
        labels={'Importance': 'Feature Importance Score', 'Feature': ''},
        color='Importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(showlegend=False, height=400)
    return fig

def create_delay_timeline(df):
    """Create delay trends over time"""
    df_time = df[df['Order_Date'].notna()].copy()
    df_time['Order_Date'] = pd.to_datetime(df_time['Order_Date'])
    df_time['Week'] = df_time['Order_Date'].dt.to_period('W').astype(str)
    
    weekly_delays = df_time.groupby('Week').agg({
        'Is_Delayed': 'mean',
        'Order_ID': 'count'
    }).reset_index()
    weekly_delays['Delay_Rate'] = weekly_delays['Is_Delayed'] * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=weekly_delays['Week'],
        y=weekly_delays['Delay_Rate'],
        mode='lines+markers',
        name='Delay Rate',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Delay Rate Trend Over Time',
        xaxis_title='Week',
        yaxis_title='Delay Rate (%)',
        hovermode='x unified'
    )
    
    return fig

def create_prediction_gauge(delay_probability):
    """Create gauge chart for delay probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=delay_probability * 100,
        title={'text': "Delay Risk Score"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "#2ecc71"},
                {'range': [33, 66], 'color': "#f39c12"},
                {'range': [66, 100], 'color': "#e74c3c"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig
