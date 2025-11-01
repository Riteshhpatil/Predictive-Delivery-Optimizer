# 🚚 NexGen Logistics - Predictive Delivery Optimizer

A cutting-edge **two-stage hierarchical ML system** that predicts delivery delays before they happen and recommends corrective actions, transforming NexGen from reactive firefighting to proactive operations.

## 📊 Project Overview

- **Problem**: 46.7% delivery delay rate impacting customer satisfaction
- **Solution**: Predictive ML model + Real-time alert system
- **Result**: 73.7% accuracy, 66.7% delay detection rate, ₹16L annual savings potential

## 🎯 Key Features

### Architecture
- **Stage 1**: Logistic Regression (Binary: On-Time vs Delayed)
- **Stage 2**: Rule-Based System (Severity: Slight vs Severe)
- **Result**: 3-class predictions with 0.81 ROC-AUC score

### Dashboard (Streamlit)
1. **Overview Dashboard** - Real-time KPIs, carrier performance, route analysis
2. **Real-Time Alerts** - Live monitoring of in-transit orders with risk scoring
3. **Predict New Delivery** - Interactive prediction for new orders
4. **Analytics & Insights** - Feature importance, custom filtering
5. **Corrective Actions** - Route optimization, carrier management, scenario planning

## 📈 Performance Metrics

| Metric | Value | Target |
|--------|-------|--------|
| 3-Class Accuracy | 65.3% | 60-70% ✅ |
| Binary Accuracy | 73.7% | 70%+ ✅ |
| ROC-AUC Score | 0.812 | >0.80 ✅ |
| Delay Detection | 66.7% | 70%+ |
| Precision | 75.0% | >70% ✅ |

## 💰 Business Impact

- **Annual Cost Savings**: ₹16,20,000
- **Delay Reduction**: 46.7% → 30% (target)
- **Customer Rating**: 3.64 → 4.2/5.0
- **ROI Payback Period**: 1.5 months

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **ML**: Scikit-learn, CatBoost
- **Dashboard**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy

