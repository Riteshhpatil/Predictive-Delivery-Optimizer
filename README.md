# 🚚 NexGen Logistics - Predictive Delivery Optimizer

A machine learning system that predicts delivery delays before they happen, enabling proactive decision-making and reducing operational costs.

## 📊 Quick Overview

- **Problem**: 46.7% delivery delay rate impacting customer satisfaction
- **Solution**: Two-stage hierarchical ML model with real-time Streamlit dashboard
- **Impact**: 73.7% accuracy | 0.81 ROC-AUC | ₹16L annual savings potential

## 🎯 Features

### Dashboard (4 Pages)
1. **Overview Dashboard** - Real-time KPIs, carrier & route performance analysis
3. **Predict New Delivery** - Interactive predictions for new orders with risk assessment
4. **Analytics & Insights** - Feature importance, custom filtering, trend analysis
5. **Corrective Actions** - Route optimization recommendations, carrier performance alerts

### Model Architecture
- **Stage 1**: Logistic Regression (Binary: On-Time vs Delayed)
- **Stage 2**: Rule-Based System (Severity: Slightly vs Severely Delayed)
- **Result**: 3-class predictions optimized for small datasets

## 📈 Performance

| Metric | Score |
|--------|-------|
| Accuracy | 73.7% |
| ROC-AUC | 0.812 |
| Delay Detection | 66.7% |
| Precision | 75.0% |

## 🛠️ Technology Stack

- Python 3.10+
- Streamlit (Dashboard)
- Scikit-learn (ML)
- Pandas & NumPy (Data Processing)
- Plotly (Visualizations)

## 📁 Project Structure
# 🚚 NexGen Logistics - Predictive Delivery Optimizer

A machine learning system that predicts delivery delays before they happen, enabling proactive decision-making and reducing operational costs.

## 📊 Quick Overview

- **Problem**: 46.7% delivery delay rate impacting customer satisfaction
- **Solution**: Two-stage hierarchical ML model with real-time Streamlit dashboard
- **Impact**: 73.7% accuracy | 0.81 ROC-AUC | ₹16L annual savings potential

## 🎯 Features

### Dashboard (5 Pages)
1. **Overview Dashboard** - Real-time KPIs, carrier & route performance analysis
2. **Real-Time Alerts** - Monitor in-transit orders with risk scoring & corrective actions
3. **Predict New Delivery** - Interactive predictions for new orders with risk assessment
4. **Analytics & Insights** - Feature importance, custom filtering, trend analysis
5. **Corrective Actions** - Route optimization recommendations, carrier performance alerts

### Model Architecture
- **Stage 1**: Logistic Regression (Binary: On-Time vs Delayed)
- **Stage 2**: Rule-Based System (Severity: Slightly vs Severely Delayed)
- **Result**: 3-class predictions optimized for small datasets

## 📈 Performance

| Metric | Score |
|--------|-------|
| Accuracy | 73.7% |
| ROC-AUC | 0.812 |
| Delay Detection | 66.7% |
| Precision | 75.0% |

## 🛠️ Technology Stack

- Python 3.10+
- Streamlit (Dashboard)
- Scikit-learn (ML)
- Pandas & NumPy (Data Processing)
- Plotly (Visualizations)

## 📁 Project Structure

# 🚚 NexGen Logistics - Predictive Delivery Optimizer

A machine learning system that predicts delivery delays before they happen, enabling proactive decision-making and reducing operational costs.

## 📊 Quick Overview

- **Problem**: 46.7% delivery delay rate impacting customer satisfaction
- **Solution**: Two-stage hierarchical ML model with real-time Streamlit dashboard
- **Impact**: 73.7% accuracy | 0.81 ROC-AUC | ₹16L annual savings potential

## 🎯 Features

### Dashboard (5 Pages)
1. **Overview Dashboard** - Real-time KPIs, carrier & route performance analysis
2. **Real-Time Alerts** - Monitor in-transit orders with risk scoring & corrective actions
3. **Predict New Delivery** - Interactive predictions for new orders with risk assessment
4. **Analytics & Insights** - Feature importance, custom filtering, trend analysis
5. **Corrective Actions** - Route optimization recommendations, carrier performance alerts

### Model Architecture
- **Stage 1**: Logistic Regression (Binary: On-Time vs Delayed)
- **Stage 2**: Rule-Based System (Severity: Slightly vs Severely Delayed)
- **Result**: 3-class predictions optimized for small datasets

## 📈 Performance

| Metric | Score |
|--------|-------|
| Accuracy | 73.7% |
| ROC-AUC | 0.812 |
| Delay Detection | 66.7% |
| Precision | 75.0% |

## 🛠️ Technology Stack

- Python 3.10+
- Streamlit (Dashboard)
- Scikit-learn (ML)
- Pandas & NumPy (Data Processing)
- Plotly (Visualizations)

## 📁 Project Structure
predictive_delivery_optimizer/ 
├── app.py                      # Main Streamlit dashboard 
├── train_hierarchical_final.py # Model training script 
├── requirements.txt            # Dependencies 
├── modules/ 
│   ├── model_hierarchical.py  # Two-stage model 
│   ├── visualization.py       # Plotly charts 
│   └── feature_engineering.py # Feature creation 
├── data/ 
│   ├── raw/                   # Raw datasets 
│   └── processed/             # Processed data & predictions 
└── models/ 
└── hierarchical_final.pkl # Trained model


## 🚀 Quick Start

### Installation

1. Clone repository
git clone https://github.com/Riteshhpatil/Predictive-Delivery-Optimizer.git cd Predictive-Delivery-Optimizer
2. Create virtual environment
python -m venv venv source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt


### Training the Model
python train_hierarchical_final.py


**Output:**
- Trains Stage 1 (Logistic Regression) & Stage 2 (Rule-Based)
- Saves model to `models/hierarchical_final.pkl`
- Saves predictions to `data/processed/hierarchical_predictions.csv`

### Running the Dashboard
streamlit run app.py


Then open: **http://localhost:8501**

## 📊 How to Use

### On Overview Dashboard
- View real-time metrics (on-time rate, customer rating, delays)
- Analyze carrier performance & route patterns

### For New Order Prediction
1. Enter order details (priority, route, distance, etc.)
2. Click "🎯 Predict Delay Risk"
3. Get delay probability + risk assessment + recommendations

### For Operations Management
- Check real-time alerts for in-transit orders
- View corrective action recommendations
- Monitor high-risk routes & problematic carriers

## 💡 Key Insights

- **Route history** is the strongest predictor of delays
- **Longer distances** (>800km) have 40% higher delay risk
- **Weather impact** increases delay probability by 15%
- **Express orders** correlate with higher operational pressure

## 📊 Model Details

### Training Data
- 200 total orders
- 150 completed deliveries (labeled)
- 112 training samples
- 38 test samples

### Features (Stage 1: 12)
- Route historical delay rate
- Carrier historical delay rate
- Carrier average rating
- Distance, priority, promised days
- Weather, special handling flags

## 🔄 Workflow
User Input → Feature Extraction → Stage 1 Prediction (Delay?) → Stage 2 Assessment (Severity?) → Risk Score → Recommendations


## ⚡ Example Predictions

**Low Risk Order:**
- Distance: 300km | Priority: Standard | Good carrier history
- Prediction: ✅ 25% delay probability | Recommended: Standard processing

**High Risk Order:**
- Distance: 900km | Priority: Express | Weather issues
- Prediction: 🚨 78% delay probability | Recommended: Carrier change, customer contact


## 🎯 Next Steps

1. Deploy to pilot with operations team
2. Retrain model monthly with new data
3. Integrate with existing order management system
4. Add GPS tracking for real-time predictions
5. Expand to international routes

## 📝 Requirements

See `requirements.txt`:
pandas==2.1.4 numpy==1.26.3 scikit-learn==1.4.0 streamlit==1.31.0 plotly==5.18.0 catboost==1.2.0 joblib==1.3.2


## 🔐 License

MIT License - See LICENSE file

## 👨‍💼 Author

**Ritesh Patil**  
Data Analyst | ML Enthusiast  
GitHub: [@Riteshhpatil](https://github.com/Riteshhpatil)
LinkedIn : [@Riteshhpatil](https://www.linkedin.com/in/ritesh-patil-5aaa722a0/)
---

**Status**: ✅ Production Ready | v1.0 | November 2025

*"Transform logistics from reactive firefighting to proactive intelligence."*

