Supply Chain Delay Analysis & Prediction
📌 Project Overview

This project analyzes a global e-commerce supply chain dataset to identify the root causes of delivery delays, evaluate their impact on profitability, and build a machine learning model to predict late deliveries.

The project includes:

Exploratory Data Analysis (EDA)
Data Cleaning & Feature Engineering
Business KPI Analysis
Delay Bottleneck Detection
Root Cause Analysis
Time-based Trend Analysis
Machine Learning for Late Delivery Prediction

🎯 Business Problem

A global e-commerce company experiences inconsistent delivery performance where actual shipping times frequently exceed scheduled delivery dates. These delays reduce customer satisfaction and affect order profitability.

The objective is to:

Analyze delivery performance
Identify bottlenecks causing delays
Understand profitability impact
Predict late deliveries using Machine Learning

📊 Dataset
Rows: 172,765 (after cleaning)
Features: 20
Target Variable: Late_delivery_risk

The dataset contains order, customer, shipping, product, and profit information used for operational analysis.

🛠️ Tech Stack
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
SMOTE (Imbalanced Learning)
📈 Project Workflow
1. Data Cleaning
Removed duplicate and unnecessary columns
Handled missing values
Converted date columns
Removed cancelled orders
2. Feature Engineering

Created new business features such as:

Order Processing Time
Delivery Delay
Delay Flag
Order Month
Order Day
Order Hour
Profitability Flag
3. Exploratory Data Analysis

Performed analysis on:

Missing values
Delivery status
Shipping modes
Customer segments
Profitability distribution
Delay distribution
4. Business KPI Analysis

Calculated important KPIs including:

Total Orders
Late Deliveries
On-Time Delivery %
Late Delivery %
Total Profit
Total Profit Lost due to Delays
90th Percentile Delay
5. Root Cause Analysis

Analyzed delivery delays by:

Shipping Mode
Customer Segment
Department
Order Region
Order Status
Payment Type
6. Time-Based Analysis

Studied delay trends across:

Month
Day of Week
Hour of Day
7. Machine Learning

Model Used:

Random Forest Classifier

Techniques:

Frequency Encoding
Train-Test Split
SMOTE Oversampling
📈 Model Performance
Metric	Score
Accuracy	74%
Precision	78%
Recall	75%

The Random Forest model successfully predicts late deliveries with balanced performance across precision and recall.

💡 Key Business Insights
More than half of the orders experienced delivery delays.
Delivery delays significantly impacted overall profitability.
Shipping mode and order region were major contributors to delivery delays.
Delay rates varied across months, weekdays, and hours.
High-profit orders also experienced delays, indicating opportunities for operational improvement.
Predictive modeling can help proactively identify high-risk orders and improve delivery planning.

📂 Project Structure
Supply-Chain-Analysis/
│
├── data/
│   └── DataCoSupplyChainDataset.csv
│
├── notebooks/
│   └── Supply_Chain_Analysis.ipynb
│
├── images/
│   └── charts/
│
├── README.md
└── requirements.txt
