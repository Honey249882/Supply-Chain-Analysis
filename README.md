# 📦 Supply Chain Delay Analysis & Prediction

## 📌 Project Overview

This project analyzes a global e-commerce supply chain dataset to identify the root causes of delivery delays, evaluate their impact on profitability, and build a machine learning model to predict late deliveries.

---

## 🎯 Business Problem

A global e-commerce company experiences inconsistent delivery performance where actual shipping times frequently exceed scheduled delivery dates.

**Objectives:**
- Analyze delivery performance
- Identify bottlenecks causing delays
- Improve operational efficiency
- Predict late deliveries using Machine Learning

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- SMOTE

---

## 📊 Dataset

| Attribute | Value |
|-----------|-------|
| Records | 172,765 |
| Features | 20 |
| Target Variable | Late_delivery_risk |

---

## 📈 Project Workflow

### 1️⃣ Data Cleaning
- Removed unnecessary columns
- Converted date columns
- Removed cancelled orders
- Handled missing values

### 2️⃣ Feature Engineering
- Order Processing Time
- Delay
- Delay Flag
- Order Month
- Order Hour

### 3️⃣ Exploratory Data Analysis
- Profitability Distribution
- Delay Distribution
- Customer Segment Analysis
- Shipping Mode Analysis

### 4️⃣ Business KPIs
- Total Orders
- Late Deliveries
- On-Time Delivery %
- Late Delivery %
- Total Profit
- Total Loss due to Delays

### 5️⃣ Root Cause Analysis
- Shipping Mode
- Customer Segment
- Order Region
- Department
- Order Status

### 6️⃣ Time-Based Analysis
- Monthly Trends
- Weekly Trends
- Hourly Trends

### 7️⃣ Machine Learning

**Model Used**
- Random Forest Classifier

**Techniques**
- Frequency Encoding
- SMOTE
- Train-Test Split

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 74% |
| Precision | 78% |
| Recall | 75% |

---

## 💡 Business Insights

- More than half of the orders experienced delivery delays.
- Delivery delays significantly affected profitability.
- Shipping mode and order region were major delay drivers.
- Delay rates varied across months, weekdays, and hours.
- Random Forest achieved 74% prediction accuracy.

---

## 🚀 Future Improvements

- Deploy using Flask/FastAPI
- Power BI Dashboard
- Hyperparameter Tuning
- XGBoost & LightGBM

---

## 📂 Project Structure

```text
Supply-Chain-Analysis/
│
├── data/
│   └── DataCoSupplyChainDataset.csv
│
├── notebooks/
│   └── Supply_Chain_Analysis.ipynb
│
├── images/
│
├── README.md
│
└── requirements.txt
```

---

## ⭐ Author

**Honey**

Data Analyst | Data Scientist | Machine Learning Enthusiast
