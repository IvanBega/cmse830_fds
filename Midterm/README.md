# Final Project ReadMe

## Heart Disease Prediction Dashboard

An interactive Streamlit application for predicting heart disease risk using clinical data, country-level statistics, and socioeconomic indicators. This comprehensive tool enables data exploration, feature engineering, machine learning modeling, and result visualization in a user-friendly interface.

## The Datasets

### 1. First Dataset: Heart Disease Clinical Data
**Source**: [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/oktayrdeki/heart-disease)
- **Description**: Patient-level clinical data including age, BMI, blood pressure, cholesterol levels, stress levels, and heart disease status
- **Size**: 10,000+ patient records
- **Key Features**: Age, BMI, Blood Pressure, Cholesterol Level, Stress Level, Heart Disease Status
- **Purpose**: Primary dataset for training machine learning models

### 2. Second Dataset: Heart Disease Rate by Country
**Source**: [World Population Review](https://worldpopulationreview.com/country-rankings/heart-disease-rates-by-country)
- **Description**: Country-level heart disease statistics including prevalence, mortality rates, and DALYs (Disability-Adjusted Life Years)
- **Size**: 200+ countries
- **Key Features**: Prevalence rates (2021), Death counts (2021), Standardized rates (2022), DALYs
- **Purpose**: Geographic analysis and country-level risk factors

### 3. Third Dataset: Health Indicators by Country
**Source**: World Bank / WHO Socioeconomic Indicators
- **Description**: Comprehensive socioeconomic and health indicators across countries
- **Size**: 180+ countries with multiple years of data
- **Key Features**: GDP, Life Expectancy, Physicians per 1000, Urban Population, Infant Mortality, Unemployment Rate
- **Purpose**: Socioeconomic context and macro-level risk factor analysis

## Features

### Advanced Data Exploration
- Interactive visualizations including parallel coordinates, 3D scatter plots, and raincloud plots
- Comprehensive statistical analysis across all datasets
- Country discrepancy analysis and data merging

### Feature Engineering
- Age, BMI, and Blood Pressure binning/discretization
- Principal Component Analysis (PCA) for dimensionality reduction
- One-hot encoding and data normalization
- SMOTE oversampling for class imbalance handling

### Machine Learning Models
- Interpretable model with regularization options
- Non-linear model with feature importance analysis
- Interactive parameter tuning with real-time updates
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)

## How to Run

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/IvanBega/cmse830_fds.git
   cd Midterm
   
2. Install packages
    ```bash
    pip install -r requirements.txt
    
3. Run the Streamlit app
    ```bash
    streamlit run main.py
