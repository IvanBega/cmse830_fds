import streamlit as st

st.set_page_config(page_title="Footnotes", page_icon="📋")

st.title("Footnotes: Project Rubric Summary")

st.header("1. Data Collection and Preparation")

st.subheader("Added the third data set, in addition to two existing ones from the Midterm project")
st.markdown("""
**Third Dataset - Country Indicators:**
- **Source**: World Bank/WHO socioeconomic indicators
- **Features**: GDP, life expectancy, physicians per 1000, urban population, infant mortality, unemployment
""")

st.write("The following steps were performed:")
st.markdown("""
1. Median imputation and data validation
2. Statistical methods to identify abnormal values
3. Encoding of categorical variables
4. Manual mapping for country name variations
""")

# =============================================================================
# RUBRIC 2: EXPLORATORY DATA ANALYSIS AND VISUALIZATION (15%)
# =============================================================================

st.header("2. Exploratory Data Analysis and Visualization")

st.markdown("""
Advanced Visualizations:

1. Parallel Coordinates Plot
2. Interactive Bubble Map
3. Raincloud Plot
4. 3D Scatter Plot
6. Confusion Matrices
7. ROC Curves
8. Bar/Line Charts
""")

st.subheader("Statistical analysis")
st.markdown("""
**Statistical Analysis Features:**
- Descriptive statistics for all datasets
- Correlation matrices and heatmaps
- Distribution analysis by subgroups
- Hypothesis testing visualizations
- Cross-dataset statistical comparisons
- Model performance metrics (accuracy, precision, recall, F1, AUC)
""")

st.header("3. Data Processing and Feature Engineering")

st.subheader("Implement multiple feature engineering techniques")
st.markdown("""
**Feature Engineering Implemented:**

1. **Binning/Discretization**:
   - Age: Young/Middle/Senior/Elderly categories
   - BMI: Underweight/Normal/Overweight/Obese
   - Blood Pressure: Clinical categories (Normal/Stage1/Stage2/Crisis)

2. **Encoding**:
   - One-hot encoding for categorical variables
   - Label encoding for ordinal features

3. **Normalization**:
   - StandardScaler for machine learning features
   - Min-max scaling for visualization
""")

st.markdown("""
**Advanced Transformations:**

1. **Principal Component Analysis (PCA)**:
   - Dimensionality reduction of socioeconomic indicators
   - Explained variance analysis (85%+ with 2 components)
   - Biplot visualization of feature contributions

2. **SMOTE Oversampling**:
   - Handling class imbalance in heart disease prediction
   - Synthetic minority class generation
   - Improved model performance on minority class
""")

st.header("4. Model Development and Evaluation")

st.subheader("Implement at least two different machine learning models")
st.markdown("""
**Models Implemented:**

1. **Logistic Regression**:
   - Interpretable linear model with regularization
   - Feature coefficient analysis
   - Probability outputs for risk assessment

2. **Decision Tree Classifier**:
   - Non-linear model with decision rules
   - Feature importance ranking
   - Visual tree-like decision structure
""")

st.markdown("""
**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (Area Under Curve)
- Confusion Matrices (TP, TN, FP, FN)
- Cross-validation scores (k-fold)
- Training vs testing performance comparison

""")

st.markdown("""
**Validation Techniques:**
1. **Train-Test Split**: Configurable ratio (10-40% test size)
2. **Cross-Validation**: k-fold validation (3-10 folds)
3. **Hyperparameter Tuning**: Interactive parameter adjustment
4. **Class Weight Balancing**: Handling imbalanced datasets
5. **Random State Control**: Reproducible results
""")

st.header("5. Streamlit App Development")

st.markdown("""

1. **Feature Selection**: Multi-select for choosing features
2. **Data Sampling Options**: Checkboxes for oversampling/scaling
3. **Visualization Controls**: Plot customization options
4. **Dataset Selectors**: Switch between different data views
5. **Reset Functionality**: Reset all settings to defaults
6. **Cache Management**: Clear cache buttons
""")

st.markdown("""
**Documentation Features:**

1. **Inline Explanations**: Every visualization and model has detailed explanations
2. **Parameter Guidance**: Tooltips and recommendations for all settings
3. **Statistical Interpretation**: Plain-language explanation of results
""")

st.subheader("Implement advanced Streamlit features")
st.markdown("""
**Advanced Features Implemented:**

1. **Caching**:
   - `@st.cache_data` for data loading and processing
   - `@st.cache_resource` for trained models
   - TTL (time-to-live) management for performance

2. **Session State**:
   - Persistent user settings across interactions
   - State management for all model parameters
   - Multi-page state consistency

""")

st.header("GitHub Repository and Documentation")
st.markdown("""
            1. The final project repository is available at the link: https://github.com/IvanBega/cmse830_fds/tree/main/Midterm
            2. There is a detailed ReadMe file in the project folder, which explains the datasets used, as well as installation guide for the Streamlit project
            """)