import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from datasource import get_model_ready_data

st.set_page_config(page_title="Model Development & Evaluation")

st.title("Model Development & Evaluation")
st.markdown("This page demonstrates machine learning model development, evaluation, and comparison for heart disease prediction.")
st.markdown("Note: your settings are saved and cached using the hyperlink")

# =============================================================================
# IMPROVED SESSION STATE INITIALIZATION WITH QUERY PARAMS
# =============================================================================

def initialize_session_state():
    """Initialize session state from query params or defaults"""
    query_params = st.query_params
    
    # Initialize with defaults or query params
    defaults = {
        'use_oversampling': True,
        'scale_features': True,
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 5,
        'data_loaded': False,
        'models_trained': False,
        'lr_params': {'C': 1.0, 'max_iter': 200, 'penalty': 'l2'},
        'dt_params': {'max_depth': 5, 'min_samples_split': 2, 'criterion': 'gini'}
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            # Try to get from query params
            param_value = query_params.get(key, None)
            if param_value:
                if isinstance(default_value, bool):
                    st.session_state[key] = param_value[0].lower() == 'true'
                elif isinstance(default_value, (int, float)):
                    try:
                        st.session_state[key] = type(default_value)(param_value[0])
                    except:
                        st.session_state[key] = default_value
                elif key == 'lr_params':
                    # Reconstruct lr_params from query params
                    lr_C = float(query_params.get('lr_C', ['1.0'])[0])
                    lr_max_iter = int(query_params.get('lr_max_iter', ['200'])[0])
                    lr_penalty = query_params.get('lr_penalty', ['l2'])[0]
                    st.session_state.lr_params = {
                        'C': lr_C,
                        'max_iter': lr_max_iter,
                        'penalty': lr_penalty
                    }
                elif key == 'dt_params':
                    # Reconstruct dt_params from query params
                    dt_max_depth = int(query_params.get('dt_max_depth', ['5'])[0])
                    dt_min_samples_split = int(query_params.get('dt_min_samples_split', ['2'])[0])
                    dt_criterion = query_params.get('dt_criterion', ['gini'])[0]
                    st.session_state.dt_params = {
                        'max_depth': dt_max_depth,
                        'min_samples_split': dt_min_samples_split,
                        'criterion': dt_criterion
                    }
                else:
                    st.session_state[key] = default_value
            else:
                st.session_state[key] = default_value
    
    # Initialize selected_features separately (can't easily serialize to URL)
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None

def update_query_params():
    """Update URL query parameters with current session state"""
    params = {}
    
    # Add simple parameters
    simple_params = ['use_oversampling', 'scale_features', 'test_size', 
                    'random_state', 'cv_folds']
    
    for param in simple_params:
        if param in st.session_state:
            params[param] = str(st.session_state[param])
    
    # Add nested parameters
    if 'lr_params' in st.session_state:
        for k, v in st.session_state.lr_params.items():
            params[f'lr_{k}'] = str(v)
    
    if 'dt_params' in st.session_state:
        for k, v in st.session_state.dt_params.items():
            params[f'dt_{k}'] = str(v)
    
    st.query_params.update(**params)

# Initialize session state
initialize_session_state()

# =============================================================================
# CACHING FUNCTIONS (UNCHANGED)
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=True)
def load_cached_model_data(use_oversampling):
    """Cache the model data loading to avoid reloading on every interaction"""
    return get_model_ready_data(use_oversampling=use_oversampling)

@st.cache_data(ttl=3600, show_spinner=True)
def cache_train_test_split(_df, features, test_size, random_state, use_oversampling):
    """Cache the train-test split results"""
    X = _df[features]
    y = _df['Heart Disease Status_encoded']
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=features)
    
    # For oversampled data, handle special split
    if 'is_synthetic' in _df.columns and use_oversampling:
        real_data = _df[_df['is_synthetic'] == 0]
        synthetic_data = _df[_df['is_synthetic'] == 1]
        
        X_real = real_data[features]
        y_real = real_data['Heart Disease Status_encoded']
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=test_size, random_state=random_state, stratify=y_real
        )
        
        X_train = pd.concat([X_train_real, synthetic_data[features]])
        y_train = pd.concat([y_train_real, synthetic_data['Heart Disease Status_encoded']])
        X_test = X_test_real
        y_test = y_test_real
        
        synthetic_info = {
            'synthetic_data': synthetic_data,
            'real_data': real_data
        }
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        synthetic_info = {}
    
    return X_train, X_test, y_train, y_test, synthetic_info

@st.cache_data(ttl=3600, show_spinner=True)
def cache_scaled_data(X_train, X_test, features, scale_features):
    """Cache the scaling operation"""
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_final = pd.DataFrame(X_train_scaled, columns=features)
        X_test_final = pd.DataFrame(X_test_scaled, columns=features)
        return X_train_final, X_test_final, scaler
    else:
        return X_train, X_test, None

@st.cache_resource(ttl=3600, show_spinner=True)
def cache_lr_model(X_train, y_train, C, max_iter, penalty, random_state, use_oversampling):
    """Cache the trained Logistic Regression model"""
    lr_model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        penalty=penalty,
        solver='saga' if penalty == 'elasticnet' else 'liblinear',
        random_state=random_state,
        class_weight='balanced' if not use_oversampling else None
    )
    lr_model.fit(X_train, y_train)
    return lr_model

@st.cache_resource(ttl=3600, show_spinner=True)
def cache_dt_model(X_train, y_train, max_depth, min_samples_split, criterion, random_state, use_oversampling):
    """Cache the trained Decision Tree model"""
    dt_model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=random_state,
        class_weight='balanced' if not use_oversampling else None
    )
    dt_model.fit(X_train, y_train)
    return dt_model

@st.cache_data(ttl=3600, show_spinner=True)
def cache_cross_val_scores(_model, X_train, y_train, cv_folds):
    """Cache cross-validation results"""
    return cross_val_score(_model, X_train, y_train, cv=cv_folds, scoring='accuracy')

# =============================================================================
# MODIFIED WIDGETS WITH PERSISTENCE
# =============================================================================

# Data sampling option - with callback to update query params
st.subheader("Use Oversampling and Scaling?")

col1, col2 = st.columns(2)
with col1:
    def update_oversampling():
        st.session_state.use_oversampling = st.session_state.oversampling_checkbox
        update_query_params()
        # Clear relevant caches when oversampling changes
        st.cache_data.clear()
    
    use_oversampling = st.checkbox(
        "Use SMOTE Oversampling to handle class imbalance", 
        value=st.session_state.use_oversampling,
        key='oversampling_checkbox',
        on_change=update_oversampling
    )

with col2:
    def update_scaling():
        st.session_state.scale_features = st.session_state.scaling_checkbox
        update_query_params()
    
    scale_features = st.checkbox(
        "Apply StandardScaler", 
        value=st.session_state.scale_features,
        key='scaling_checkbox',
        on_change=update_scaling
    )

# Load modeling data with caching
with st.spinner('Loading data...'):
    df_model = load_cached_model_data(st.session_state.use_oversampling)
    st.session_state.data_loaded = True

st.subheader("Dataset Stats for Machine Learning Model")
st.write(f"**Shape:** {df_model.shape}")
st.write(f"**Number of Features:** {len(df_model.columns) - 1}")  # Excluding target
st.write(f"**Using Oversampling:** {'Yes' if st.session_state.use_oversampling else 'No'}")

# Show target distribution
st.subheader("Distribution of Heart Disease")
if 'is_synthetic' in df_model.columns and st.session_state.use_oversampling:
    # Show synthetic vs real data distribution
    synthetic_counts = df_model['is_synthetic'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Real Data", f"{synthetic_counts[0]}")
    with col2:
        st.metric("Synthetic Data", f"{synthetic_counts[1]}")
    with col3:
        st.metric("Total Samples", len(df_model))

target_counts = df_model['Heart Disease Status_encoded'].value_counts()
target_percentages = (target_counts / len(df_model) * 100).round(2)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("No Heart Disease", f"{target_counts[0]}")
with col2:
    st.metric("Heart Disease", f"{target_counts[1]}")
with col3:
    imbalance_ratio = target_counts[0] / target_counts[1] if target_counts[1] > 0 else 0
    st.metric("Class Balance", f"{imbalance_ratio:.2f}:1" if not st.session_state.use_oversampling else "1:1 (Balanced)")

# Feature selection with persistence
st.subheader("Which Features to Use?")
exclude_cols = ['Heart Disease Status_encoded', 'is_synthetic'] if 'is_synthetic' in df_model.columns else ['Heart Disease Status_encoded']
feature_cols = [col for col in df_model.columns if col not in exclude_cols]

# Use a session state key that persists
if st.session_state.selected_features is None:
    default_features = [col for col in feature_cols if col not in ['Age_bin', 'BMI_category', 'BP_category']][:10]
    st.session_state.selected_features = default_features

# Modified multiselect with callback
def update_selected_features():
    st.session_state.selected_features = st.session_state.features_multiselect
    # Clear cache since features changed
    st.cache_data.clear()

selected_features = st.multiselect(
    "Features",
    options=feature_cols,
    default=st.session_state.selected_features,
    key='features_multiselect',
    on_change=update_selected_features
)

# Prepare feature matrix X and target y
if len(selected_features) > 0:
    # Show feature information
    st.write(f"**Selected {len(selected_features)} features:**")
    feature_info = pd.DataFrame({
        'Feature': selected_features,
        'Type': ['Categorical' if '_encoded' in col or '_bin' in col or '_category' in col else 'Numerical' for col in selected_features],
        'Missing Values': df_model[selected_features].isnull().sum().values,
        'Unique Values': [df_model[col].nunique() for col in selected_features]
    })
    st.dataframe(feature_info)
    
    # Train-test split with caching
    st.subheader("Train-Test Split")
    
    def update_test_size():
        st.session_state.test_size = st.session_state.test_size_slider
        update_query_params()
        # Clear cache since test size changed
        st.cache_data.clear()
    
    test_size = st.slider(
        "Select test set size:", 
        0.1, 0.4, 
        st.session_state.test_size, 
        0.05,
        key='test_size_slider',
        on_change=update_test_size
    )
    
    # ADDED EXPLANATION: Test set size recommendation
    st.write("**Test Set Size Recommendation:**")
    st.write("""
    The default test size of 20% (0.2) is a good balance for most datasets:
    - **Increase test size** (e.g., to 30-40%) if you have a large dataset (>10,000 samples) and want more reliable performance estimates
    - **Decrease test size** (e.g., to 10%) if you have a small dataset (<1,000 samples) to preserve more data for training
    - The 80/20 split is a common standard that provides enough data for both training and reliable evaluation
    """)
    
    def update_random_state():
        st.session_state.random_state = st.session_state.random_state_input
        update_query_params()
        # Clear cache since random state changed
        st.cache_data.clear()
    
    random_state = st.number_input(
        "Random seed:", 
        0, 100, 
        st.session_state.random_state,
        key='random_state_input',
        on_change=update_random_state
    )
    
    # Cache the train-test split
    with st.spinner('Splitting data into train/test sets...'):
        X_train, X_test, y_train, y_test, synthetic_info = cache_train_test_split(
            df_model, selected_features, test_size, random_state, st.session_state.use_oversampling
        )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Samples", len(X_train))
    with col2:
        st.metric("Test Samples", len(X_test))
    with col3:
        train_real = len(X_train) - (len(synthetic_info.get('synthetic_data', [])) if 'synthetic_data' in synthetic_info else 0)
        st.metric("Real Training Samples", train_real)
    with col4:
        if st.session_state.use_oversampling and 'synthetic_data' in synthetic_info:
            st.metric("Synthetic Training Samples", len(synthetic_info['synthetic_data']))
    
    # Cache scaling operation
    with st.spinner('Scaling features...' if st.session_state.scale_features else 'Preparing features...'):
        X_train_final, X_test_final, _ = cache_scaled_data(
            X_train, X_test, selected_features, st.session_state.scale_features
        )
    
    # =============================================================================
    # SECTION 2: MODEL 1 - LOGISTIC REGRESSION
    # =============================================================================
    
    st.header("Model 1: Logistic Regression")
    
    # ADDED EXPLANATION: What is Logistic Regression
    st.write("""
    **What is Logistic Regression?**
    Logistic Regression is a statistical model used for binary classification problems (like predicting heart disease yes/no). 
    Unlike linear regression which predicts continuous values, logistic regression predicts the probability of an outcome 
    using a sigmoid function that outputs values between 0 and 1. It's interpretable because each feature has a coefficient 
    showing its relationship with the outcome - positive coefficients increase disease risk, negative coefficients decrease it.
    """)
    
    st.markdown(f"""
    **Using {'oversampled' if st.session_state.use_oversampling else 'original'} data.
    """)
    
    # Model training
    st.subheader("Model Training Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        def update_C_value():
            st.session_state.lr_params['C'] = st.session_state.C_value_input
            update_query_params()
            # Clear model cache since parameter changed
            st.cache_resource.clear()
        
        C_value = st.number_input(
            "Regularization (C):", 
            0.01, 10.0, 
            st.session_state.lr_params['C'], 
            0.1,
            key='C_value_input',
            on_change=update_C_value
        )
    with col2:
        def update_max_iter():
            st.session_state.lr_params['max_iter'] = st.session_state.max_iter_input
            update_query_params()
            st.cache_resource.clear()
        
        max_iter = st.number_input(
            "Max iterations:", 
            100, 1000, 
            st.session_state.lr_params['max_iter'], 
            50,
            key='max_iter_input',
            on_change=update_max_iter
        )
    with col3:
        def update_penalty():
            st.session_state.lr_params['penalty'] = st.session_state.penalty_selectbox
            update_query_params()
            st.cache_resource.clear()
        
        penalty_type = st.selectbox(
            "Penalty:", 
            ['l2', 'l1', 'elasticnet'],
            index=['l2', 'l1', 'elasticnet'].index(st.session_state.lr_params['penalty']),
            key='penalty_selectbox',
            on_change=update_penalty
        )
    
    # Cache the trained model
    with st.spinner('Training Logistic Regression model...'):
        lr_model = cache_lr_model(
            X_train_final, y_train, C_value, max_iter, penalty_type, 
            random_state, st.session_state.use_oversampling
        )
    
    # Predictions
    y_train_pred = lr_model.predict(X_train_final)
    y_test_pred = lr_model.predict(X_test_final)
    y_test_prob = lr_model.predict_proba(X_test_final)[:, 1]
    
    # Add parameter explanations
    st.write("""
    **Regularization (C):**
    Controls overfitting by penalizing large coefficients. 
    - **Lower C** (e.g., 0.01): Strong regularization, simpler model, less overfitting
    - **Higher C** (e.g., 10.0): Weak regularization, more complex model, risk of overfitting
    - **Default (1.0)**: Balanced approach
    """)
    
    st.write("""
    **Max Iterations:**
    Maximum number of iterations for the solver to converge.
    - **Too low**: Model may not converge (warning message appears)
    - **Too high**: Unnecessary computation time
    - **Default (200)**: Usually sufficient for most datasets
    Increase if you see "ConvergenceWarning" in the output.
    """)
    
    st.write("""
    **Penalty Type:**         
    Type of regularization to apply:
    - **L2 (Ridge)**: Penalizes squared coefficients. Tends to keep all features with smaller coefficients.
    - **L1 (Lasso)**: Penalizes absolute coefficients. Can shrink some coefficients to zero (feature selection).
    - **ElasticNet**: Combines L1 and L2 penalties. Best of both worlds but slower.
    For interpretability with feature selection, try L1. For general use, L2 is standard.
    """)
    
    # Model coefficients
    st.subheader("Feature Importance (Coefficients)")
    coefficients = pd.DataFrame({
        'Feature': selected_features,
        'Coefficient': lr_model.coef_[0],
        'Absolute Value': np.abs(lr_model.coef_[0]),
        'Direction': ['Positive' if c > 0 else 'Negative' for c in lr_model.coef_[0]]
    }).sort_values('Absolute Value', ascending=False)
    
    st.dataframe(coefficients.round(4))
    
    # Visualize coefficients
    fig, ax = plt.subplots(figsize=(10, 6))
    top_coeffs = coefficients.head(10)
    colors = ['green' if c > 0 else 'red' for c in top_coeffs['Coefficient']]
    ax.barh(top_coeffs['Feature'], top_coeffs['Coefficient'], color=colors)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Top 10 Feature Coefficients (Logistic Regression)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    st.pyplot(fig)
    
    # =============================================================================
    # SECTION 3: MODEL 2 - DECISION TREE
    # =============================================================================
    
    st.header("Model 2: Decision Tree Classifier")
    
    # ADDED EXPLANATION: What is a Decision Tree and split criteria
    st.write("""
    **What is a Decision Tree?**
    A Decision Tree is a non-linear model that makes predictions by learning simple decision rules from data features.
    It works by repeatedly splitting the data into subsets based on feature values, creating a tree-like structure.
    Each internal node represents a "test" on a feature, each branch represents the outcome, and each leaf node represents a prediction.
    
    **Split Criteria - Gini vs Entropy:**
    - **Gini Impurity**: Measures how often a randomly chosen element would be incorrectly classified. Faster to compute, default in sklearn.
    - **Entropy (Information Gain)**: Measures the amount of information or uncertainty. Tends to create more balanced trees.
    Both usually give similar results, but Gini is slightly faster computationally.
    """)
    
    st.markdown(f"""
    **Using {'oversampled' if st.session_state.use_oversampling else 'original'} data.
    """)
    
    # Model training
    st.subheader("Model Training Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        def update_max_depth():
            st.session_state.dt_params['max_depth'] = st.session_state.dt_depth_slider
            update_query_params()
            st.cache_resource.clear()
        
        max_depth = st.slider(
            "Max tree depth:", 
            1, 20, 
            st.session_state.dt_params['max_depth'], 
            key="dt_depth_slider",
            on_change=update_max_depth
        )
    with col2:
        def update_min_samples_split():
            st.session_state.dt_params['min_samples_split'] = st.session_state.dt_split_slider
            update_query_params()
            st.cache_resource.clear()
        
        min_samples_split = st.slider(
            "Min samples to split:", 
            2, 20, 
            st.session_state.dt_params['min_samples_split'], 
            key="dt_split_slider",
            on_change=update_min_samples_split
        )
    with col3:
        def update_criterion():
            st.session_state.dt_params['criterion'] = st.session_state.dt_criterion_selectbox
            update_query_params()
            st.cache_resource.clear()
        
        criterion = st.selectbox(
            "Split criterion:", 
            ['gini', 'entropy'],
            index=['gini', 'entropy'].index(st.session_state.dt_params['criterion']),
            key="dt_criterion_selectbox",
            on_change=update_criterion
        )
    
    # Cache the trained model
    with st.spinner('Training Decision Tree model...'):
        dt_model = cache_dt_model(
            X_train_final, y_train, max_depth, min_samples_split, 
            criterion, random_state, st.session_state.use_oversampling
        )
    
    # Predictions
    y_train_pred_dt = dt_model.predict(X_train_final)
    y_test_pred_dt = dt_model.predict(X_test_final)
    y_test_prob_dt = dt_model.predict_proba(X_test_final)[:, 1]
    
    st.session_state.models_trained = True
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': dt_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.dataframe(importance_df.round(4))
    
    # Visualize feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    top_importance = importance_df.head(10)
    ax.barh(top_importance['Feature'], top_importance['Importance'])
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 10 Feature Importances (Decision Tree)')
    st.pyplot(fig)
    
    # =============================================================================
    # SECTION 4: MODEL EVALUATION & COMPARISON
    # =============================================================================
    
    st.header("Model Evaluation & Comparison")
    
    # Calculate metrics for both models
    def calculate_metrics(y_true, y_pred, y_prob, model_name):
        return {
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
            'ROC-AUC': roc_auc_score(y_true, y_prob) if y_prob is not None else None
        }
    
    # Calculate metrics
    lr_metrics = calculate_metrics(y_test, y_test_pred, y_test_prob, 'Logistic Regression')
    dt_metrics = calculate_metrics(y_test, y_test_pred_dt, y_test_prob_dt, 'Decision Tree')
    
    # Create comparison dataframe
    metrics_df = pd.DataFrame([lr_metrics, dt_metrics])
    
    st.subheader("Performance Metrics Comparison")
    st.dataframe(metrics_df.round(4))
    
    # Visual comparison
    st.subheader("Visual Comparison")
    
    # Bar chart comparison
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        axes[idx].bar(['Logistic Regression', 'Decision Tree'], 
                     metrics_df[metric].values, 
                     color=['blue', 'green'], alpha=0.7)
        axes[idx].set_title(f'{metric} Comparison')
        axes[idx].set_ylabel(metric)
        axes[idx].set_ylim([0, 1])
        # Add value labels
        for i, v in enumerate(metrics_df[metric].values):
            axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Confusion matrices
    st.subheader("Confusion Matrices")
    
    # ADDED EXPLANATION: What is a Confusion Matrix
    st.write("""
    **What is a Confusion Matrix?**
    A confusion matrix is a table that shows how well a classification model performs on test data. It compares:
    - **True Positives (TP)**: Correctly predicted heart disease cases (top-right)
    - **True Negatives (TN)**: Correctly predicted no disease cases (top-left)
    - **False Positives (FP)**: Incorrectly predicted as disease (bottom-left, Type I error)
    - **False Negatives (FN)**: Missed heart disease cases (top-right, Type II error)
    
    In healthcare, false negatives (missing heart disease) are often more serious than false positives.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Logistic Regression:**")
        cm_lr = confusion_matrix(y_test, y_test_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Logistic Regression Confusion Matrix')
        st.pyplot(fig)
    
    with col2:
        st.write("**Decision Tree:**")
        cm_dt = confusion_matrix(y_test, y_test_pred_dt)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['No Disease', 'Disease'],
                    yticklabels=['No Disease', 'Disease'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Decision Tree Confusion Matrix')
        st.pyplot(fig)
    
    # ROC Curves
    st.subheader("ROC Curves Comparison")
    
    # ADDED EXPLANATION: What is a ROC Curve
    st.write("""
    **What is a ROC Curve?**
    ROC (Receiver Operating Characteristic) Curve shows the diagnostic ability of a binary classifier as its discrimination threshold is varied.
    - **X-axis**: False Positive Rate (1 - Specificity) - how many healthy people are incorrectly flagged
    - **Y-axis**: True Positive Rate (Sensitivity/Recall) - how many sick people are correctly identified
    - **AUC (Area Under Curve)**: Measures overall performance. Higher is better:
      * 0.5 = Random guessing (diagonal line)
      * 0.7-0.8 = Acceptable
      * 0.8-0.9 = Excellent
      * >0.9 = Outstanding
    The curve shows the trade-off between sensitivity and specificity at different threshold levels.
    """)
    
    # Calculate ROC curves
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_prob)
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_test_prob_dt)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    roc_auc_dt = auc(fpr_dt, tpr_dt)
    
    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_lr, tpr_lr, color='blue', lw=2, 
            label=f'Logistic Regression (AUC = {roc_auc_lr:.3f})')
    ax.plot(fpr_dt, tpr_dt, color='green', lw=2, 
            label=f'Decision Tree (AUC = {roc_auc_dt:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # =============================================================================
    # SECTION 5: OVERSAMPLING IMPACT ANALYSIS
    # =============================================================================
    
    if st.session_state.use_oversampling:
        
        # Compare with hypothetical no-oversampling scenario
        st.subheader("Class Balance Comparison")
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        # Before oversampling (estimated)
        before_counts = [target_counts.sum() * 0.85, target_counts.sum() * 0.15]  # Estimated 85:15 split
        ax[0].pie(before_counts, labels=['No Disease', 'Disease'], autopct='%1.1f%%',
                 colors=['lightblue', 'lightcoral'])
        ax[0].set_title('Estimated Distribution\n(Before Oversampling)')
        
        # After oversampling
        ax[1].pie([target_counts[0], target_counts[1]], labels=['No Disease', 'Disease'], 
                 autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        ax[1].set_title('Actual Distribution\n(After Oversampling)')
        
        st.pyplot(fig)
    
    # =============================================================================
    # SECTION 6: CROSS-VALIDATION & MODEL SELECTION
    # =============================================================================
    
    st.header("Cross-Validation & Model Selection")
    
    st.subheader("Cross-Validation Results")
    
    def update_cv_folds():
        st.session_state.cv_folds = st.session_state.cv_folds_slider
        update_query_params()
        # Clear cache since cv folds changed
        st.cache_data.clear()
    
    cv_folds = st.slider(
        "Number of CV folds:", 
        3, 10, 
        st.session_state.cv_folds,
        key="cv_folds_slider",
        on_change=update_cv_folds
    )
    
    # Cache cross-validation results
    with st.spinner('Performing cross-validation...'):
        lr_cv_scores = cache_cross_val_scores(lr_model, X_train_final, y_train, cv_folds)
        dt_cv_scores = cache_cross_val_scores(dt_model, X_train_final, y_train, cv_folds)
    
    cv_results = pd.DataFrame({
        'Fold': range(1, cv_folds + 1),
        'Logistic Regression': lr_cv_scores,
        'Decision Tree': dt_cv_scores
    })
    
    st.write("**Cross-Validation Accuracy Scores:**")
    st.dataframe(cv_results.round(4))
    
    # Cross-validation summary
    cv_summary = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree'],
        'Mean CV Accuracy': [lr_cv_scores.mean(), dt_cv_scores.mean()],
        'Std CV Accuracy': [lr_cv_scores.std(), dt_cv_scores.std()],
        'Min CV Accuracy': [lr_cv_scores.min(), dt_cv_scores.min()],
        'Max CV Accuracy': [lr_cv_scores.max(), dt_cv_scores.max()]
    })
    
    st.write("**Cross-Validation Summary:**")
    st.dataframe(cv_summary.round(4))
    
    # Show current URL with settings
    st.subheader("Share Your Configuration")
    current_url = st.query_params.to_dict()
    if current_url:
        st.write("**Current settings saved in URL:**")
        st.json(current_url)
        st.write("You can Bookmark this page to save your settings!")
    
else:
    st.warning("Please select at least one feature to proceed with modeling.")

# =============================================================================
# RESET BUTTON WITH QUERY PARAM CLEARING
# =============================================================================
st.divider()
col1, col2 = st.columns(2)

with col1:
    if st.button("Reset All Settings to Defaults", type="secondary"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Clear query parameters
        st.query_params.clear()
        
        # Clear all caches
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Rerun to apply defaults
        st.rerun()