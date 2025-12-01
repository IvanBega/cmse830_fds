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

# =============================================================================
# SECTION 1: DATA PREPARATION
# =============================================================================

# Data sampling option
st.subheader("Use Oversampling and Scaling?")
use_oversampling = st.checkbox("Use SMOTE Oversampling to handle class imbalance", value=True)
scale_features = st.checkbox("Apply StandardScaler", value=True)

# Load modeling data with or without oversampling
df_model = get_model_ready_data(use_oversampling=use_oversampling)

st.subheader("Dataset Stats for Machine Learning Model")
st.write(f"**Shape:** {df_model.shape}")
st.write(f"**Number of Features:** {len(df_model.columns) - 1}")  # Excluding target
st.write(f"**Using Oversampling:** {'Yes' if use_oversampling else 'No'}")

# Show target distribution
st.subheader("Distribution of Heart Disease")
if 'is_synthetic' in df_model.columns and use_oversampling:
    # Show synthetic vs real data distribution
    synthetic_counts = df_model['is_synthetic'].value_counts()
    synthetic_percentages = (synthetic_counts / len(df_model) * 100).round(2)
    
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
    st.metric("Class Balance", f"{imbalance_ratio:.2f}:1" if not use_oversampling else "1:1 (Balanced)")

# Feature selection
st.subheader("Which Features to Use?")

# Select features for modeling
exclude_cols = ['Heart Disease Status_encoded', 'is_synthetic'] if 'is_synthetic' in df_model.columns else ['Heart Disease Status_encoded']
feature_cols = [col for col in df_model.columns if col not in exclude_cols]

# Let user select features
selected_features = st.multiselect(
    "Features",
    options=feature_cols,
    default=[col for col in feature_cols if col not in ['Age_bin', 'BMI_category', 'BP_category']][:10]  # Default to first 10 non-binned features
)

# Prepare feature matrix X and target y
if len(selected_features) > 0:
    X = df_model[selected_features]
    y = df_model['Heart Disease Status_encoded']
    
    # Show feature information
    st.write(f"**Selected {len(selected_features)} features:**")
    feature_info = pd.DataFrame({
        'Feature': selected_features,
        'Type': ['Categorical' if '_encoded' in col or '_bin' in col or '_category' in col else 'Numerical' for col in selected_features],
        'Missing Values': X.isnull().sum().values,
        'Unique Values': [X[col].nunique() for col in selected_features]
    })
    st.dataframe(feature_info)
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=selected_features)
    
    # Train-test split
    st.subheader("Train-Test Split")
    test_size = st.slider("Select test set size:", 0.1, 0.4, 0.2, 0.05)
    
    # ADDED EXPLANATION: Test set size recommendation
    st.write("**Test Set Size Recommendation:**")
    st.write("""
    The default test size of 20% (0.2) is a good balance for most datasets:
    - **Increase test size** (e.g., to 30-40%) if you have a large dataset (>10,000 samples) and want more reliable performance estimates
    - **Decrease test size** (e.g., to 10%) if you have a small dataset (<1,000 samples) to preserve more data for training
    - The 80/20 split is a common standard that provides enough data for both training and reliable evaluation
    """)
    
    random_state = st.number_input("Random seed:", 0, 100, 42)
    
    # For oversampled data, we need to ensure we don't split synthetic and real data together
    if 'is_synthetic' in df_model.columns and use_oversampling:
        # Separate real and synthetic data
        real_data = df_model[df_model['is_synthetic'] == 0]
        synthetic_data = df_model[df_model['is_synthetic'] == 1]
        
        # Split real data
        X_real = real_data[selected_features]
        y_real = real_data['Heart Disease Status_encoded']
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X_real, y_real, test_size=test_size, random_state=random_state, stratify=y_real
        )
        
        # Add synthetic data to training only
        X_train = pd.concat([X_train_real, synthetic_data[selected_features]])
        y_train = pd.concat([y_train_real, synthetic_data['Heart Disease Status_encoded']])
        X_test = X_test_real
        y_test = y_test_real
        
    else:
        # Regular train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Samples", len(X_train))
    with col2:
        st.metric("Test Samples", len(X_test))
    with col3:
        train_real = len(X_train) - (len(synthetic_data) if 'synthetic_data' in locals() else 0)
        st.metric("Real Training Samples", train_real)
    with col4:
        if use_oversampling and 'synthetic_data' in locals():
            st.metric("Synthetic Training Samples", len(synthetic_data))
    
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_final = pd.DataFrame(X_train_scaled, columns=selected_features)
        X_test_final = pd.DataFrame(X_test_scaled, columns=selected_features)

    else:
        X_train_final = X_train
        X_test_final = X_test
    
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
    **Using {'oversampled' if use_oversampling else 'original'} data.
    """)
    
    # Model training
    st.subheader("Model Training Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        C_value = st.number_input("Regularization (C):", 0.01, 10.0, 1.0, 0.1)
    with col2:
        max_iter = st.number_input("Max iterations:", 100, 1000, 200, 50)
    with col3:
        penalty_type = st.selectbox("Penalty:", ['l2', 'l1', 'elasticnet'])
    
    # Train model
    lr_model = LogisticRegression(
        C=C_value,
        max_iter=max_iter,
        penalty=penalty_type,
        solver='saga' if penalty_type == 'elasticnet' else 'liblinear',
        random_state=random_state,
        class_weight='balanced' if not use_oversampling else None  # Use class_weight only if not oversampled
    )
    
    lr_model.fit(X_train_final, y_train)
    
    # Predictions
    y_train_pred = lr_model.predict(X_train_final)
    y_test_pred = lr_model.predict(X_test_final)
    y_test_prob = lr_model.predict_proba(X_test_final)[:, 1]
    
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
    **Using {'oversampled' if use_oversampling else 'original'} data.
    """)
    
    # Model training
    st.subheader("Model Training Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_depth = st.slider("Max tree depth:", 1, 20, 5, key="dt_depth")
    with col2:
        min_samples_split = st.slider("Min samples to split:", 2, 20, 2, key="dt_split")
    with col3:
        criterion = st.selectbox("Split criterion:", ['gini', 'entropy'], key="dt_criterion")
    
    # Train model
    dt_model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=random_state,
        class_weight='balanced' if not use_oversampling else None  # Use class_weight only if not oversampled
    )
    
    dt_model.fit(X_train_final, y_train)
    
    # Predictions
    y_train_pred_dt = dt_model.predict(X_train_final)
    y_test_pred_dt = dt_model.predict(X_test_final)
    y_test_prob_dt = dt_model.predict_proba(X_test_final)[:, 1]
    
    st.success(f"Decision Tree model trained successfully!")
    
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
    
    if use_oversampling:
        st.header("Oversampling Impact Analysis")
        
        st.markdown("""
        ### **Impact of SMOTE Oversampling:**
        
        **Benefits:**
        - ✅ Balanced classes (1:1 ratio)
        - ✅ Better representation of minority class
        - ✅ Reduced model bias toward majority class
        - ✅ Improved recall for heart disease detection
        
        **Considerations:**
        - ⚠️ Synthetic data may not perfectly represent real-world patterns
        - ⚠️ Test set contains only real data for fair evaluation
        - ⚠️ Potential overfitting to synthetic patterns
        
        **Current Implementation:**
        - Synthetic data used **only in training**
        - Test set contains **100% real data**
        - This ensures fair evaluation of model generalization
        """)
        
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
    
    cv_folds = st.slider("Number of CV folds:", 3, 10, 5, key="cv_folds")
    
    # For oversampled data, we need special cross-validation
    if use_oversampling:
        
        
        # Use only real data for cross-validation
        if 'real_data' in locals():
            X_cv = real_data[selected_features]
            y_cv = real_data['Heart Disease Status_encoded']
        else:
            X_cv = X
            y_cv = y
    
    # Perform cross-validation
    lr_cv_scores = cross_val_score(lr_model, X_train_final, y_train, 
                                  cv=cv_folds, scoring='accuracy')
    dt_cv_scores = cross_val_score(dt_model, X_train_final, y_train, 
                                  cv=cv_folds, scoring='accuracy')
    
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