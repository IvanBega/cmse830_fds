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

st.set_page_config(page_title="Model Development & Evaluation", page_icon="🤖")

st.title("Model Development & Evaluation")
st.markdown("This page demonstrates machine learning model development, evaluation, and comparison for heart disease prediction.")

# =============================================================================
# SECTION 1: DATA PREPARATION
# =============================================================================

st.header("1. Data Preparation for Modeling")

# Load modeling data
df_model = get_model_ready_data()

st.subheader("1.1 Dataset Overview")
st.write(f"**Dataset Shape:** {df_model.shape}")
st.write(f"**Number of Features:** {len(df_model.columns) - 1}")  # Excluding target
st.write(f"**Target Variable:** Heart Disease Status (encoded: 0=No, 1=Yes)")

# Show target distribution
st.subheader("1.2 Target Variable Distribution")
target_counts = df_model['Heart Disease Status_encoded'].value_counts()
target_percentages = (target_counts / len(df_model) * 100).round(2)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("No Heart Disease", f"{target_counts[0]}", f"{target_percentages[0]}%")
with col2:
    st.metric("Heart Disease", f"{target_counts[1]}", f"{target_percentages[1]}%")
with col3:
    imbalance_ratio = target_counts[0] / target_counts[1] if target_counts[1] > 0 else 0
    st.metric("Class Imbalance Ratio", f"{imbalance_ratio:.2f}:1")

# Feature selection
st.subheader("1.3 Feature Selection")

# Select features for modeling (excluding target and synthetic data flag if present)
exclude_cols = ['Heart Disease Status_encoded', 'is_synthetic']  # Exclude target and synthetic flag
feature_cols = [col for col in df_model.columns if col not in exclude_cols]

# Let user select features
st.write(f"**Available Features ({len(feature_cols)} total):**")
selected_features = st.multiselect(
    "Select features for modeling:",
    options=feature_cols,
    default=feature_cols[:min(10, len(feature_cols))]  # Default to first 10 features
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
        st.warning(f"Dataset contains {X.isnull().sum().sum()} missing values. Using median imputation.")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=selected_features)
    
    # Train-test split
    st.subheader("1.4 Train-Test Split")
    test_size = st.slider("Select test set size:", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random seed:", 0, 100, 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training Samples", len(X_train))
    with col2:
        st.metric("Test Samples", len(X_test))
    with col3:
        st.metric("Training %", f"{(1-test_size)*100:.1f}%")
    with col4:
        st.metric("Test %", f"{test_size*100:.1f}%")
    
    # Feature scaling
    st.subheader("1.5 Feature Scaling")
    scale_features = st.checkbox("Apply feature scaling (StandardScaler)", value=True)
    
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_final = pd.DataFrame(X_train_scaled, columns=selected_features)
        X_test_final = pd.DataFrame(X_test_scaled, columns=selected_features)
        st.success("Features scaled using StandardScaler (zero mean, unit variance)")
    else:
        X_train_final = X_train
        X_test_final = X_test
        st.info("Using original feature scales")
    
    # =============================================================================
    # SECTION 2: MODEL 1 - LOGISTIC REGRESSION
    # =============================================================================
    
    st.header("2. Model 1: Logistic Regression")
    
    st.markdown("""
    **Logistic Regression** is a linear model for binary classification that:
    - Models the probability of class membership
    - Provides interpretable coefficients
    - Works well with linearly separable data
    """)
    
    # Model training
    st.subheader("2.1 Model Training")
    
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
        random_state=random_state
    )
    
    lr_model.fit(X_train_final, y_train)
    
    # Predictions
    y_train_pred = lr_model.predict(X_train_final)
    y_test_pred = lr_model.predict(X_test_final)
    y_test_prob = lr_model.predict_proba(X_test_final)[:, 1]
    
    st.success(f"Logistic Regression model trained successfully!")
    
    # Model coefficients
    st.subheader("2.2 Feature Importance (Coefficients)")
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
    
    st.header("3. Model 2: Decision Tree Classifier")
    
    st.markdown("""
    **Decision Tree** is a non-linear model that:
    - Creates a tree-like structure of decisions
    - Handles non-linear relationships
    - Provides feature importance scores
    - Easy to interpret and visualize
    """)
    
    # Model training
    st.subheader("3.1 Model Training")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_depth = st.slider("Max tree depth:", 1, 20, 5)
    with col2:
        min_samples_split = st.slider("Min samples to split:", 2, 20, 2)
    with col3:
        criterion = st.selectbox("Split criterion:", ['gini', 'entropy'])
    
    # Train model
    dt_model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=random_state
    )
    
    dt_model.fit(X_train_final, y_train)
    
    # Predictions
    y_train_pred_dt = dt_model.predict(X_train_final)
    y_test_pred_dt = dt_model.predict(X_test_final)
    y_test_prob_dt = dt_model.predict_proba(X_test_final)[:, 1]
    
    st.success(f"Decision Tree model trained successfully!")
    
    # Feature importance
    st.subheader("3.2 Feature Importance")
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
    
    st.header("4. Model Evaluation & Comparison")
    
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
    
    st.subheader("4.1 Performance Metrics Comparison")
    st.dataframe(metrics_df.round(4))
    
    # Visual comparison
    st.subheader("4.2 Visual Comparison")
    
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
    st.subheader("4.3 Confusion Matrices")
    
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
    st.subheader("4.4 ROC Curves Comparison")
    
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
    # SECTION 5: CROSS-VALIDATION & MODEL SELECTION
    # =============================================================================
    
    st.header("5. Cross-Validation & Model Selection")
    
    st.subheader("5.1 Cross-Validation Results")
    
    cv_folds = st.slider("Number of CV folds:", 3, 10, 5)
    
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
    
    # Visualize CV results
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(cv_folds)
    width = 0.35
    
    ax.bar(x_pos - width/2, lr_cv_scores, width, label='Logistic Regression', 
           color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, dt_cv_scores, width, label='Decision Tree', 
           color='green', alpha=0.7)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Cross-Validation Accuracy by Fold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(cv_folds)])
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Model selection recommendation
    st.subheader("5.2 Model Selection Recommendation")
    
    # Determine best model
    if roc_auc_lr > roc_auc_dt:
        best_model = "Logistic Regression"
        best_auc = roc_auc_lr
        reason = "Higher ROC-AUC score indicates better overall performance"
    else:
        best_model = "Decision Tree"
        best_auc = roc_auc_dt
        reason = "Higher ROC-AUC score indicates better overall performance"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Recommended Model", best_model)
    with col2:
        st.metric("Best ROC-AUC", f"{best_auc:.4f}")
    
    st.markdown(f"""
    **Recommendation Rationale:**
    - **{best_model}** is recommended based on ROC-AUC score ({best_auc:.4f})
    - **Reason:** {reason}
    - **Considerations for Healthcare Applications:**
      * **Interpretability:** Logistic Regression provides clear coefficients
      * **Non-linearity:** Decision Tree captures complex relationships
      * **Clinical Use:** Consider precision (avoid false positives) or recall (avoid false negatives) based on clinical priorities
    """)
    
    # =============================================================================
    # SECTION 6: MODEL DEPLOYMENT PREPARATION
    # =============================================================================
    
    st.header("6. Model Deployment Preparation")
    
    st.subheader("6.1 Final Model Performance")
    
    # Show final test performance of selected model
    if best_model == "Logistic Regression":
        final_y_pred = y_test_pred
        final_y_prob = y_test_prob
        final_model = lr_model
    else:
        final_y_pred = y_test_pred_dt
        final_y_prob = y_test_prob_dt
        final_model = dt_model
    
    # Detailed classification report
    st.write("**Detailed Classification Report:**")
    class_report = classification_report(y_test, final_y_pred, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    st.dataframe(class_report_df.round(4))
    
    # Feature importance for deployment
    st.subheader("6.2 Key Features for Prediction")
    
    if best_model == "Logistic Regression":
        # Show top coefficients
        top_features = coefficients.sort_values('Absolute Value', ascending=False).head(5)
        st.write("**Top 5 Most Important Features (by coefficient magnitude):**")
        for idx, row in top_features.iterrows():
            direction = "increases" if row['Coefficient'] > 0 else "decreases"
            st.write(f"- **{row['Feature']}**: Coefficient = {row['Coefficient']:.4f} ({direction} heart disease risk)")
    else:
        # Show top feature importances
        top_features = importance_df.head(5)
        st.write("**Top 5 Most Important Features (by importance score):**")
        for idx, row in top_features.iterrows():
            st.write(f"- **{row['Feature']}**: Importance = {row['Importance']:.4f}")
    
    # Deployment considerations
    st.subheader("6.3 Deployment Considerations")
    
    st.markdown("""
    **For Production Deployment:**
    
    1. **Model Persistence:** Save the trained model and scaler using joblib or pickle
    2. **API Development:** Create REST API for real-time predictions
    3. **Monitoring:** Track model drift and performance degradation
    4. **Retraining:** Schedule periodic retraining with new data
    5. **Ethical Considerations:** Ensure model fairness across different demographic groups
    
    **Clinical Integration:**
    - Integrate with electronic health record (EHR) systems
    - Provide confidence scores alongside predictions
    - Include model limitations and assumptions in documentation
    - Ensure compliance with healthcare regulations (HIPAA, GDPR)
    """)
    
else:
    st.error("Please select at least one feature for modeling.")

# =============================================================================
# SECTION 7: NEXT STEPS & CONCLUSION
# =============================================================================

st.header("7. Conclusion & Next Steps")

st.markdown("""
### **Summary of Findings:**

#### **Model Performance:**
- Both Logistic Regression and Decision Tree models were successfully trained
- Performance metrics were calculated and compared
- Cross-validation was performed to ensure model stability
- ROC-AUC analysis provided insights into model discrimination ability

#### **Key Insights:**
1. **Feature Importance:** Identified the most predictive features for heart disease
2. **Model Trade-offs:** Understood the interpretability vs. performance trade-off
3. **Validation:** Demonstrated robust validation techniques to prevent overfitting

### **Next Steps for Improvement:**

#### **1. Advanced Modeling Techniques:**
- Ensemble methods (Random Forest, Gradient Boosting)
- Neural networks for complex pattern recognition
- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV

#### **2. Feature Engineering Enhancement:**
- Incorporate country-level PCA features
- Create interaction terms between clinical and socioeconomic factors
- Implement more sophisticated feature selection techniques

#### **3. Clinical Validation:**
- Validate model on external datasets
- Conduct clinical trials to assess real-world impact
- Collaborate with healthcare professionals for domain-specific insights

#### **4. Deployment Pipeline:**
- Build automated ML pipeline (MLOps)
- Implement model monitoring and alerting systems
- Develop user-friendly interfaces for clinicians

### **Final Recommendation:**
Based on this analysis, proceed with **{best_model}** for initial deployment due to its superior ROC-AUC performance. 
However, continue exploring ensemble methods that might combine the strengths of both linear and non-linear approaches.
""")