import pandas as pd
import numpy as np
import streamlit as st
from datasource import *
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Advanced Feature Engineering", page_icon="⚙️")

st.title("Advanced Feature Engineering & Data Processing")
st.markdown("This page demonstrates sophisticated feature engineering techniques to enhance predictive modeling.")

# Load and prepare data
df_heart = load_encoded_dropped()
df_country = load_country()
df_indicators = load_indicators()
merged_df = merge_country_datasets()
# =============================================================================
# SECTION 1: INDIVIDUAL-LEVEL FEATURE ENGINEERING
# =============================================================================

st.header("1. Individual-Level Clinical Feature Engineering")

st.subheader("Original Clinical Features")
st.dataframe(df_heart.select_dtypes(include=['number']).head(10))

# Define numerical features first
numeric_features = ['Age', 'BMI', 'Cholesterol Level', 'Blood Pressure', 'Stress Level_encoded']
available_numeric = [f for f in numeric_features if f in df_heart.columns]

# Check for missing values first
st.subheader("Missing Values Analysis")
if available_numeric:
    missing_summary = df_heart[available_numeric].isnull().sum()
    st.write("Missing values in numerical features:")
    st.dataframe(missing_summary[missing_summary > 0])
else:
    st.warning("No numeric features available for analysis")

# Polynomial Features
st.subheader("1.1 Polynomial Features & Interactions")

if available_numeric:
    # Handle missing values before polynomial features
    df_heart_clean = df_heart[available_numeric].copy()
    
    # Show original missing values
    original_missing = df_heart_clean.isnull().sum().sum()
    st.write(f"**Total missing values before imputation: {original_missing}**")
    
    # Option 1: Impute missing values with median
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    df_heart_imputed = pd.DataFrame(
        imputer.fit_transform(df_heart_clean),
        columns=available_numeric,
        index=df_heart_clean.index
    )
    
    st.write(f"**Imputed {original_missing} missing values using median**")
    
    # Create polynomial features (degree=2 includes squares and interactions)
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(df_heart_imputed)
    poly_feature_names = poly.get_feature_names_out(available_numeric)
    
    # Create DataFrame with polynomial features
    df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_heart_imputed.index)
    
    # Filter to show only meaningful polynomial features (remove near-constant)
    meaningful_poly = df_poly.loc[:, df_poly.std() > 0.1]
    
    st.write(f"**Generated {len(poly_feature_names)} polynomial features**")
    st.write(f"**Keeping {len(meaningful_poly.columns)} meaningful features (std > 0.1)**")
    
    # Show sample of polynomial features
    st.write("**Sample of Generated Polynomial Features:**")
    st.dataframe(meaningful_poly.head(10))
    
    # Calculate correlation with target
    poly_with_target = meaningful_poly.copy()
    poly_with_target['target'] = df_heart['Heart Disease Status_encoded']
    poly_correlations = poly_with_target.corr()['target'].abs().sort_values(ascending=False)
    
    st.write("**Top Polynomial Features by Correlation with Heart Disease:**")
    st.dataframe(poly_correlations.head(10).round(3))
    
    # Visualization of top polynomial features
    top_poly_features = poly_correlations.index[1:6]  # Exclude target itself
    if len(top_poly_features) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_poly_features[:6]):
            if i < len(axes):
                # Create temporary dataframe for plotting
                plot_df = meaningful_poly[feature].copy()
                plot_df = pd.DataFrame({
                    'feature': plot_df,
                    'heart_disease': df_heart['Heart Disease Status_encoded']
                })
                plot_df = plot_df.dropna()
                
                sns.boxplot(data=plot_df, x='heart_disease', y='feature', ax=axes[i])
                axes[i].set_title(f'{feature[:20]}...' if len(feature) > 20 else feature)
                axes[i].set_xlabel('Heart Disease (0=No, 1=Yes)')
        
        # Remove empty subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show feature descriptions
        st.write("**Polynomial Feature Descriptions:**")
        feature_descriptions = []
        for feature in top_poly_features[:5]:
            if '^2' in feature:
                desc = f"{feature}: Squared term"
            elif ' ' in feature:
                parts = feature.split(' ')
                desc = f"{feature}: Interaction between {parts[0]} and {parts[1]}"
            else:
                desc = f"{feature}: Original feature"
            feature_descriptions.append(desc)
        
        for desc in feature_descriptions:
            st.write(f"- {desc}")
else:
    st.warning("No numeric features available for polynomial expansion")

# Binning & Discretization
st.subheader("1.2 Binning & Discretization")

# Handle missing values for binning features
binning_features = ['Age', 'BMI', 'Blood Pressure']
available_binning = [f for f in binning_features if f in df_heart.columns]

if available_binning:
    # Use imputed data for consistent binning (if we created it above)
    if 'df_heart_imputed' in locals():
        binning_source = df_heart_imputed
    else:
        # If no imputation was done, use original data and handle missing values
        binning_source = df_heart[available_binning].copy()
        # Simple forward fill for binning (or you could use median imputation here too)
        binning_source = binning_source.fillna(binning_source.median())
    
    for feature in available_binning:
        if feature in binning_source.columns:
            if feature == 'Age':
                df_heart[f'{feature}_bin'] = pd.cut(binning_source[feature], 
                                                   bins=[0, 35, 50, 65, 100], 
                                                   labels=['Young', 'Middle', 'Senior', 'Elderly'])
            elif feature == 'BMI':
                df_heart[f'{feature}_category'] = pd.cut(binning_source[feature], 
                                                        bins=[0, 18.5, 25, 30, 100],
                                                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            elif feature == 'Blood Pressure':
                df_heart[f'{feature}_category'] = pd.cut(binning_source[feature],
                                                        bins=[0, 120, 130, 140, 180, 300],
                                                        labels=['Normal', 'Elevated', 'Stage1', 'Stage2', 'Crisis'])
    
    # Show binning results
    binning_cols = [col for col in df_heart.columns if '_bin' in col or '_category' in col]
    if binning_cols:
        st.write("**Created Binned Features:**")
        binning_results = []
        for col in binning_cols:
            binning_results.append({
                'Feature': col,
                'Unique Values': df_heart[col].nunique(),
                'Non-Null Count': df_heart[col].count(),
                'Null Count': df_heart[col].isnull().sum()
            })
        st.dataframe(pd.DataFrame(binning_results))
        
        # Show distribution of binned features
        col1, col2 = st.columns(2)
        with col1:
            if 'Age_bin' in df_heart.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                df_heart['Age_bin'].value_counts().sort_index().plot(kind='bar', ax=ax)
                ax.set_title('Age Bin Distribution')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(fig)
        
        with col2:
            if 'BMI_category' in df_heart.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                df_heart['BMI_category'].value_counts().sort_index().plot(kind='bar', ax=ax)
                ax.set_title('BMI Category Distribution')
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(fig)
        
        # Show heart disease rates by bins
        st.write("**Heart Disease Rates by Binned Categories:**")
        for col in binning_cols[:2]:  # Show first 2 to avoid clutter
            rates = df_heart.groupby(col)['Heart Disease Status_encoded'].mean()
            st.write(f"{col}:")
            st.dataframe(rates.round(3))
else:
    st.warning("No features available for binning")

# =============================================================================
# SECTION 2: COUNTRY-LEVEL FEATURE ENGINEERING
# =============================================================================

st.header("2. Country-Level Feature Engineering")

if len(merged_df) > 0:
    st.subheader("2.1 PCA - Socioeconomic Development Index")
    
    # Select socioeconomic indicators for PCA
    socioecon_indicators = ['GDP', 'Life expectancy', 'Physicians per thousand', 
                           'Urban_population', 'Infant mortality']
    available_indicators = [ind for ind in socioecon_indicators if ind in merged_df.columns]
    
    if len(available_indicators) > 1:
        # Prepare data for PCA
        X_socio = merged_df[available_indicators].dropna()
        
        if len(X_socio) > 0:
            # Standardize the data
            X_standardized = (X_socio - X_socio.mean()) / X_socio.std()
            
            # Perform PCA
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_standardized)
            
            # Create PCA DataFrame
            df_pca = pd.DataFrame(data=principal_components, 
                                columns=['PC1_Development', 'PC2_Development'],
                                index=X_socio.index)
            
            # Add back to merged_df
            merged_df_pca = merged_df.loc[X_socio.index].copy()
            merged_df_pca['PC1_Development'] = df_pca['PC1_Development']
            merged_df_pca['PC2_Development'] = df_pca['PC2_Development']
            
            st.write("**PCA Results - Socioeconomic Development:**")
            st.write(f"Explained variance ratio: {pca.explained_variance_ratio_}")
            st.write(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")
            
            # Show component loadings
            loadings = pd.DataFrame(pca.components_.T, 
                                  columns=['PC1', 'PC2'],
                                  index=available_indicators)
            st.write("**PCA Component Loadings:**")
            st.dataframe(loadings.round(3))
            
            # Visualization of PCA
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], 
                               c=merged_df_pca['std_rate_2022'], cmap='viridis', alpha=0.6)
            ax.set_xlabel('PC1 - Development Index')
            ax.set_ylabel('PC2 - Development Index')
            ax.set_title('PCA: Countries by Socioeconomic Development')
            plt.colorbar(scatter, label='Heart Disease Rate')
            st.pyplot(fig)

    st.subheader("2.2 Healthcare Access Metric")
    
    # Create composite healthcare access score
    healthcare_metrics = ['Physicians per thousand', 'Out of pocket health expenditure']
    available_healthcare = [m for m in healthcare_metrics if m in merged_df.columns]
    
    if len(available_healthcare) > 0:
        # Standardize and combine (inverse for out-of-pocket expenditure - lower is better)
        healthcare_scores = []
        for metric in available_healthcare:
            if metric == 'Out of pocket health expenditure':
                # Lower out-of-pocket spending is better (inverse)
                score = -merged_df[metric]  # Negative because lower is better
            else:
                score = merged_df[metric]
            
            # Standardize
            score_std = (score - score.mean()) / score.std()
            healthcare_scores.append(score_std)
        
        # Combine scores
        merged_df['Healthcare_Access_Score'] = sum(healthcare_scores) / len(healthcare_scores)
        
        st.write("**Healthcare Access Score Distribution:**")
        fig, ax = plt.subplots(figsize=(10, 4))
        merged_df['Healthcare_Access_Score'].hist(bins=20, ax=ax, edgecolor='black')
        ax.set_xlabel('Healthcare Access Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Healthcare Access Scores')
        st.pyplot(fig)
        
        # Show correlation with heart disease
        if 'std_rate_2022' in merged_df.columns:
            healthcare_corr = merged_df[['Healthcare_Access_Score', 'std_rate_2022']].corr().iloc[0,1]
            st.metric("Correlation with Heart Disease Rate", f"{healthcare_corr:.3f}")

# =============================================================================
# SECTION 3: CROSS-LEVEL FEATURE ENGINEERING
# =============================================================================

st.header("3. Cross-Level Feature Engineering")

st.subheader("3.1 Relative Risk Indicators")

# For demonstration, create relative risk indicators using available data
if 'BMI' in df_heart.columns and 'BMI' in merged_df.columns:
    # Calculate country-level averages
    country_bmi_avg = merged_df.groupby('country')['BMI'].mean()
    
    # Create relative BMI (individual BMI / country average)
    # This would require matching individuals to countries - simplified for demo
    st.info("Relative risk indicators require individual-country matching. This would be implemented with proper data linkage.")
    
    # Example calculation structure
    example_data = pd.DataFrame({
        'Metric': ['BMI', 'Cholesterol', 'Blood Pressure'],
        'Individual_Value': [25, 200, 130],
        'Country_Average': [26, 195, 125],
        'Relative_Risk': [25/26, 200/195, 130/125]
    })
    st.write("**Example Relative Risk Calculation:**")
    st.dataframe(example_data.round(3))

# =============================================================================
# SECTION 4: ADVANCED TRANSFORMATIONS
# =============================================================================

st.header("4. Advanced Data Transformations")

st.subheader("4.1 Target Encoding for Categorical Variables")

# Example with stress level (simplified)
if 'Stress Level_encoded' in df_heart.columns:
    # Regularized target encoding
    df_heart['Stress_Level_TargetEnc'] = df_heart.groupby('Stress Level_encoded')['Heart Disease Status_encoded'].transform(
        lambda x: (x.sum() + 10 * 0.5) / (len(x) + 10)  # Regularization with alpha=10
    )
    
    st.write("**Target Encoding Results for Stress Level:**")
    target_encoding_summary = df_heart.groupby('Stress Level_encoded').agg({
        'Heart Disease Status_encoded': ['mean', 'count'],
        'Stress_Level_TargetEnc': 'mean'
    }).round(3)
    st.dataframe(target_encoding_summary)
    
    # Compare with original encoding
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Original encoding
    sns.boxplot(data=df_heart, x='Stress Level_encoded', y='Heart Disease Status_encoded', ax=ax1)
    ax1.set_title('Original Encoding')
    
    # Target encoding
    sns.boxplot(data=df_heart, x='Stress Level_encoded', y='Stress_Level_TargetEnc', ax=ax2)
    ax2.set_title('Target Encoding')
    
    plt.tight_layout()
    st.pyplot(fig)

# =============================================================================
# FEATURE ENGINEERING SUMMARY
# =============================================================================

st.header("Feature Engineering Summary")

st.subheader("Engineered Features Overview")
engineered_features_summary = pd.DataFrame({
    'Feature Type': ['Polynomial Features', 'Binned Features', 'PCA Components', 
                    'Composite Scores', 'Target Encoded'],
    'Count': [len(meaningful_poly.columns) if 'meaningful_poly' in locals() else 0,
             3,  # Age, BMI, BP bins
             2 if 'merged_df_pca' in locals() else 0,
             1 if 'Healthcare_Access_Score' in merged_df.columns else 0,
             1],
    'Description': ['Non-linear transformations and interactions',
                   'Clinical categorization of continuous variables',
                   'Socioeconomic development indices',
                   'Combined healthcare access metric',
                   'Regularized mean encoding for categories']
})

st.dataframe(engineered_features_summary)

st.subheader("Next Steps for Modeling")
st.markdown("""
These engineered features will be used in the Machine Learning Model Development page to:
1. **Compare feature importance** between original and engineered features
2. **Evaluate predictive performance** with and without feature engineering
3. **Analyze model interpretability** with different feature types
4. **Select optimal feature set** for final model deployment
""")