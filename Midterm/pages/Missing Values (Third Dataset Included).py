import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datasource import *
from sklearn.impute import KNNImputer
from pathlib import Path

st.set_page_config(page_title="Data Source", page_icon="*")
df = load_and_clean()
st.title("What is the data?")
st.subheader("Dataset 1: Health Indicators")
st.markdown('''
            This dataset contains over 10 thousand records with 21 recorded features.
            Although most values are recorded, there are some blank values which need
            to be addressed. Below is the heatmap of missing values for this dataset''')

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap='plasma', ax=ax)
ax.set_title('Missing Values Heatmap')
ax.set_xlabel('Columns')
st.pyplot(fig)

st.markdown('''
            Since all features except Alcohol Consumption appear to be missing randomly,
            and the number of such rows is negligible compared to the dataset size, a simple KNN
            imputer will be used.
            ''')

df_original = df.copy()
    
# Show original missing values
st.subheader("Missing Values by Feature")
missing_summary = pd.DataFrame({
    'Missing Count': df.isnull().sum(),
}).round(2)
st.dataframe(missing_summary)

df_encoded = load_encoded()
df_imputed = load_imputed()
stats_encoded = df_encoded.describe().T[['mean', 'std']]
stats_imputed = df_imputed.describe().T[['mean', 'std']]
comparison_df = pd.concat(
    [stats_encoded.add_prefix('Before_'), stats_imputed.add_prefix('After_')],
    axis=1
)
comparison_df = comparison_df[['Before_mean', 'After_mean', 'Before_std', 'After_std']] # to reorder columns
comparison_df = comparison_df.round(3)
st.subheader("Comparison of Mean and Standard Deviation Before vs After Imputation (KNN Imputer, n = 5)")
st.dataframe(comparison_df)
st.subheader("Dataset 2: Health Disease by Country")

df_country = load_country()
country_no_flag = df_country[['country', 'std_rate_2022', 'dalys_2021', 'deaths_2021', 'prevalence_2021']]
total_features = country_no_flag.count()
st.dataframe(total_features)
st.markdown('''
            As we can see from table above, one country is missing standardized death rate - Ivory Coast
            ''')

script_dir = Path(__file__).resolve().parent
picture_file = script_dir / 'Picture1.png'

st.image(picture_file)
target = ['Liberia', 'Guinea', 'Mali', 'Sierra Leone', 'Burkina Faso']
mask = df_country['country'].isin(target)
df_target = df_country[mask]
df_target = df_target[['country', 'std_rate_2022']]
st.markdown(f'''By performing KNN Imputation, using the data from five closest countires: Liberia, Guinea, Mali, Sierra Leone, Burkina Faso, we impute Ivory Coast with the std_rate_2022 value: {df_target['std_rate_2022'].mean()}''')

st.dataframe(df_target)

def load_indicators_imputed():
    """Load indicators dataset with KNN imputation applied"""
    df = load_indicators()
    
    # Separate numerical columns for imputation
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(numerical_cols) > 0 and df[numerical_cols].isnull().sum().sum() > 0:
        # Perform KNN imputation on numerical columns
        imputer = KNNImputer(n_neighbors=5)
        df_imputed_num = pd.DataFrame(
            imputer.fit_transform(df[numerical_cols]),
            columns=numerical_cols,
            index=df.index
        )
        
        # Combine back with categorical columns
        df_imputed = df_imputed_num
        if len(categorical_cols) > 0:
            df_imputed = pd.concat([df_imputed, df[categorical_cols]], axis=1)
        
        return df_imputed
    else:
        return df
    
st.subheader("Dataset 3: Country Indicators")
st.markdown('''
            The country indicators dataset contains socioeconomic and health metrics for various countries.
            Below is the missing values analysis for this dataset.
            ''')

# Load the indicators dataset
df_indicators = load_indicators()

# Display missing values heatmap for indicators dataset
st.subheader("Missing Values Heatmap - Country Indicators")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df_indicators.isnull(), cmap='viridis', ax=ax, cbar_kws={'label': 'Missing Value'})
ax.set_title('Missing Values Heatmap - Country Indicators Dataset')
ax.set_xlabel('Columns')
ax.set_ylabel('Countries')
st.pyplot(fig)

# Missing values summary for indicators dataset
st.subheader("Missing Values Summary - Country Indicators")
missing_indicators = pd.DataFrame({
    'Missing Count': df_indicators.isnull().sum(),
    'Missing %': (df_indicators.isnull().sum() / len(df_indicators) * 100).round(2),
    'Data Type': df_indicators.dtypes
})
st.dataframe(missing_indicators)

# KNN Imputation for indicators dataset
st.subheader("KNN Imputation - Country Indicators")

# Show pre-imputation statistics
st.write("**Before Imputation:**")
st.dataframe(df_indicators.describe().round(3))

# Perform KNN imputation
if df_indicators.isnull().sum().sum() > 0:
    # Separate numerical columns for imputation
    numerical_cols = df_indicators.select_dtypes(include=['number']).columns
    categorical_cols = df_indicators.select_dtypes(include=['object']).columns
    
    # Keep Country column separate
    country_col = df_indicators['Country'] if 'Country' in df_indicators.columns else None
    
    if len(numerical_cols) > 0:
        # Perform KNN imputation on numerical columns
        imputer = KNNImputer(n_neighbors=5)
        df_indicators_imputed_num = pd.DataFrame(
            imputer.fit_transform(df_indicators[numerical_cols]),
            columns=numerical_cols,
            index=df_indicators.index
        )
        
        # Combine back with categorical columns
        df_indicators_imputed = df_indicators_imputed_num
        if len(categorical_cols) > 0:
            df_indicators_imputed = pd.concat([df_indicators_imputed, df_indicators[categorical_cols]], axis=1)
        if country_col is not None:
            df_indicators_imputed['Country'] = country_col.values
        
        st.write(f"KNN Imputation was able to fill {df_indicators.isnull().sum().sum()} missing values.")
        
        # Show post-imputation statistics
        st.write("**After Imputation:**")
        st.dataframe(df_indicators_imputed.describe().round(3))
        
        # Comparison of key statistics before and after
        st.subheader("Comparison Before vs After Imputation - Key Indicators")
        
        # Select key numerical columns for comparison
        key_columns = [col for col in ['Life expectancy', 'GDP', 'Physicians per Thousand', 
                                      'Unemployment rate', 'Infant mortality'] 
                      if col in numerical_cols]
        
        if key_columns:
            comparison_data = []
            for col in key_columns:
                comparison_data.append({
                    'Feature': col,
                    'Before_Mean': df_indicators[col].mean(),
                    'After_Mean': df_indicators_imputed[col].mean(),
                    'Before_Std': df_indicators[col].std(),
                    'After_Std': df_indicators_imputed[col].std(),
                    'Before_Missing': df_indicators[col].isnull().sum(),
                    'After_Missing': df_indicators_imputed[col].isnull().sum()
                })
            
            comparison_df_indicators = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df_indicators.round(3))
            
            # Visual comparison
            st.subheader("Distribution Comparison - Before vs After Imputation")
            
            for col in key_columns[:3]:  # Show first 3 to avoid clutter
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Before imputation
                df_indicators[col].hist(bins=20, ax=ax1, alpha=0.7, color='red', label='Before', edgecolor='black')
                ax1.set_title(f'{col} - Before Imputation\n(Missing: {df_indicators[col].isnull().sum()})')
                ax1.set_xlabel(col)
                ax1.set_ylabel('Frequency')
                
                # After imputation
                df_indicators_imputed[col].hist(bins=20, ax=ax2, alpha=0.7, color='green', label='After', edgecolor='black')
                ax2.set_title(f'{col} - After Imputation\n(Missing: {df_indicators_imputed[col].isnull().sum()})')
                ax2.set_xlabel(col)
                ax2.set_ylabel('Frequency')
                
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.info("No numerical columns found for imputation in the indicators dataset.")
else:
    st.success("No missing values found in the indicators dataset!")