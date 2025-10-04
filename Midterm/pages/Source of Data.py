import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datasource import *
from sklearn.impute import KNNImputer

st.set_page_config(page_title="Data Source", page_icon="*")
df = load_and_clean()
st.title("What is the data?")
st.subheader("Dataset 1: Health Indicators")
st.markdown('''
            This dataset contains over 55 thousand records with 21 recorded features.
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
st.subheader("Comparison of Mean and Standard Deviation Before vs After Imputation")
st.dataframe(comparison_df)
st.subheader("Dataset 2: Health Disease by Country")

df_country = load_country()
st.write(df_country.describe())