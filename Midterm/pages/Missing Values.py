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

st.image("Picture1.png")
target = ['Liberia', 'Guinea', 'Mali', 'Sierra Leone', 'Burkina Faso']
mask = df_country['country'].isin(target)
df_target = df_country[mask]
df_target = df_target[['country', 'std_rate_2022']]
st.markdown(f'''By performing KNN Imputation, using the data from five closest countires: Liberia, Guinea, Mali, Sierra Leone, Burkina Faso, we impute Ivory Coast with the std_rate_2022 value: {df_target['std_rate_2022'].mean()}''')

st.dataframe(df_target)
