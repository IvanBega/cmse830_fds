import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datasource import *
import plotly.express as px

df = load_and_clean()
df_country = load_country()
print(df.head(50))
st.set_page_config(page_title="Heart Disease Analysis", layout="wide")
st.title("Heart Disease Analysis")
st.markdown('''
            The goal of this app is to analyze different factors that may impact your chances of having a heart-related disease.
            
            
            With ove a dozen of features in Heart Diasease dataset, accompanied by the geographical data of heart disease by country,
            the most important information will be considered in determining whether you or someone you know has a risk of developing a heart disease.
            ''')


st.subheader("Overview of the Heart Disease data, first 10 rows")

st.dataframe(df.head(10))
# https://docs.streamlit.io/develop/api-reference/layout/st.columns

col1, col2, col3, col4 = st.columns(4)

with col1:
    count = len(df)
    st.metric("Total data points", count)
    
with col2:
    disease_count = (df['Heart Disease Status'] == 'Yes').sum()
    percentage = (disease_count / count) * 100
    st.metric("Percentage of People with Heart Disease", f"{percentage}%")
    
with col3:
    st.metric("Average age", f"{df['Age'].mean()}")

with col4:
    st.metric("Average BMI", f"{df['BMI'].mean()    }")
    

# https://docs.streamlit.io/develop/api-reference/text/st.markdown
st.markdown("<h3 style='text-align: center;'>Age distribution of dataset participants</h3>", unsafe_allow_html=True)
st.markdown('''Age plays an important factor in evaluating the health of the heart.
            When working with a dataset, it is important to make sure that every population group
            is well-represented. In our case, we have a pretty uniform distribution of data
            by age.''')
col1, col2 = st.columns(2)

with col1:
    # Age distribution by every year chart
    age_counts = df['Age'].value_counts().sort_index()
    st.bar_chart(age_counts)

with col2:
    # Age distribution by bins of 5 chart
    max_age = int(df['Age'].max()) + 5
    age_bins = list(range(15, max_age, 5))
    age_labels = [f"{i}-{i+4}" for i in range(15, max_age-5, 5)]
    df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    age_distribution = df['Age Group'].value_counts().sort_index()
    st.bar_chart(age_distribution)

st.markdown('''
            Basic statistical summaries of the dataset. The information from max and min values can
            be used to determine possible outliers. By looking at them, it appears that all values are within
            acceptable range and do not contain outliers or errors. For examples, sleep hours are in
            the range between 4 and 10, Body Mass Index (BMI) is between 18 and 40, Cholesterol Level
            is ranging from 150 to 300 and so on.
            ''')

st.subheader("Encoding")

st.write("In order to process the data correctly, some encoding had to be performed on categorical (yes/no or 0/1) variables such as Gender, Diabetes, Smoking. For ordinal variable Stress Level, which can be either Low, Medium, or High, ordinal encoder was used. It is important because Stress levels have an order and can be compared between each other, for example, Medium stress is higher that Low stress")
st.dataframe(load_encoded_dropped().head(10))

health_stats = df.describe().T
st.dataframe(health_stats, use_container_width=True)


country_data = load_country()

st.title("Heart Disease Distrubution ")
st.write("This dataset enables us to see the distribution of heart disease around the world.")
st.write("It is important because this features are influenced by sociel, economic, political factors.")
# https://plotly.com/python/choropleth-maps/
# https://plotly.github.io/plotly.py-docs/generated/plotly.express.choropleth.html
countries = px.choropleth(
    country_data, locations="country",
    locationmode="country names",
    color="std_rate_2022",
    hover_name="country",
    color_continuous_scale=px.colors.sequential.Plasma,
    title="Global Distribution of Heart Disease"
)

st.plotly_chart(countries)

st.subheader("Overview of the Heart Disease by Country, first 10 rows")

st.markdown('''
            std_rate_2022 - Heart Disease Age Standardized Rate(2022)
            
            dalys_2021 - Disability-Adjusted Life Years(2021) = Years of Life Lost (YLL) + Years Lived with Disability (YLD)
            
            mortality_2021 - Standardized Death Per 100,000 (2021)
            
            prevalence_2021 - People living with heart disease''')

st.dataframe(df_country.head(10))

st.markdown('''
            Basic statistical summaries of the dataset
            ''')
country_stats = df_country.describe().T
st.dataframe(country_stats, use_container_width=True)