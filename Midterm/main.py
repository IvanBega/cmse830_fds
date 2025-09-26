import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datasource import load_and_clean

df = load_and_clean()
print(df.head(50))
st.set_page_config(page_title="Heart Disease Analysis", layout="wide")

st.title("Heart Disease Analysis")
st.markdown('''
            For this app, we will be alanyzing different factors that may impact your chances of having a heart-related disease.
            
            
            With over 20 features in out dataset, we will try to extract the most number of useful information,
            so that you can take care of your health and wellness of loved ones.
            ''')


st.subheader("Overview of the Heart data")

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
    
st.subheader("Age Distribution of our Participants!", anchor=False)

# Age distribution by every year chart
age_counts = df['Age'].value_counts().sort_index()
st.bar_chart(age_counts)

# Age distribution by bins of 5 chart
max_age = int(df['Age'].max()) + 5
age_bins = list(range(15, max_age, 5))
age_labels = [f"{i}-{i+4}" for i in range(15, max_age-5, 5)]
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
age_distribution = df['Age Group'].value_counts().sort_index()
st.bar_chart(age_distribution)