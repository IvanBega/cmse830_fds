import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datasource import load_and_clean

st.set_page_config(page_title="Test page", page_icon="*")

st.title("Page 1 title")
st.write("Page 1 write")

df = load_and_clean()

st.sidebar.header("Age Filter")
min_age = int(df['Age'].min())
max_age = int(df['Age'].max())

age_range = st.sidebar.slider(
    "Age Range:",
    min_value = min_age,
    max_value = max_age,
    value=(min_age, max_age)
)

filtered_df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]


st.markdown("<h3 style='text-align: center;'>Age Distribution</h3>", unsafe_allow_html=True)

max_age_filtered = int(filtered_df['Age'].max()) + 5
age_bins = list(range(15, max_age_filtered, 5))
age_labels = [f"{i}-{i+4}" for i in range(15, max_age_filtered-5, 5)]
filtered_df['Age Group'] = pd.cut(filtered_df['Age'], bins=age_bins, labels=age_labels)
age_distribution = filtered_df['Age Group'].value_counts().sort_index()
st.bar_chart(age_distribution)

st.write(f"Showing {len(filtered_df)} individuals aged {age_range[0]} to {age_range[1]}")