from datasource import *
import streamlit as st
import matplotlib.pyplot as plt
st.set_page_config(page_title="Class Imbalance", page_icon="*")

df = load_and_clean()

st.title("Class Imbalance by Smoking Status")
fig, ax = plt.subplots(figsize=(10,3))
ax.pie(df['Heart Disease Status'].value_counts(), labels=df['Heart Disease Status'].value_counts().index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

st.markdown('''
            Since the analysis is based on the smoking status of a person, data has been oversampled using SMOTE method.
            
            The results of the application can be seen below
            ''')

st.dataframe(load_encoded_dropped())
df_new = load_oversampled(load_encoded_dropped())


fig, ax = plt.subplots(figsize=(10,3))
ax.pie(df_new['Heart Disease Status_encoded'].value_counts(), labels=df_new['Heart Disease Status_encoded'].value_counts().index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

st.title("Heart Disease Class Distribution Before and After SMOTE")

# --- Bar plots ---
st.subheader("Bar Plots")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
