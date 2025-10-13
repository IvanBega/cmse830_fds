from datasource import *
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(page_title="Class Imbalance", page_icon="*")

df = load_and_clean()

st.title("Class Imbalance by Smoking Status")
fig, ax = plt.subplots(figsize=(10,3))
ax.pie(df['Heart Disease Status'].value_counts(), labels=df['Heart Disease Status'].value_counts().index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

st.markdown('''
            Since the analysis is based on the smoking status of a person, data has been oversampled using SMOTE method.
            
            The results of the application can be seen below.
            ''')

#st.dataframe(load_encoded_dropped())
df_new = load_oversampled(load_encoded_dropped())


fig, ax = plt.subplots(figsize=(10,3))
ax.pie(df_new['Heart Disease Status_encoded'].value_counts(), labels=df_new['Heart Disease Status_encoded'].value_counts().index, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)


st.title("Distributions Before vs After SMOTe")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(10, 4))

    sns.stripplot(x='Age', y='BMI',
                hue='is_synthetic',
                data=df_new.sample(200),
                jitter=True,
                
                ax=ax1)

    #ax.set_yticks([])
    ax1.set_ylabel('BMI')
    ax1.set_xlabel('Age')
    st.pyplot(fig1)
    
with col2:
    fig, ax = plt.subplots(figsize=(10, 4))

    sns.stripplot(x='Blood Pressure', y='Cholesterol Level',
                hue='is_synthetic',
                data=df_new.sample(200),
                jitter=True,
                ax=ax)

   # ax.set_yticks([])
    ax.set_ylabel('Cholesterol Level')
    ax.set_xlabel('Blood Pressure')
    st.pyplot(fig)