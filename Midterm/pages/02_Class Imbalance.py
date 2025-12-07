from datasource import *
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
st.set_page_config(page_title="Class Imbalance", page_icon="*")

df = load_and_clean()
df_new = load_oversampled(load_encoded_dropped())
st.title("Class Imbalance by Heart Disease Status")

st.markdown('''When analyzing heart health, we must ensure that we have enough data. If the number of people who
            have a Heart Disease is not represented well in the dataset, like in our example,
            we might have a bias towards people with no Heart Disease. To resolve this issue, an
            oversampling will be performed using synthetic data.''')
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(10,3))
    counts = df['Heart Disease Status'].value_counts()
    label_map = {
        'Yes': 'Heart Disease Present',
        'No': 'No Heart Disease'
    }
    labels = [label_map.get(label, label) for label in counts.index]
    ax.pie(df['Heart Disease Status'].value_counts(), labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(10,3))
    ax.pie(df_new['Heart Disease Status_encoded'].value_counts(), labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

st.markdown('''
            Two pie charts above demonstrate the distribution of people with Heart Disease in the
            original dataset, and an oversampled dataset. Oversampling was performed with SMOTE method by imblearn library
            ''')

#st.dataframe(load_encoded_dropped())



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