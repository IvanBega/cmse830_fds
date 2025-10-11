import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer

def load_country():
    df = pd.read_csv("country.csv")
    df = df.rename(columns={
    'HeartDiseaseRatesAgeStandardizedRate_2022': 'std_rate_2022',
    'HeartDiseaseRatesASRDALYsPer100k_2021': 'dalys_2021',
    'HeartDiseaseRatesASRDeathsPer100k_2021': 'deaths_2021',
    'HeartDiseaseRatesASRPrevalencePer100k_2021': 'prevalence_2021'
})
    return df
def load_and_clean():
    df = pd.read_csv("heart_disease.csv")
    columns_to_keep = ["Age", "Gender", "Blood Pressure", "Cholesterol Level", "Smoking", "Diabetes", "BMI", "High Blood Pressure", "Stress Level", "Heart Disease Status"]
    df = df[columns_to_keep]
    return df

def load_encoded():
    # Source for some code - HW 2
    df = load_and_clean()
    le_gender = LabelEncoder()
    le_smoker = LabelEncoder()
    df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])
    df['Smoking_encoded'] = le_smoker.fit_transform(df['Smoking'])
    
    stress = [['Low', 'Medium', 'High']]
    # ChatGPT 5 was used on October 10 for line below, to help with handle_unknown
    oe = OrdinalEncoder(categories=stress, handle_unknown='use_encoded_value', unknown_value=-1)
    df['Stress Level_encoded'] = oe.fit_transform(df[['Stress Level']])
    
    return df

def load_imputed():
    df = load_encoded()
    imputer = KNNImputer(n_neighbors=5)
    
    df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['number'])), columns=df.select_dtypes(include=['number']).columns)
    return df_imputed