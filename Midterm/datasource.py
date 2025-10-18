import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from pathlib import Path

#heart_disease_file = 'heart_disease.csv'
script_dir = Path(__file__).resolve().parent
heart_disease_file = script_dir / 'heart_disease.csv'
country_file = script_dir / 'country.csv'

le_gender = LabelEncoder()
le_smoker = LabelEncoder()
le_diabetes = LabelEncoder()
le_high_blood_pressure = LabelEncoder()
le_heart_disease = LabelEncoder()
# ChatGPT 5 was used on October 10 for line below, to help with handle_unknown
oe = OrdinalEncoder(categories=[['Low', 'Medium', 'High']], handle_unknown='use_encoded_value', unknown_value=-1)
    
def load_country():
    df = pd.read_csv(country_file)
    df = df.rename(columns={
    'HeartDiseaseRatesAgeStandardizedRate_2022': 'std_rate_2022',
    'HeartDiseaseRatesASRDALYsPer100k_2021': 'dalys_2021',
    'HeartDiseaseRatesASRDeathsPer100k_2021': 'deaths_2021',
    'HeartDiseaseRatesASRPrevalencePer100k_2021': 'prevalence_2021'
})
    
    # Performing imputation on the spot
    
    countries_for_avg = ['Guinea', 'Liberia', 'Sierra Leone', 'Burkina Faso', 'Mali']
    avg_std_rate = df[df['country'].isin(countries_for_avg)]['std_rate_2022'].mean()
    df.loc[df['country'] == 'Ivory Coast', 'std_rate_2022'] = avg_std_rate
    
    return df
def load_and_clean():
    df = pd.read_csv(heart_disease_file)
    columns_to_keep = ["Age", "Gender", "Blood Pressure", "Cholesterol Level", "Smoking", "Diabetes", "BMI", "High Blood Pressure", "Stress Level", "Heart Disease Status",
                       "Triglyceride Level", "Sleep Hours"]
    df = df[columns_to_keep]
    return df

def load_encoded():
    # Source for some code - HW 2
    df = load_and_clean()
    # Encoding for finary variables
    df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])
    df['Smoking_encoded'] = le_smoker.fit_transform(df['Smoking'])
    df['Diabetes_encoded'] = le_diabetes.fit_transform(df['Diabetes'])
    df['High Blood Pressure_encoded'] = le_high_blood_pressure.fit_transform(df['High Blood Pressure'])
    df['Heart Disease Status_encoded'] = le_heart_disease.fit_transform(df['Heart Disease Status'])
    #Encoding for ordinal variable
    df['Stress Level_encoded'] = oe.fit_transform(df[['Stress Level']])
    
    return df

def decode_df(df):
    # Decoding back using same encoders
    df['Gender'] = le_gender.inverse_transform(df['Gender_encoded'])
    df['Smoking'] = le_smoker.inverse_transform(df['Smoking_encoded'])
    df['Diabetes'] = le_diabetes.inverse_transform(df['Diabetes_encoded'])
    df['High Blood Pressure'] = le_high_blood_pressure.inverse_transform(df['High Blood Pressure_encoded'])
    df['Heart Disease Status'] = le_heart_disease.inverse_transform(df['Heart Disease Status_encoded'])
    df['Stress Level'] = oe.inverse_transform(df[['Stress Level_encoded']])[:, 0]
    return df
def load_encoded_dropped():
    #Returns only encoded features, with corresponding original features dropped
    df = load_encoded()
    df = df.drop(['Gender', 'Stress Level', 'Diabetes', 'High Blood Pressure', 'Smoking', 'Heart Disease Status'], axis=1)
    return df


def load_imputed():
    #KNN Imputation for missing values
    df = load_encoded()
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=['number'])), columns=df.select_dtypes(include=['number']).columns)
    return df_imputed

def load_oversampled(df=None):
    # Oversample by smoking status using imblearn library
    if df is None:
        df = load_encoded_dropped()
    
    
    # https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
    df = df.dropna(axis=0)
    y = df['Heart Disease Status_encoded']
    X = df.drop('Heart Disease Status_encoded', axis=1)
    smote = SMOTE()
    X_new, y_new = smote.fit_resample(X, y)
    
    # ChatGPT 4o was used here on 10/13/2025 to help add a column 'is_synthetic'
    
    n_original = len(X)
    n_resampled = len(X_new)
    is_synthetic = [0]*n_original + [1]*(n_resampled-n_original)
    
    df_new = pd.concat([X_new.reset_index(drop=True), y_new.reset_index(drop=True)], axis=1)
    df_new['is_synthetic'] = is_synthetic
    return df_new