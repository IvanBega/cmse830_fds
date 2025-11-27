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
indicators_file = script_dir / 'world-data-2023.csv'
le_gender = LabelEncoder()
le_smoker = LabelEncoder()
le_diabetes = LabelEncoder()
le_high_blood_pressure = LabelEncoder()
le_heart_disease = LabelEncoder()
# ChatGPT 5 was used on October 10 for line below, to help with handle_unknown
oe = OrdinalEncoder(categories=[['Low', 'Medium', 'High']], handle_unknown='use_encoded_value', unknown_value=-1)
    
def load_indicators():
    """Load the third indicators dataset with only the 10 most relevant features"""
    df = pd.read_csv(indicators_file)
    
    # Define the 10 most important features for heart disease analysis
    important_features = [
        'Country', 
        'Life expectancy', 
        'GDP', 
        'Physicians per thousand',
        'Unemployment rate', 
        'Out of pocket health expenditure', 
        'Urban_population', 
        'Infant mortality', 
        'CPI', 
        'Fertility rate', 
        'Agricultural land (%)'
    ]
    
    # Keep only features that exist in the dataset
    available_features = [col for col in important_features if col in df.columns]
    df = df[available_features]
    
    # Basic cleaning
    df = df.drop_duplicates()
    df['Country'] = df['Country'].str.strip().str.title()
    
    # Clean numeric columns - remove special characters and convert to numeric
    # Clean GDP - remove dollar signs, commas and convert to float
    if 'GDP' in df.columns:
        df['GDP'] = df['GDP'].astype(str).str.replace('$', '', regex=False)
        df['GDP'] = df['GDP'].str.replace(',', '', regex=False)
        df['GDP'] = pd.to_numeric(df['GDP'], errors='coerce')
    
    # Clean other numeric columns that might have percentage signs or commas
    numeric_columns_to_clean = [
        'Physicians per thousand',
        'Unemployment rate', 
        'Out of pocket health expenditure', 
        'Urban_population', 
        'Infant mortality', 
        'CPI', 
        'Fertility rate', 
        'Agricultural land (%)'
    ]
    
    for col in numeric_columns_to_clean:
        if col in df.columns:
            # Remove percentage signs and commas
            df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure Life expectancy is numeric (in case it has any formatting)
    if 'Life expectancy' in df.columns:
        df['Life expectancy'] = pd.to_numeric(df['Life expectancy'], errors='coerce')
    
    return df


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


# Add to datasource.py - country name resolution function
def resolve_country_names():
    """Manual resolution of country name discrepancies between datasets"""
    country_mapping = {
        # Countries in Heart Data -> Indicators Data
        'Dr Congo': 'Democratic Republic Of The Congo',
        'Timor-Leste': 'East Timor',
        'Micronesia': 'Federated States Of Micronesia',
        'Ireland': 'Republic Of Ireland',
        'Palestine': 'Palestinian National Authority',
        'Bahamas': 'The Bahamas',
        'Gambia': 'The Gambia',
        
        # Countries we might want to add mappings for if needed
        # 'American Samoa': None,  # Not in indicators data
        # 'Bermuda': None,        # Not in indicators data
        # 'Cook Islands': None,   # Not in indicators data
    }
    return country_mapping

def merge_country_datasets():
    """Merge heart disease data with indicators data after resolving name discrepancies"""
    df_country = load_country()
    df_indicators = load_indicators()
    
    # Apply country name standardization
    country_mapping = resolve_country_names()
    
    # Create clean country names for merging
    df_country['country_clean'] = df_country['country'].str.strip().str.title()
    df_indicators['Country_clean'] = df_indicators['Country'].str.strip().str.title()
    
    # Apply the mapping to heart disease data
    df_country['country_clean'] = df_country['country_clean'].replace(country_mapping)
    
    # Merge datasets
    merged_df = pd.merge(
        df_country, 
        df_indicators, 
        left_on='country_clean', 
        right_on='Country_clean', 
        how='inner'
    )
    
    # Drop the temporary clean columns
    merged_df = merged_df.drop(['country_clean', 'Country_clean'], axis=1)
    
    return merged_df