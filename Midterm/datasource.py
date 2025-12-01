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


# Add these functions to datasource.py

def load_engineered_features():
    """Load dataset with engineered features for modeling"""
    df = load_encoded_dropped()
    
    # Handle missing values for numerical features
    numeric_features = ['Age', 'BMI', 'Cholesterol Level', 'Blood Pressure', 'Stress Level_encoded']
    available_numeric = [f for f in numeric_features if f in df.columns]
    
    if available_numeric:
        # Impute missing values with median
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        df_imputed = pd.DataFrame(
            imputer.fit_transform(df[available_numeric]),
            columns=available_numeric,
            index=df.index
        )
        
        # Add imputed features back to dataframe
        for col in available_numeric:
            df[col] = df_imputed[col]
    
    return df

def create_polynomial_features(df=None):
    """Create polynomial features from numerical clinical data"""
    if df is None:
        df = load_engineered_features()
    
    numeric_features = ['Age', 'BMI', 'Cholesterol Level', 'Blood Pressure', 'Stress Level_encoded']
    available_numeric = [f for f in numeric_features if f in df.columns]
    
    if available_numeric:
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(df[available_numeric])
        poly_feature_names = poly.get_feature_names_out(available_numeric)
        
        # Create DataFrame with polynomial features
        df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
        
        # Keep only meaningful features (std > 0.1)
        meaningful_poly = df_poly.loc[:, df_poly.std() > 0.1]
        
        # Combine with original data
        df_combined = pd.concat([df, meaningful_poly], axis=1)
        return df_combined
    
    return df

def create_binned_features(df=None):
    """Create binned/categorical features from continuous variables"""
    if df is None:
        df = load_engineered_features()
    
    # Age binning
    if 'Age' in df.columns:
        df['Age_bin'] = pd.cut(df['Age'], bins=[0, 35, 50, 65, 100], 
                              labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    # BMI categorization
    if 'BMI' in df.columns:
        df['BMI_category'] = pd.cut(df['BMI'], 
                                   bins=[0, 18.5, 25, 30, 100],
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Blood Pressure staging
    if 'Blood Pressure' in df.columns:
        df['BP_category'] = pd.cut(df['Blood Pressure'],
                                  bins=[0, 120, 130, 140, 180, 300],
                                  labels=['Normal', 'Elevated', 'Stage1', 'Stage2', 'Crisis'])
    
    return df

def create_target_encoded_features(df=None):
    """Create target encoded features for categorical variables"""
    if df is None:
        df = load_engineered_features()
    
    # Target encoding for stress level (regularized)
    if 'Stress Level_encoded' in df.columns:
        # Regularized target encoding
        df['Stress_Level_TargetEnc'] = df.groupby('Stress Level_encoded')['Heart Disease Status_encoded'].transform(
            lambda x: (x.sum() + 10 * 0.5) / (len(x) + 10)  # Regularization with alpha=10
        )
    
    return df

def create_country_pca_features():
    """Create PCA features from country socioeconomic indicators"""
    merged_df = merge_country_datasets()
    
    if len(merged_df) > 0:
        # Select socioeconomic indicators for PCA
        socioecon_indicators = ['GDP', 'Life expectancy', 'Physicians per thousand', 
                               'Urban_population', 'Infant mortality']
        available_indicators = [ind for ind in socioecon_indicators if ind in merged_df.columns]
        
        if len(available_indicators) > 1:
            # Prepare data for PCA
            X_socio = merged_df[available_indicators].dropna()
            
            if len(X_socio) > 0:
                # Standardize the data
                X_standardized = (X_socio - X_socio.mean()) / X_socio.std()
                
                # Perform PCA
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(X_standardized)
                
                # Create PCA DataFrame
                df_pca = pd.DataFrame(data=principal_components, 
                                    columns=['PC1_Development', 'PC2_Development'],
                                    index=X_socio.index)
                
                # Add back to merged_df
                merged_df_pca = merged_df.loc[X_socio.index].copy()
                merged_df_pca['PC1_Development'] = df_pca['PC1_Development']
                merged_df_pca['PC2_Development'] = df_pca['PC2_Development']
                
                return merged_df_pca
    
    return merged_df

def create_healthcare_access_score():
    """Create composite healthcare access score from country indicators"""
    merged_df = merge_country_datasets()
    
    if len(merged_df) > 0:
        # Create composite healthcare access score
        healthcare_metrics = ['Physicians per thousand', 'Out of pocket health expenditure']
        available_healthcare = [m for m in healthcare_metrics if m in merged_df.columns]
        
        if len(available_healthcare) > 0:
            # Standardize and combine (inverse for out-of-pocket expenditure - lower is better)
            healthcare_scores = []
            for metric in available_healthcare:
                if metric == 'Out of pocket health expenditure':
                    # Lower out-of-pocket spending is better (inverse)
                    score = -merged_df[metric]  # Negative because lower is better
                else:
                    score = merged_df[metric]
                
                # Standardize
                score_std = (score - score.mean()) / score.std()
                healthcare_scores.append(score_std)
            
            # Combine scores
            merged_df['Healthcare_Access_Score'] = sum(healthcare_scores) / len(healthcare_scores)
    
    return merged_df

def get_all_engineered_features():
    """Get complete dataset with all engineered features for modeling"""
    # Start with basic engineered features
    df = load_engineered_features()
    df = create_polynomial_features(df)
    df = create_binned_features(df)
    df = create_target_encoded_features(df)
    
    # Get country-level engineered features
    country_pca = create_country_pca_features()
    country_healthcare = create_healthcare_access_score()
    
    # Note: In a real scenario, you would merge these with individual data
    # based on country matching. For now, we return them separately.
    
    return {
        'individual_features': df,
        'country_pca_features': country_pca,
        'country_healthcare_features': country_healthcare
    }

def get_modeling_dataset():
    """Get final dataset ready for machine learning modeling"""
    # Get individual features with all engineering
    df_model = load_engineered_features()
    df_model = create_polynomial_features(df_model)
    df_model = create_binned_features(df_model)
    df_model = create_target_encoded_features(df_model)
    
    # For demonstration, we'll use only individual features
    # In a complete implementation, you would merge with country features
    
    return df_model

# Add these functions to datasource.py

def get_data_with_bins():
    """Get heart disease data with binned features"""
    df = load_and_clean()
    
    # Age binning
    if 'Age' in df.columns:
        df['Age_bin'] = pd.cut(df['Age'], 
                              bins=[0, 35, 50, 65, 100], 
                              labels=['Young', 'Middle', 'Senior', 'Elderly'])
    
    # BMI categorization
    if 'BMI' in df.columns:
        df['BMI_category'] = pd.cut(df['BMI'], 
                                   bins=[0, 18.5, 25, 30, 100],
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    
    # Blood Pressure staging
    if 'Blood Pressure' in df.columns:
        df['BP_category'] = pd.cut(df['Blood Pressure'],
                                  bins=[0, 120, 130, 140, 180, 300],
                                  labels=['Normal', 'Elevated', 'Stage1', 'Stage2', 'Crisis'])
    
    return df

def get_pca_features():
    """Get PCA features from country socioeconomic indicators"""
    df = load_indicators()
    
    # Select socioeconomic indicators for PCA
    socioecon_indicators = ['GDP', 'Life expectancy', 'Physicians per thousand', 
                           'Urban_population', 'Infant mortality']
    available_indicators = [ind for ind in socioecon_indicators if ind in df.columns]
    
    if len(available_indicators) > 1:
        # Prepare data
        X = df[available_indicators].dropna()
        
        if len(X) > 0:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA
            pca = PCA(n_components=min(3, len(available_indicators)))
            principal_components = pca.fit_transform(X_scaled)
            
            # Create PCA DataFrame
            df_pca = pd.DataFrame(data=principal_components, 
                                columns=[f'PC{i+1}' for i in range(principal_components.shape[1])],
                                index=X.index)
            
            # Add back country and other columns
            result_df = df.loc[X.index].copy()
            for col in df_pca.columns:
                result_df[col] = df_pca[col]
            
            # Add PCA metadata
            result_df['PCA_variance_explained'] = sum(pca.explained_variance_ratio_)
            result_df['PCA_n_components'] = pca.n_components_
            
            return result_df, pca
    
    # Return empty DataFrame if PCA fails
    return pd.DataFrame(), None

def get_combined_features():
    """Get combined dataset with binned and PCA features ready for modeling"""
    # Get data with bins
    df_bins = get_data_with_bins()
    
    # Get PCA features
    df_pca, pca_model = get_pca_features()
    
    # Note: In a real implementation, you would merge these datasets
    # based on country matching. For demonstration, we'll return them separately.
    
    return {
        'binned_features': df_bins,
        'pca_features': df_pca,
        'pca_model': pca_model
    }

def get_model_ready_data():
    """Get final dataset ready for machine learning modeling"""
    # Get encoded and cleaned data for modeling
    df_model = load_encoded_dropped()
    
    # Add binned features
    df_bins = get_data_with_bins()
    
    # Merge binned features with model data (using index)
    bin_cols = [col for col in df_bins.columns if '_bin' in col or '_category' in col]
    if bin_cols:
        for col in bin_cols:
            if col in df_bins.columns:
                df_model[col] = df_bins[col]
    
    # Note: PCA features would require country-individual matching
    # For now, we return individual-level features with bins
    
    return df_model