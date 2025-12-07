import pandas as pd
import streamlit as st
from datasource import *
import matplotlib.pyplot as plt
st.set_page_config(page_title="Third Dataset & Advanced Preprocessing")

st.title("Third Dataset & Advanced Preprocessing")
st.markdown("This page provides comprehensive analysis of the country indicators dataset and identifies discrepancies with the heart disease data.")

# Load the indicators dataset
df_indicators = load_indicators()

st.header("Initial Data Analysis: Country Indicators Dataset")

# Display first rows
st.subheader("Overview of Indicators Data (First 10 rows)")
st.dataframe(df_indicators.head(10))

# Key metrics in columns
st.subheader("Key Metrics Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_countries = len(df_indicators)
    st.metric("Total Countries", total_countries)
    
with col2:
    total_features = len(df_indicators.columns)
    st.metric("Total Features", total_features)
    
with col3:
    missing_total = df_indicators.isnull().sum().sum()
    st.metric("Total Missing Values", missing_total)
    
with col4:
    complete_rows = df_indicators.dropna().shape[0]
    st.metric("Complete Records", complete_rows)

# Basic statistical summary
st.subheader("Statistical Summary of Numerical Features")
numerical_cols = df_indicators.columns
if len(numerical_cols) > 0:
    stats_summary = df_indicators[numerical_cols].describe().T
    st.dataframe(stats_summary, use_container_width=True)
else:
    st.info("No numerical columns found in the dataset")


st.subheader("Key Health & Economic Indicators")

# Select some key indicators for quick visualization
key_indicators = ['Life Expectancy', 'GDP', 'Physicians per Thousand', 
                  'Unemployment Rate', 'Urban Population', 'Infant Mortality']

available_indicators = [col for col in key_indicators if col in df_indicators.columns]

if available_indicators:
    # Display distributions for key indicators
    for indicator in available_indicators[:4]:  # Show first 4 to avoid clutter
        if pd.api.types.is_numeric_dtype(df_indicators[indicator]):
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(10, 3))
                df_indicators[indicator].hist(bins=20, ax=ax, edgecolor='black')
                ax.set_title(f'Distribution of {indicator}')
                ax.set_xlabel(indicator)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
            with col2:
                st.write(f"**{indicator} Stats**")
                st.write(f"Mean: {df_indicators[indicator].mean():.2f}")
                st.write(f"Median: {df_indicators[indicator].median():.2f}")
                st.write(f"Std Dev: {df_indicators[indicator].std():.2f}")
                st.write(f"Min: {df_indicators[indicator].min():.2f}")
                st.write(f"Max: {df_indicators[indicator].max():.2f}")

# =============================================================================
# COUNTRY DISCREPANCY ANALYSIS
# =============================================================================

st.header("Country Name Discrepancy Analysis")

# Load heart disease dataset for comparison
df_country = load_country()

# Find country discrepancies
countries_in_heart = set(df_country['country'].str.strip().str.title())
countries_in_indicators = set(df_indicators['Country'].str.strip().str.title())

in_heart_not_indicators = countries_in_heart - countries_in_indicators
in_indicators_not_heart = countries_in_indicators - countries_in_heart

# Display discrepancies
st.subheader("Country Name Discrepancies")

col1, col2 = st.columns(2)

with col1:
    st.write("**Countries in Heart Data (Missing from Indicators)**")
    if in_heart_not_indicators:
        st.write(f"**{len(in_heart_not_indicators)} countries missing:**")
        for country in sorted(in_heart_not_indicators):
            st.write(f"- {country}")

with col2:
    st.write("**Countries in Indicators (Missing from Heart Data)**")
    if in_indicators_not_heart:
        st.write(f"**{len(in_indicators_not_heart)} countries missing:**")
        for country in sorted(in_indicators_not_heart):
            st.write(f"- {country}")

# Summary statistics
st.subheader("Merging Possibility for two datasets")
merge_col1, merge_col2, merge_col3, merge_col4 = st.columns(4)

with merge_col1:
    st.metric("Heart Disease Countries", len(countries_in_heart))
with merge_col2:
    st.metric("Indicator Countries", len(countries_in_indicators))
with merge_col3:
    overlap = len(countries_in_heart & countries_in_indicators)
    st.metric("Overlapping Countries", overlap)
with merge_col4:
    merge_rate = (overlap / len(countries_in_heart)) * 100
    st.metric("Potential Merge Rate", f"{merge_rate:.1f}%")

st.warning("As can be seen, not all countries are named the same in two datasets. This is natural because there is no single agreement on how to name every country. We will try to identify couhtry names and manually map them below.")  
# Add this section after displaying the discrepancies
st.header("Country Name Mapping")

st.subheader("Name Mappings that are possible to resolve")
country_mapping = resolve_country_names()
if country_mapping:
    st.write("The following country name mappings will be applied:")
    mapping_df = pd.DataFrame(list(country_mapping.items()), columns=['Heart Data Name', 'Indicators Data Name'])
    st.dataframe(mapping_df)
    
    # Show how many will be resolved
    resolved_count = len(country_mapping)
else:
    st.info("No automatic country name mappings defined.")


st.header("Datasets After Merging")

merged_df = merge_country_datasets()
st.dataframe(merged_df.head(10))

# Show merge statistics
st.subheader("Merge Statistics")
col1, col2 = st.columns(2)
with col1:
    st.metric("Original Heart Countries", len(df_country))
with col2:
    st.metric("Successfully Merged", len(merged_df))
    