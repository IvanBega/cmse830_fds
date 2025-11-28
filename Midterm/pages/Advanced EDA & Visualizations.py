import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datasource import *
import numpy as np
from scipy import stats

st.set_page_config(page_title="Advanced EDA and Visualization", page_icon="📊")

st.title("Advanced EDA and Visualization")
st.markdown("This page showcases advanced exploratory data analysis and visualization techniques across all datasets.")

# Load all datasets
df_heart = load_and_clean()
df_country = load_country()
df_indicators = load_indicators()
merged_df = merge_country_datasets()

# =============================================================================
# VISUALIZATION 1: HEART DISEASE DATASET - Parallel Coordinates Plot
# =============================================================================

st.header("1. Parallel Coordinates Plot - Heart Disease Risk Factors")

st.markdown("""
The plot below demonstrates correlation between the following features: Age, BMI, Cholesterol, Blood Pressure, Stress Level
Heart Disease risk. As can be seen, it is very difficult to establish any relationship between first four features""")

# Prepare data for parallel coordinates
df_encoded = load_encoded_dropped()
df_parallel = df_encoded[['Age', 'BMI', 'Cholesterol Level', 'Blood Pressure', 
                         'Stress Level_encoded', 'Heart Disease Status_encoded']].copy()

# Normalize numerical columns for better visualization
numerical_cols = ['Age', 'BMI', 'Cholesterol Level', 'Blood Pressure']
for col in numerical_cols:
    df_parallel[col] = (df_parallel[col] - df_parallel[col].min()) / (df_parallel[col].max() - df_parallel[col].min())

# Create parallel coordinates plot
fig_parallel = px.parallel_coordinates(
    df_parallel,
    color="Heart Disease Status_encoded",
    color_continuous_scale=px.colors.diverging.Tealrose,
    title="Parallel Coordinates of Heart Disease Risk Factors",
    labels={
        "Age": "Age (normalized)",
        "BMI": "BMI (normalized)", 
        "Cholesterol Level": "Cholesterol (normalized)",
        "Blood Pressure": "Blood Pressure (normalized)",
        "Stress Level_encoded": "Stress Level",
        "Heart Disease Status_encoded": "Heart Disease Risk"
    }
)

st.plotly_chart(fig_parallel, use_container_width=True)

# Statistical analysis
st.subheader("Statistical Analysis: Top Risk Factor Correlations")
st.write("Below you can see which factors contribute the most to the heart disease.")
correlation_matrix = df_encoded.corr()['Heart Disease Status_encoded'].sort_values(ascending=False)
st.dataframe(correlation_matrix.iloc[1:6])  # Top 5 correlations excluding itself

# =============================================================================
# VISUALIZATION 2: COUNTRY DATASET - Bubble Map
# =============================================================================

st.header("2. Interactive Bubble Map - Global Heart Disease Burden")

st.markdown("""
This interactive bubble map combines two datasets.
It allows us to see heart disease prevalence, mortality, and burden simultaneously across countries.
""")

# Create bubble map
fig_bubble = px.scatter_geo(
    df_country,
    locations="country",
    locationmode="country names",
    size="prevalence_2021",
    color="deaths_2021",
    hover_name="country",
    hover_data={
        "std_rate_2022": True,
        "dalys_2021": True,
        "deaths_2021": True,
        "prevalence_2021": True
    },
    color_continuous_scale="Viridis",
    title="Global Heart Disease Burden: Prevalence (Size) vs Mortality (Color)",
    projection="natural earth"
)

fig_bubble.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='equirectangular'
    ),
    height=600
)

st.plotly_chart(fig_bubble, use_container_width=True)


# =============================================================================
# VISUALIZATION 5: RAINCLOUD PLOT - Development Indicators by Income Groups
# =============================================================================

st.header("3. Raincloud Plot - Country Development Indicators by Income Groups")

st.markdown("""
The Raincloud Plot combines three plots at the same time: box plot, violin plot, and a scatterplot.  
In this example we are attempting to compare some of the most important life-changing indicators by the income group.
In particular, we used Life expectancy, number of physicians per thousand, and an infant mortality rates.

Not surprisingly, it is easy to see that the higher income the country has, the longer life expectancy and number of physicians per 1000 population they have.

Conversely, the infant mortality rate appears to decrease with the increase of income.""")

# Prepare data for raincloud plot
df_raincloud = df_indicators.copy()

# Create income groups based on GDP tertiles
if 'GDP' in df_raincloud.columns:
    df_raincloud['Income_Group'] = pd.qcut(df_raincloud['GDP'], q=3, 
                                         labels=['Low Income', 'Medium Income', 'High Income'])
    
    # Select key development indicators
    indicators = ['Life expectancy', 'Physicians per thousand', 'Infant mortality']
    
    # Filter to indicators that exist in the dataset
    available_indicators = [ind for ind in indicators if ind in df_raincloud.columns]
    
    if len(available_indicators) > 0 and 'Income_Group' in df_raincloud.columns:
        # Create subplots
        fig, axes = plt.subplots(1, len(available_indicators), figsize=(5*len(available_indicators), 6))
        if len(available_indicators) == 1:
            axes = [axes]
        
        for i, indicator in enumerate(available_indicators):
            # Prepare data for this indicator
            plot_data = []
            groups = []
            for group in ['Low Income', 'Medium Income', 'High Income']:
                group_data = df_raincloud[df_raincloud['Income_Group'] == group][indicator].dropna()
                plot_data.append(group_data)
                groups.append(group)
            
            # Create raincloud plot components
            # Violin plot
            parts = axes[i].violinplot(plot_data, showmeans=False, showmedians=False, showextrema=False)
            
            # Color the violins
            for pc in parts['bodies']:
                pc.set_facecolor('lightblue')
                pc.set_alpha(0.7)
            
            # Box plot
            axes[i].boxplot(plot_data, positions=range(1, len(plot_data)+1), 
                           widths=0.3, patch_artist=True,
                           boxprops=dict(facecolor='lightgray', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))
            
            # Scatter points (jittered)
            for j, data in enumerate(plot_data):
                x = np.random.normal(j+1, 0.1, size=len(data))
                axes[i].scatter(x, data, alpha=0.5, color='black', s=20)
            
            axes[i].set_title(f'{indicator}')
            axes[i].set_xlabel('Income Group')
            axes[i].set_ylabel(indicator)
            axes[i].set_xticks(range(1, len(groups)+1))
            axes[i].set_xticklabels(groups, rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistical summary
        st.subheader("Quantitative Summaries by GDP Groups of the plot above")
        summary_data = []
        for indicator in available_indicators:
            for group in ['Low Income', 'Medium Income', 'High Income']:
                group_data = df_raincloud[df_raincloud['Income_Group'] == group][indicator]
                summary_data.append({
                    'Indicator': indicator,
                    'Income Group': group,
                    'Count': len(group_data.dropna()),
                    'Mean': group_data.mean(),
                    'Median': group_data.median(),
                    'Std Dev': group_data.std()
                })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df.round(3))
        
    else:
        st.warning("Required columns not available for raincloud plot")
else:
    st.warning("GDP column not available for creating income groups")


# =============================================================================
# VISUALIZATION 4: MERGED DATASET - 3D Scatter Plot
# =============================================================================

st.header("4. 3D Scatter Plot - Multidimensional Heart Disease Analysis")

st.markdown("""
The 3D plots you see combine four features at the same time: 

**Features across x,y,z axis**: heart disease prevalence, GDP of a country, and life expectancy

**Color of the data point**: Heart Disease Rate

From the first left plot we see that *China* and the *United States* go far beyound all other countries. To go more into the details, these countries are removed from the plot on the right.""")

if len(merged_df) > 0:

    # Select relevant columns
    available_cols = merged_df.columns
    x_col = 'GDP' if 'GDP' in available_cols else 'std_rate_2022'
    y_col = 'Life expectancy' if 'Life expectancy' in available_cols else 'dalys_2021'
    z_col = 'Physicians per Thousand' if 'Physicians per Thousand' in available_cols else 'prevalence_2021'
    size_col = 'Urban population' if 'Urban population' in available_cols else 'std_rate_2022'

    # =======================================================
    # FIRST ROW — SIDE-BY-SIDE (Full vs Without China & US)
    # =======================================================
    col1, col2 = st.columns(2)

    # -------------------------
    # Left: Full dataset
    # -------------------------
    with col1:
        st.subheader("Full Dataset")

        fig_full = px.scatter_3d(
            merged_df,
            x=x_col, y=y_col, z=z_col,
            size=size_col,
            color='std_rate_2022',
            hover_name='country',
            hover_data={
                'std_rate_2022': True,
                'GDP': ':.2f' if 'GDP' in available_cols else None,
                'Life expectancy': ':.1f' if 'Life expectancy' in available_cols else None
            },
            color_continuous_scale=px.colors.sequential.Plasma,
            title="3D Analysis: Economic vs Health vs Healthcare Factors",
            labels={
                'std_rate_2022': 'Heart Disease Rate',
                'GDP': 'Economic Development',
                'Life expectancy': 'Health Outcomes',
                'Physicians per Thousand': 'Healthcare Access'
            }
        )

        fig_full.update_layout(
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            height=700
        )

        st.plotly_chart(fig_full, use_container_width=True)

    # -------------------------
    # Right: Without China & US
    # -------------------------
    with col2:
        st.subheader("Without China & United States")

        filtered_df = merged_df[~merged_df['country'].isin(['China', 'United States'])]

        fig_filtered = px.scatter_3d(
            filtered_df,
            x=x_col, y=y_col, z=z_col,
            size=size_col,
            color='std_rate_2022',
            hover_name='country',
            hover_data={
                'std_rate_2022': True,
                'GDP': ':.2f' if 'GDP' in available_cols else None,
                'Life expectancy': ':.1f' if 'Life expectancy' in available_cols else None
            },
            color_continuous_scale=px.colors.sequential.Plasma,
            title="3D Analysis (Filtered)",
            labels={
                'std_rate_2022': 'Heart Disease Rate',
                'GDP': 'Economic Development',
                'Life expectancy': 'Health Outcomes',
                'Physicians per Thousand': 'Healthcare Access'
            }
        )

        fig_filtered.update_layout(
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            height=700
        )

        st.plotly_chart(fig_filtered, use_container_width=True)

    # =======================================================
    # SECOND ROW — Full Copy but with 7 more countries removed
    # =======================================================
    st.subheader("Filtered Further: Removing Top Economic Powers")

    remove_countries = [
        "China", "United States",
        "Japan", "Germany", "India",
        "United Kingdom", "South Korea",
        "France", "Italy"
    ]

    deeply_filtered_df = merged_df[~merged_df['country'].isin(remove_countries)]

    fig_deep = px.scatter_3d(
        deeply_filtered_df,
        x=x_col, y=y_col, z=z_col,
        size=size_col,
        color='std_rate_2022',
        hover_name='country',
        hover_data={
            'std_rate_2022': True,
            'GDP': ':.2f' if 'GDP' in available_cols else None,
            'Life expectancy': ':.1f' if 'Life expectancy' in available_cols else None
        },
        color_continuous_scale=px.colors.sequential.Plasma,
        title="9 additional largest economies removed to compare other countries with lower GDP",
        labels={
            'std_rate_2022': 'Heart Disease Rate',
            'GDP': 'Economic Development',
            'Life expectancy': 'Health Outcomes',
            'Physicians per Thousand': 'Healthcare Access'
        }
    )

    fig_deep.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        height=700
    )

    st.plotly_chart(fig_deep, use_container_width=True)


# =============================================================================
# COMPREHENSIVE STATISTICAL ANALYSIS
# =============================================================================


st.header("Cross-Dataset Correlation Heatmap")
st.write("""
         Since two datasets were combine into one, it is natural to ask yourself a question: how does the correlation heatmap look between two datasets?
         
         On the bottom left we see features in very bright red colors from the first dataset, while features from the dataset have a lower correlation.
         
         This observation will be important for feature engineering when building a machine learning model, since one dataset with only four features might dominate over a larger dataset.
         """)
if len(merged_df) > 0:
    # Select numerical columns for correlation
    numerical_merged = merged_df.select_dtypes(include=[np.number])
    if len(numerical_merged.columns) > 1:
        # Calculate correlation matrix
        corr_matrix = numerical_merged.corr()
        
        # Create annotated heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig_corr.update_layout(
            title="Correlation Matrix: Heart Disease vs Socioeconomic Indicators",
            height=600,
            xaxis_title="Features",
            yaxis_title="Features"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)

# Statistical summaries
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Overview")
    overview_data = {
        'Dataset': ['Heart Disease', 'Country Stats', 'Indicators', 'Merged'],
        'Records': [len(df_heart), len(df_country), len(df_indicators), len(merged_df)],
        'Features': [len(df_heart.columns), len(df_country.columns), 
                    len(df_indicators.columns), len(merged_df.columns)]
    }
    st.dataframe(pd.DataFrame(overview_data))

with col2:
    st.subheader("Data Quality Summary")
    quality_data = {
        'Dataset': ['Heart Disease', 'Country Stats', 'Indicators'],
        'Missing Values': [df_heart.isnull().sum().sum(), 
                          df_country.isnull().sum().sum(),
                          df_indicators.isnull().sum().sum()],
        'Complete Rate %': [((1 - df_heart.isnull().sum().sum() / (len(df_heart) * len(df_heart.columns))) * 100).round(2),
                           ((1 - df_country.isnull().sum().sum() / (len(df_country) * len(df_country.columns))) * 100).round(2),
                           ((1 - df_indicators.isnull().sum().sum() / (len(df_indicators) * len(df_indicators.columns))) * 100).round(2)]
    }
    st.dataframe(pd.DataFrame(quality_data))

# Advanced statistical tests
st.subheader("Advanced Statistical Tests")
st.write("Question to the audience - are there any additional comprehensive statistical analysis steps I can add?")