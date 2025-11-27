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
st.subheader("Statistical Analysis - Risk Factor Correlations")
correlation_matrix = df_encoded.corr()['Heart Disease Status_encoded'].sort_values(ascending=False)
st.dataframe(correlation_matrix.iloc[1:6])  # Top 5 correlations excluding itself

# =============================================================================
# VISUALIZATION 2: COUNTRY DATASET - Bubble Map
# =============================================================================

st.header("2. Interactive Bubble Map - Global Heart Disease Burden")

st.markdown("""
**Advanced Technique:** Bubble Map with Multiple Dimensions  
**Purpose:** Visualize geographic distribution with size and color encoding multiple variables  
**Insight:** Shows heart disease prevalence, mortality, and burden simultaneously across countries
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

# Statistical analysis
st.subheader("Statistical Analysis - Country Level Metrics")
country_stats = df_country[['std_rate_2022', 'dalys_2021', 'deaths_2021', 'prevalence_2021']].describe()
st.dataframe(country_stats)

# =============================================================================
# VISUALIZATION 3: INDICATORS DATASET - KDE Plot All Countries
# =============================================================================

st.header("3. KDE Analysis - Economic vs Health Indicators")

st.markdown("""
**Advanced Technique:** 2D Kernel Density Estimation  
**Purpose:** Visualize the joint distribution between economic development and health outcomes  
**Insight:** Reveals density patterns and correlations between GDP and life expectancy across all countries
""")

# Prepare data for KDE plot
df_kde = df_indicators.copy()

# Remove rows with missing values for clean visualization
df_kde_clean = df_kde[['Country', 'GDP', 'Life expectancy']].dropna()

st.write(f"**Data available for KDE:** {len(df_kde_clean)} countries with both GDP and Life expectancy data")

if len(df_kde_clean) > 1:
    # Show a preview of the cleaned data
    st.subheader("Data Preview")
    st.dataframe(df_kde_clean.head(10))
    
    # Create KDE plot for all countries
    st.subheader("KDE Plot - GDP vs Life Expectancy (All Countries)")
    
    # Use simple histogram2d for KDE visualization
    fig_kde = go.Figure()
    
    fig_kde.add_trace(go.Histogram2d(
        x=df_kde_clean['GDP'],
        y=df_kde_clean['Life expectancy'],
        colorscale='Blues',
        nbinsx=20,
        nbinsy=20
    ))
    
    fig_kde.update_layout(
        height=500,
        title="GDP vs Life Expectancy - All Countries",
        xaxis_title="GDP ($)",
        yaxis_title="Life Expectancy (Years)",
        showlegend=False
    )
    
    # Add correlation annotation
    correlation = df_kde_clean['GDP'].corr(df_kde_clean['Life expectancy'])
    fig_kde.add_annotation(
        x=0.05, y=0.95,
        xref="paper", yref="paper",
        text=f"Correlation: {correlation:.3f}",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    st.plotly_chart(fig_kde, use_container_width=True)
    
    # Scatter plot overlay to show individual countries
    st.subheader("Scatter Plot with Country Labels")
    
    fig_scatter = px.scatter(
        df_kde_clean,
        x='GDP',
        y='Life expectancy',
        hover_name='Country',
        title="GDP vs Life Expectancy - Individual Countries",
        labels={
            'GDP': 'GDP ($)',
            'Life expectancy': 'Life Expectancy (Years)'
        }
    )
    
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Log scale version for better visualization
    st.subheader("Log Scale Version for Better Visualization")
    
    # Apply log scale to GDP
    df_kde_log = df_kde_clean.copy()
    df_kde_log['GDP_log'] = np.log10(df_kde_log['GDP'] + 1)
    
    fig_log = go.Figure()
    
    fig_log.add_trace(go.Histogram2d(
        x=df_kde_log['GDP_log'],
        y=df_kde_log['Life expectancy'],
        colorscale='Greens',
        nbinsx=25,
        nbinsy=25
    ))
    
    # Customize x-axis ticks to show actual GDP values
    gdp_ticks = [1e10, 1e11, 1e12, 1e13, 1e14]
    gdp_tick_labels = ['$10B', '$100B', '$1T', '$10T', '$100T']
    gdp_log_ticks = [np.log10(gdp) for gdp in gdp_ticks]
    
    fig_log.update_layout(
        height=500,
        title="GDP (Log Scale) vs Life Expectancy - All Countries",
        xaxis_title="GDP (Log Scale)",
        yaxis_title="Life Expectancy (Years)",
        xaxis=dict(
            tickvals=gdp_log_ticks,
            ticktext=gdp_tick_labels
        ),
        showlegend=False
    )
    
    # Add correlation annotation for log scale
    correlation_log = df_kde_log['GDP_log'].corr(df_kde_log['Life expectancy'])
    fig_log.add_annotation(
        x=0.05, y=0.95,
        xref="paper", yref="paper",
        text=f"Correlation: {correlation_log:.3f}",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1
    )
    
    st.plotly_chart(fig_log, use_container_width=True)

else:
    st.error(f"""
    Insufficient data available for KDE visualization. 
    
    **Required columns:** 'GDP', 'Life expectancy'
    **Available data:** {len(df_kde_clean)} rows with both GDP and Life expectancy data
    **Total dataset:** {len(df_indicators)} countries
    """)

# Statistical analysis
st.subheader("Statistical Analysis - GDP vs Life Expectancy")

if len(df_kde_clean) > 1:
    # Calculate basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Number of Countries", len(df_kde_clean))
        st.metric("Correlation", f"{correlation:.3f}")
    
    with col2:
        st.metric("Average GDP", f"${df_kde_clean['GDP'].mean():,.0f}")
        st.metric("GDP Std Dev", f"${df_kde_clean['GDP'].std():,.0f}")
    
    with col3:
        st.metric("Average Life Expectancy", f"{df_kde_clean['Life expectancy'].mean():.1f} years")
        st.metric("Life Exp Std Dev", f"{df_kde_clean['Life expectancy'].std():.1f} years")
    
    with col4:
        st.metric("GDP Range", f"${df_kde_clean['GDP'].min():,.0f} - ${df_kde_clean['GDP'].max():,.0f}")
        st.metric("Life Exp Range", f"{df_kde_clean['Life expectancy'].min():.1f} - {df_kde_clean['Life expectancy'].max():.1f} years")
    
    # Detailed statistics table
    st.subheader("Detailed Statistics")
    stats_summary = df_kde_clean[['GDP', 'Life expectancy']].describe()
    st.dataframe(stats_summary.round(3))
    
    # Top and bottom countries
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Countries with Highest GDP:**")
        top_gdp = df_kde_clean.nlargest(5, 'GDP')[['Country', 'GDP', 'Life expectancy']]
        st.dataframe(top_gdp.reset_index(drop=True))
    
    with col2:
        st.write("**Countries with Highest Life Expectancy:**")
        top_life = df_kde_clean.nlargest(5, 'Life expectancy')[['Country', 'Life expectancy', 'GDP']]
        st.dataframe(top_life.reset_index(drop=True))

# Interpretation
st.subheader("Interpretation")
if len(df_kde_clean) > 1:
    if correlation > 0.7:
        st.success("**Strong positive correlation** - Higher GDP is strongly associated with longer life expectancy across countries.")
    elif correlation > 0.3:
        st.info("**Moderate positive correlation** - Economic development shows a meaningful relationship with health outcomes.")
    elif correlation > -0.3:
        st.warning("**Weak correlation** - Limited relationship between economic factors and life expectancy in this dataset.")
    else:
        st.error("**Negative correlation** - Unexpected inverse relationship detected.")
    
    st.write("The KDE plot shows where countries cluster in terms of economic development and health outcomes, revealing global patterns and potential outliers.")
        

# =============================================================================
# VISUALIZATION 4: MERGED DATASET - 3D Scatter Plot
# =============================================================================

st.header("4. 3D Scatter Plot - Multidimensional Heart Disease Analysis")

st.markdown("""
**Advanced Technique:** 3D Scatter Plot with Color and Size Encoding  
**Purpose:** Explore relationships between three continuous variables and heart disease rates  
**Insight:** Reveals complex interactions between economic, healthcare, and demographic factors
""")

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
        title="3D Analysis (Removing 9 Largest Economies)",
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

st.header("📈 Comprehensive Statistical Analysis")

# Correlation heatmap for merged dataset
st.subheader("Cross-Dataset Correlation Heatmap")

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

if len(merged_df) > 0 and 'std_rate_2022' in merged_df.columns:
    # ANOVA test between income groups and heart disease rates
    if 'GDP' in merged_df.columns:
        # Create income groups
        merged_df['Income_Group'] = pd.qcut(merged_df['GDP'], q=3, labels=['Low', 'Medium', 'High'])
        
        # Perform ANOVA
        groups = [group['std_rate_2022'].values for name, group in merged_df.groupby('Income_Group')]
        f_stat, p_value = stats.f_oneway(*groups)
        
        st.write(f"**ANOVA Test: Heart Disease Rates by Income Groups**")
        st.write(f"F-statistic: {f_stat:.4f}, p-value: {p_value:.4f}")
        st.write("Significant difference between income groups" if p_value < 0.05 else "No significant difference between income groups")

# Distribution analysis
st.subheader("Distribution Analysis - Key Variables")

key_variables = []
if 'std_rate_2022' in merged_df.columns:
    key_variables.append('std_rate_2022')
if 'GDP' in merged_df.columns:
    key_variables.append('GDP')
if 'Life expectancy' in merged_df.columns:
    key_variables.append('Life expectancy')

if key_variables:
    for var in key_variables:
        col1, col2 = st.columns(2)
        with col1:
            # Normality test
            stat, p_val = stats.normaltest(merged_df[var].dropna())
            st.write(f"**{var} - Normality Test**")
            st.write(f"p-value: {p_val:.4f} ({'Normal' if p_val > 0.05 else 'Not Normal'})")
        
        with col2:
            # Skewness and Kurtosis
            skew = merged_df[var].skew()
            kurt = merged_df[var].kurtosis()
            st.write(f"Skewness: {skew:.3f}, Kurtosis: {kurt:.3f}")