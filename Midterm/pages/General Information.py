import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datasource import *
import plotly.express as px

st.set_page_config(page_title="General Information", page_icon="*")

country_data = load_country()

st.title("Heart Disease Distrubution ")

# https://plotly.com/python/choropleth-maps/
# https://plotly.github.io/plotly.py-docs/generated/plotly.express.choropleth.html
countries = px.choropleth(
    country_data, locations="country",
    locationmode="country names",
    color="HeartDiseaseRatesAgeStandardizedRate_2022",
    hover_name="country",
    color_continuous_scale=px.colors.sequential.Plasma,
    title="Global Distribution of Heart Disease"
)

st.plotly_chart(countries)