import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datasource import load_and_clean

df = load_and_clean()
df.info()