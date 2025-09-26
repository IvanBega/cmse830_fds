import pandas as pd
import numpy as np
import streamlit as st

def load_and_clean():
    df = pd.read_csv("heart_disease.csv")
    return df