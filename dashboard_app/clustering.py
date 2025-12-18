import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Cluster Insights", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_excel("./models/cluster_summary_k10.xlsx")
    return df

cluster_summary = load_data()

# --- HEADER ---
st.title("🎧 Spotify Music Profiles")
st.markdown("""
On this page, explore the distinct music profiles identified through clustering analysis.
Each cluster represents a unique combination of audio characteristics, offering insight
into different listening styles and musical preferences.
""")

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["🧭 Cluster Profiles", "🛠️ Building the Clusters"])

# ==============================================================================
# TAB 1: EXECUTIVE INSIGHTS (BUSINESS)
# ==============================================================================
