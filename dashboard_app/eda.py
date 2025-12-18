import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy import stats

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Spotify EDA & Trends", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # Data file is in the /data/processed/ directory
    df = pd.read_csv('./data/processed/cleaned_spotify_dataset.csv')
    # Pre-calculate minutes for the whole dashboard
    df['duration_min'] = df['duration_ms'] / 60000
    return df

df = load_data()

# --- HEADER ---
st.title("🎵 Spotify Data Analysis & Trends")
st.markdown("""
This dashboard explores the characteristics of 109k tracks to understand global music trends. 
Audio features, genre popularity, and the impact of song duration are analyzed to inform our Recommendation Engine.
""")

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["📊 Executive Insights", "🧪 Data Abalytics Lab"])

# ==============================================================================
# TAB 1: EXECUTIVE INSIGHTS (BUSINESS)
# ==============================================================================
with tab1:
    st.header("Global Music Trends & Key Insights")
    
    # --- SECTION 1: GENRE TRENDS (FULL WIDTH) ---
    st.subheader("1. The 'Mainstream' Gap")
    st.info("**Insight:** Popularity is not evenly distributed. A small set of genres (Pop, K-Pop) dominates, while niche genres show high variance.")
    
    # Prepare Data for Box Plot
    genre_ranking = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False)
    top_10 = genre_ranking.head(10).index.tolist()
    bottom_10 = genre_ranking.tail(10).index.tolist()
    
    df_summary = df[df['track_genre'].isin(top_10 + bottom_10)].copy()
    df_summary['Type'] = df_summary['track_genre'].apply(lambda x: 'Mainstream (Top 10)' if x in top_10 else 'Niche (Bottom 10)')
    
    # Sort order for the x-axis to be clean
    sort_order = top_10 + bottom_10
    
    fig_box = px.box(df_summary, x='track_genre', y='popularity', color='Type', 
                     title="Popularity Distribution: Mainstream vs. Niche Genres",
                     category_orders={"track_genre": sort_order},
                     color_discrete_sequence=['#1DB954', '#FF4B4B']) # Spotify Green & Red
    
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # --- SECTION 2: DURATION TRENDS (TWO COLUMNS) ---
    st.subheader("2. The 'Radio Edit' Effect")
    st.info("**Insight:** Song length matters. Tracks between **3-4 minutes** are statistically most likely to be popular (The 'Sweet Spot'). This justify the prevalence of 'Radio Edits' in the music industry.")

    col_radio1, col_radio2 = st.columns(2)

    with col_radio1:
        #st.markdown("**Average Popularity by Duration Bin**")
        # Prepare Binned Data
        bins = [0, 2, 3, 4, 5, 20]
        labels = ['< 2 min', '2-3 min', '3-4 min', '4-5 min', '> 5 min']
        df['duration_bin'] = pd.cut(df['duration_min'], bins=bins, labels=labels)
        bin_pop = df.groupby('duration_bin')['popularity'].mean().reset_index()
        
        fig_bar = px.bar(bin_pop, x='duration_bin', y='popularity', 
                         title="Avg Popularity per Time Bin",
                         color='popularity', color_continuous_scale='Greens')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_radio2:
        # st.markdown("**Density Cluster (Where the hits live)**")
        # Density Heatmap (Interactive Hexbin alternative)
        fig_density = px.density_heatmap(df, x='duration_min', y='popularity',
                                         title="Density: Duration vs Popularity",
                                         nbinsx=30, nbinsy=30,
                                         color_continuous_scale='Blues',
                                         range_x=[0, 8]) # Limit x-axis to relevant range
        st.plotly_chart(fig_density, use_container_width=True)

    st.divider()
    
    # --- SECTION 3: BUSINESS RECOMMENDATIONS from EDA ---
    st.subheader("💡 Potential enhancements for the recommendation system based on Analysis Outcome")
    st.markdown("""
    * **Cold Start Strategy:** Likely need a **"Popularity Bias"** weight for new users, favoring the 'Mainstream' cluster (Pop-Film, K-Pop) until we learn their specific tastes.
    * **Content Filtering Rules:** Implement a soft filter for tracks **> 6 minutes** for general listeners. These should only be recommended if the user specifically interacts with 'Classical', 'Techno', or 'Ambient' genres.
    * **New Feature:** Danceability and Energy are key drivers for the general user base. However, 'Acousticness' is the defining feature for niche communities use this to quickly segment "Chill" users from "Party" users.
    """)

# ==============================================================================
# TAB 2: DATA ABALYTICS LAB (TECHNICAL)
# ==============================================================================
with tab2:
    st.header("Technical EDA & Statistical Validation")
    st.markdown("This section details data distribution, normality checks, and formal hypothesis testing.")

    # --- SECTION 1: DATA DISTRIBUTIONS & NORMALITY ---
    st.subheader("1. Distributions & Normality Checks")
    
    col_tech1, col_tech2 = st.columns([1, 2])
    
    with col_tech1:
        st.write("**Normality Tests (Shapiro-Wilk)**")
        st.caption("Sample size N=5000 for performance")
        
        normality_results = {}
        features = ['popularity', 'danceability', 'energy', 'tempo', 'valence']
        
        for feat in features:
            stat, p = stats.shapiro(df[feat].sample(5000, random_state=42))
            normality_results[feat] = p
            
        norm_df = pd.DataFrame.from_dict(normality_results, orient='index', columns=['P-Value'])
        norm_df['Distribution'] = norm_df['P-Value'].apply(lambda x: 'Normal' if x > 0.05 else 'Skewed (Reject H0)')
        st.dataframe(norm_df)
        
        st.warning("⚠️ **Finding:** All key features are **Non-Normal**. We must use Non-Parametric tests (Mann-Whitney, Kruskal-Wallis) and MinMax Scaling.")

    with col_tech2:
        feature_select = st.selectbox("Visualize Distribution:", features)
        fig_hist = px.histogram(df, x=feature_select, nbins=50, title=f"Distribution of {feature_select}", marginal="box")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # --- SECTION 2: CORRELATION ---
    st.subheader("2. Feature Correlation Matrix")
    
    corr_cols = ['popularity', 'duration_ms', 'danceability', 'energy', 'loudness', 
                 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    corr_matrix = df[corr_cols].corr()
    
    fig_corr, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig_corr)
    
    st.markdown("""
    **Key Technical Findings:**
    * **Energy vs. Loudness (0.78):** High collinearity. *Action:* We may drop 'Loudness' for clustering to avoid double-counting this signal.
    * **Duration vs. Popularity (~0.00):** No linear correlation, confirming the need for non-linear binning (proven in Hypothesis 2).
    """)
    
    st.divider()

    # --- SECTION 3: HYPOTHESIS TESTING ---
    st.subheader("3. Formal Hypothesis Testing")

    # Hypothesis 1
    st.markdown("#### H1: Genre Popularity Difference")
    st.code("Test: Mann-Whitney U | Groups: Top 10 Genres vs Bottom 10 Genres", language="python")
    
    # Logic for hypothesis testing display
    top_pop = df[df['track_genre'].isin(top_10)]['popularity']
    bot_pop = df[df['track_genre'].isin(bottom_10)]['popularity']
    stat_h1, p_h1 = stats.mannwhitneyu(top_pop, bot_pop)
    
    col_h1_1, col_h1_2 = st.columns(2)
    col_h1_1.metric("Mann-Whitney U Statistic", f"{stat_h1:.2e}")
    col_h1_2.metric("P-Value", f"{p_h1:.2e}", delta="Significant" if p_h1 < 0.05 else "Not Significant")
    
    # Hypothesis 2
    st.markdown("#### H2: Duration Impact on Popularity")
    st.code("Test: Kruskal-Wallis H | Groups: 5 Duration Bins", language="python")
    
    groups = [df[df['duration_bin'] == label]['popularity'] for label in labels]
    stat_h2, p_h2 = stats.kruskal(*groups)
    
    col_h2_1, col_h2_2 = st.columns(2)
    col_h2_1.metric("K-Wallis H Statistic", f"{stat_h2:.2f}")
    col_h2_2.metric("P-Value", f"{p_h2:.2e}", delta="Significant" if p_h2 < 0.05 else "Not Significant")

    st.success("""
    **Conclusion:** 
    1. There is a statistically significant popularity gap. So, the genre is a considerable predictor of popularity.
    2. Duration has a statistically significant non-linear "sweet spot."
    """)