import streamlit as st

pages = [
    st.Page("intro.py", title="Introduction", icon=":material/home:"),
    st.Page("eda.py", title="EDA - Exploratory Data Analysis", icon=":material/bar_chart:"),
    st.Page("clustering.py", title="Clustering", icon=":material/diversity_3:"),
    st.Page("prediction_recommendation.py", title="Prediction - Recommendation", icon=":material/genres:"),
]

pg = st.navigation(pages)
pg.run()