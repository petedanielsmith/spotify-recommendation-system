import streamlit as st

st.set_page_config(
    layout="wide",
)

st.html("""
<div style="background-color:#1DB954;padding:10px;border-radius:10px">
    <img src="https://storage.googleapis.com/pr-newsroom-wp/1/2023/05/Spotify_Full_Logo_RGB_White.png" alt="Spotify Logo" style="height:60px;display:block;margin-left:auto;margin-right:auto;">
    <h1 style="color:white;text-align:center;">Spotify Music Recommendation Dashboard</h1>
</div>        
        """)

st.markdown("""
This dashboard provides insights into Spotify's music data, including exploratory data analysis, clustering of songs based on audio features, 
and a prediction & recommendation engine that suggests songs based on user-defined audio characteristics.
""")

st.divider()

st.markdown("""
## Created by:
-  Cosmin Manolescu - [https://www.linkedin.com/in/cosmin-manolescu95/](https://www.linkedin.com/in/cosmin-manolescu95/)
- Duminda Gamage - [https://www.linkedin.com/in/dumindap-gamage/](https://www.linkedin.com/in/dumindap-gamage/)
- Kumudu Saranath Liyanage - [ https://www.linkedin.com/in/kumudu-s-liyanage/]( https://www.linkedin.com/in/kumudu-s-liyanage/)
- Pete Smith - [https://www.linkedin.com/in/petedanielsmith/](https://www.linkedin.com/in/petedanielsmith/)
""")

st.divider()

st.markdown("""
            The objective of this project is to design and evaluate a data-driven music recommendation system using Spotify track data. The system is developed for educational purposes, demonstrating the application of data engineering, exploratory analysis, unsupervised learning, and predictive modelling techniques.

1. Data Ingestion and ETL
  - Ingest a publicly available Spotify dataset containing track metadata and audio features.
  - Clean, transform, and structure the data to ensure consistency, completeness, and suitability for analysis.
  - Handle missing values, duplicates, and feature scaling as part of the ETL pipeline.
  - Produce a processed dataset that can be reused across analysis and modelling stages.

2. Exploratory Data Analysis (EDA)
  - Perform exploratory analysis to understand the distribution and relationships of key audio features (e.g. energy, tempo, valence, popularity).
  - Identify trends, correlations, and potential biases within the dataset.
  - Visualise feature distributions and clustering tendencies to inform modelling decisions.

3. Clustering and Segmentation
  - Apply unsupervised learning techniques (e.g. K-Means) to group tracks based on audio feature similarity.
  - Evaluate cluster quality and interpretability.
  - Use clustering results to identify distinct musical styles or listening profiles.

4. Predictive Recommendation System
  - Develop a recommendation approach that suggests tracks based on similarity to a given song or cluster.
  - Use audio features and clustering outputs to generate predictive recommendations.

5. Evaluation and Documentation
  - Assess model outputs qualitatively and quantitatively where appropriate.
  - Document assumptions, limitations, and design decisions.
            """)