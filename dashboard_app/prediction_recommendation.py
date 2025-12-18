import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import euclidean_distances
from pathlib import Path

# --- Constants & Configuration ---
KEY_MAPPING = {
    -1: "Unknown",
    0: "C",
    1: "C♯ / D♭",
    2: "D",
    3: "D♯ / E♭",
    4: "E",
    5: "F",
    6: "F♯ / G♭",
    7: "G",
    8: "G♯ / A♭",
    9: "A",
    10: "A♯ / B♭",
    11: "B",
}

# Mapping cluster IDs to meaningful names (Top Genre per cluster)
CLUSTER_NAMES = {
    0: "Extreme/Metal",
    1: "Rhythmic/Pagode",
    2: "Electronic/House",
    3: "Country/Folk",
    4: "Techno/Minimal",
    5: "Ambient/New-Age",
    6: "Pop/J-Dance",
    7: "Party/Dance",
    8: "Latin",
    9: "Acoustic/Piano",
}

CURRENT_DIR = Path(__file__).resolve().parents[1]
# --- Define Absolute File Paths ---
MODEL_PATH = CURRENT_DIR / 'models/best_spotify_model.pkl'
CLUSTER_DATA = CURRENT_DIR / 'data/labelled/clusters_k10_labels.csv'

# Set page config
st.set_page_config(page_title="Music Recommendation Engine", layout="wide")

@st.cache_data
def load_data():
    """Loads the dataset with cluster labels."""
    try:
        df = pd.read_csv(CLUSTER_DATA)
        # Clean up index columns if they exist
        df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], errors='ignore')
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at: {CLUSTER_DATA}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    """Loads the pre-trained classification model."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {MODEL_PATH}. Please run the training notebook first.")
        return None

def main():
    st.title("🎵 Music Recommendation & Cluster Prediction")
    st.markdown("""
    Adjust the audio features below to discover which cluster your music taste belongs to, 
    and get song recommendations tailored to those specific values.
    """)

    # Load resources
    df = load_data()
    model = load_model()

    if model is None or df.empty:
        return

    # --- Sidebar: User Inputs ---
    st.sidebar.header("Audio Feature Selection")
    
    # Define features (Must match the training order)
    features = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo'
    ]

    # Create input widgets
    input_data = {}
    
    input_data['danceability'] = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5)
    input_data['energy'] = st.sidebar.slider("Energy", 0.0, 1.0, 0.5)
    
    # Key selection with meaningful labels
    input_data['key'] = st.sidebar.selectbox(
        "Musical Key", 
        options=range(12), 
        format_func=lambda x: KEY_MAPPING.get(x, str(x))
    )
    
    input_data['loudness'] = st.sidebar.slider("Loudness (dB)", -60.0, 0.0, -10.0)
    input_data['mode'] = st.sidebar.selectbox("Mode (0: Minor, 1: Major)", [0, 1], 1)
    input_data['speechiness'] = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.05)
    input_data['acousticness'] = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.1)
    input_data['instrumentalness'] = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.0)
    input_data['liveness'] = st.sidebar.slider("Liveness", 0.0, 1.0, 0.1)
    input_data['valence'] = st.sidebar.slider("Valence (Positivity)", 0.0, 1.0, 0.5)
    input_data['tempo'] = st.sidebar.slider("Tempo (BPM)", 50.0, 200.0, 120.0)

    # Convert input to DataFrame
    user_input_df = pd.DataFrame([input_data])

    # --- Main Content: Prediction ---
    if st.button("Predict & Recommend"):
        # 1. Predict Cluster
        predicted_cluster = model.predict(user_input_df)[0]
        
        # Get meaningful label
        predicted_label = CLUSTER_NAMES.get(predicted_cluster, f"Cluster {predicted_cluster}")
        
        st.subheader(f"Results")
        st.success(f"Based on your inputs, these songs belong to: **{predicted_label}**")

        # 2. Filter data for that cluster
        cluster_songs = df[df['cluster'] == predicted_cluster].copy()
        
        # 3. Find most relevant songs (Nearest Neighbors by Euclidean Distance)
        distances = euclidean_distances(user_input_df[features], cluster_songs[features])
        cluster_songs['similarity_distance'] = distances[0]
        
        # Sort by distance (smaller distance = more relevant)
        recommendations = cluster_songs.sort_values(by='similarity_distance').head(50)

        # Store in session state
        st.session_state['recommendations'] = recommendations
        st.session_state['predicted_cluster'] = predicted_cluster

    # --- Artist Filtering Section ---
    if 'recommendations' in st.session_state:
        recs = st.session_state['recommendations']
        
        st.write("---")
        st.subheader("🎧 Recommended Songs")
        
        # Artist Filter
        artist_filter = st.text_input("Filter by Artist Name (Optional):", placeholder="e.g., Ed Sheeran")
        
        if artist_filter:
            final_display = recs[recs['artists'].str.contains(artist_filter, case=False, na=False)]
        else:
            final_display = recs

        # Group duplicate songs & aggregate genres
        grouped_display = (
            final_display
            .groupby(['artists', 'track_name', 'album_name', 'popularity'], as_index=False)
            .agg({'track_genre': lambda x: sorted(set(x))})
        )
        grouped_display['track_genre'] = grouped_display['track_genre'].apply(lambda genres: ", ".join(genres))

        # Display Logic
        if final_display.empty:
            st.warning("No songs found matching that artist in the top recommendations.")
        else:
            # Retrieve label again for display
            cluster_val = st.session_state['predicted_cluster']
            cluster_label = CLUSTER_NAMES.get(cluster_val, f"Cluster {cluster_val}")
            
            st.write(f"Showing top results for **{cluster_label}** sorted by relevance:")
            
            for idx, row in grouped_display.iterrows():
                with st.expander(f"🎵 {row['artists']} — {row['track_name']}"):
                    st.write(f"**Album:** {row['album_name']}")
                    st.write(f"**Popularity:** {row['popularity']}")
                    st.write(f"**Genres:** {row['track_genre']}")

            # Feature Comparison
            st.write("### Feature Comparison")
            st.write("Compare your input (Red line) with the average of the recommended songs (Blue bars).")
            
            avg_recs = final_display[features].mean()
            user_vals = user_input_df.iloc[0]
            
            chart_data = pd.DataFrame({
                'Feature': features,
                'Recommended Avg': avg_recs.values,
                'User Input': user_vals.values
            }).set_index('Feature')
            
            st.bar_chart(chart_data)

if __name__ == "__main__":
    main()