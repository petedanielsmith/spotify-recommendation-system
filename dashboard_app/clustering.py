import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parents[1]
# --- Define Absolute File Paths ---
CLUSTER_SUMMARY_PATH = CURRENT_DIR / 'models/cluster_summary_k10.xlsx'

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Cluster Insights", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_excel(CLUSTER_SUMMARY_PATH)
    return df

cluster_summary = load_data()

cluster_names = {
    0: "Extreme & Aggressive",
    1: "Live & Rhythmic",
    2: "Upbeat Dance Mainstream",
    3: "Groovy & Energetic",
    4: "Long-Form Electronic",
    5: "Ambient & Classical Calm",
    6: "Danceable & Playful",
    7: "Positive Party Pop",
    8: "Neutral Dance Groove",
    9: "Acoustic & Traditional"
}

# add cluster names to the dataframe
cluster_summary['cluster_name'] = cluster_summary['cluster'].map(cluster_names)

# needed for radar plot
RADAR_FEATURES = [
    "danceability (%)",
    "energy (%)",
    "tempo (%)",
    "loudness (%)",
    "valence (%)",
    "acousticness (%)",
    "instrumentalness (%)",
    "speechiness (%)",
    "liveness (%)"
]


def plot_cluster_radar(cluster_summary, cluster_id, features):
    """
    Creates a more easthetic radar plot for the given cluster.
    
    :param cluster_summary: DataFrame containing cluster summary statistics
    :param cluster_id: ID of the cluster to plot
    :param features: List of features to include in the radar plot

    :return: Matplotlib figure object
    """     

    values = cluster_summary.loc[cluster_id, features].values

    # Radar setup
    N = len(features)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += [angles[0]]
    values = np.concatenate([values, [values[0]]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, values, linewidth=2, color="#1DB954")  # Spotify green
    ax.fill(angles, values, alpha=0.25, color="#1DB954")

    ax.set_thetagrids(np.degrees(angles[:-1]), features, fontsize=9)
    ax.set_ylim(0, 100)

    ax.set_title(
        f"{cluster_summary.loc[cluster_id, 'cluster_name']}",
        fontsize=14,
        pad=20
    )

    ax.grid(color="gray", alpha=0.3)
    ax.spines["polar"].set_visible(False)

    return fig

# profile info dictionary
cluster_profiles = {
    0: {
        "name": "Extreme & Aggressive",
        "profile": "Very high energy and loudness, low danceability, low acousticness, low valence.",
        "story": (
            "This cluster groups intense, aggressive tracks with high loudness and energy. "
            "Songs are emotionally charged and raw rather than danceable, often favoring "
            "distortion and power over groove."
        ),
        "example_genres": ["Grindcore", "Death Metal", "Black Metal"],
        "user_facing": "You might like this if you enjoy loud, aggressive, high-energy music."
    },

    1: {
        "name": "Live & Rhythmic",
        "profile": "High liveness, moderate energy, moderate danceability, longer duration.",
        "story": (
            "Tracks in this cluster feel live, rhythmic, and performance-driven. "
            "They balance groove with crowd interaction, often sounding like recordings "
            "made for or inspired by live settings."
        ),
        "example_genres": ["Pagode", "Sertanejo", "Samba"],
        "user_facing": "You might like this if you enjoy rhythmic music with a live, organic feel."
    },

    2: {
        "name": "Upbeat Dance Mainstream",
        "profile": "High danceability, high energy, fast tempo, positive valence.",
        "story": (
            "This cluster captures upbeat, energetic tracks designed for dancing and broad appeal. "
            "Songs are rhythm-forward, emotionally positive, and highly suitable for parties or workouts."
        ),
        "example_genres": ["House", "EDM", "Dance"],
        "user_facing": "You might like this if you enjoy energetic, feel-good dance music."
    },

    3: {
        "name": "Groovy & Energetic",
        "profile": "Similar to upbeat dance tracks, but with slightly lower valence and tempo.",
        "story": (
            "Tracks here emphasize groove and momentum over pure brightness. "
            "Compared to more upbeat dance clusters, these songs feel slightly darker "
            "or more grounded while remaining energetic."
        ),
        "example_genres": ["Country", "Party", "Salsa"],
        "user_facing": (
            "You might like this if you enjoy energetic music with a strong groove "
            "but less overt positivity."
        )
    },

    4: {
        "name": "Long-Form Electronic",
        "profile": "Very long duration, high energy, moderate danceability, low acousticness.",
        "story": (
            "This cluster contains extended tracks with evolving structure and sustained intensity. "
            "Songs often prioritize atmosphere and progression over short, punchy hooks."
        ),
        "example_genres": ["Minimal Techno", "Detroit Techno", "Chicago House"],
        "user_facing": "You might like this if you enjoy long, immersive electronic tracks."
    },

    5: {
        "name": "Ambient & Classical Calm",
        "profile": (
            "Very high acousticness, very low energy and loudness, high instrumentalness."
        ),
        "story": (
            "This cluster represents calm, introspective, and often instrumental music. "
            "Tracks are quiet, spacious, and emotionally neutral or reflective."
        ),
        "example_genres": ["New Age", "Classical", "Ambient"],
        "user_facing": (
            "You might like this if you enjoy calm, instrumental, or relaxing music."
        )
    },

    6: {
        "name": "Danceable & Playful",
        "profile": "High danceability, moderate energy, high tempo, moderate valence.",
        "story": (
            "Songs in this cluster are rhythmically engaging and playful, with a strong sense of movement. "
            "Compared to mainstream dance clusters, they feel more quirky or stylistically diverse."
        ),
        "example_genres": ["J-Dance", "Dancehall", "Funk"],
        "user_facing": (
            "You might like this if you enjoy fun, danceable music with personality."
        )
    },

    7: {
        "name": "Positive Party Pop",
        "profile": (
            "High danceability and energy, with higher valence than similar dance clusters."
        ),
        "story": (
            "This cluster leans toward brighter, happier party tracks. "
            "Compared to other dance-heavy clusters, songs here are more emotionally upbeat "
            "and celebratory."
        ),
        "example_genres": ["Party", "Synth Pop", "Power Pop"],
        "user_facing": (
            "You might like this if you enjoy upbeat, cheerful party music."
        )
    },

    8: {
        "name": "Neutral Dance Groove",
        "profile": (
            "Danceable and energetic tracks with lower emotional extremes."
        ),
        "story": (
            "Tracks here are rhythmically strong but emotionally neutral. "
            "They work well as background dance or club music without demanding "
            "emotional engagement."
        ),
        "example_genres": ["Latino", "Turkish", "Reggaeton"],
        "user_facing": (
            "You might like this if you enjoy steady, groove-based dance music."
        )
    },

    9: {
        "name": "Acoustic & Traditional",
        "profile": "Highly acoustic, low energy, moderate duration, low loudness.",
        "story": (
            "This cluster groups acoustic, traditional, and folk-oriented tracks. "
            "Songs emphasize instrumentation and storytelling over intensity or rhythm."
        ),
        "example_genres": ["Honky Tonk", "Tango", "Romance"],
        "user_facing": (
            "You might like this if you enjoy acoustic, traditional, or folk-style music."
        )
    }
}

# --- HEADER ---
st.title("🎧 Spotify Music Profiles")
st.markdown("""
On this page, explore the distinct music profiles identified through clustering analysis.
""")

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["🧭 Cluster Profiles", "🛠️ Building the Clusters"])

# ==============================================================================
# TAB 1: EXECUTIVE INSIGHTS (BUSINESS)
# ==============================================================================
with tab1:
    st.info(
    "Each cluster represents a distinct music listening profile based on audio characteristics. "
)

    # Build selection mapping, show names in selectbox, but use cluster IDs internally
    cluster_options = (
        cluster_summary[["cluster", "cluster_name"]]
        .drop_duplicates()
        .sort_values("cluster")
    )

    cluster_label_to_id = {
        row["cluster_name"]: row["cluster"]
        for _, row in cluster_options.iterrows()
    }

    selected_label = st.selectbox(

    "Select a profile to explore its sound signature and characteristics:",
        options=list(cluster_label_to_id.keys())
    )

    selected_cluster = cluster_label_to_id[selected_label]

    # create radar plot
    fig = plot_cluster_radar(
        cluster_summary,
        selected_cluster,
        RADAR_FEATURES
    )

    st.pyplot(fig)

    # profile info
    profile = cluster_profiles[selected_cluster]

    st.markdown("### Profile description")
    st.markdown(f"""
    ***{profile['name']}***

    - **Profile:**  
    {profile['profile']}

    - **Story:**  
    {profile['story']}

    - **Example genres:**  
    {", ".join(profile['example_genres'])}

    > *{profile['user_facing']}*
    """)