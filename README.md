<p align="center">
 <img width="100%" src="./images/banner_image.png " align="center" alt="Project banner" />
 <h1 align="center">Spotify clustering and recommendation system</h1>
 <p align="center">Analysing Spotify data, clustering and creating a predictive recommendation system.</p>
</p>

<p align="center">
  <br/>
  <a href="https://www.python.org/" title="Python official website"><img alt="Python Logo" height="30px" src="./images/python-logo.png" /></a>
  <a href="https://pandas.pydata.org/" title="Pandas official wesbite"><img alt="Pandas Logo" height="30px" src="./images/pandas-logo.png" /></a>
   <a href="https://matplotlib.org/stable/" title="Matplotlib offical website"><img alt="Matplotlib Logo" height="30px" src="./images/matplotlib-logo.png" /></a>
  <a href="https://seaborn.pydata.org/" title="Seaborn offical website"><img alt="Seaborn Logo" height="30px" src="./images/seaborn-logo.png" /></a>
  <a href="https://www.kaggle.com/" title="Kaggle offical website"><img alt="Kaggle Logo" height="30px" src="./images/kaggle-logo.png" /></a>
  <br />
</p>

<p align="center">
  <a href="https://scikit-learn.org/stable/" title="Scikit-learn official website"><img alt="Scikit-learn Logo" height="30px" src="./images/sckit-learn.png" /></a>
  <a href="https://docs.streamlit.io/" title="Streamlit offical website"><img alt="Streamlit Logo" height="30px" src="./images/streamlit.png" /></a>
  <br />
</p>

<p align="center">
    <a href="https://github.com/users/petedanielsmith/projects/5">Project Board</a>
    &nbsp;&nbsp;-&nbsp;&nbsp;
    <a href="./jupyter_notebooks/01_data_clean_and_eda.ipynb">Jupyter Notebooks</a>
    &nbsp;&nbsp;-&nbsp;&nbsp;
    <a href="https://spotify-recommendation-system-code-institute.streamlit.app/">Streamlit Dashboard</a>
    &nbsp;&nbsp;-&nbsp;&nbsp;
    <a href="#conclusions">Conclusions</a>
    <br/><br/><br/>
</p>

<details>
<summary align="center">Table of contents (Click to show)</summary>

<p>
  <br/>
</p>

- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
- [Hypothesis and how to validate](#hypothesis-and-how-to-validate)
- [Project Plan](#project-plan)
- [The rationale to map the business requirements to the Data Visualisations](#the-rationale-to-map-the-business-requirements-to-the-data-visualisations)
- [Analysis techniques used](#analysis-techniques-used)
- [Ethical considerations](#ethical-considerations)
- [Dashboard Design](#dashboard-design)
- [Unfixed Bugs](#unfixed-bugs)
- [Development Roadmap](#development-roadmap)
- [Conclusions](#conclusions)
- [Deployment](#deployment)
- [Main Data Analysis Libraries](#main-data-analysis-libraries)
- [Credits](#credits)
  - [Content](#content)
  - [Media](#media)

---

</details>

<details>
<summary align="center">How to use this repo (Click to show)</summary>

<p>
  <br/>
</p>

**Make sure you have:**

- Python installed, this project used V3.12,
- VS Code latest

**Inside VS Code:**

Open Extensions (Ctrl+Shift+X or ⇧⌘X on macOS)
Install these extensions if you don't have them:

- Python extension (by Microsoft in the Extensions Marketplace)
- Jupyter extension (also by Microsoft)

**From the terminal:**

Open the folder in a terminal where you want the project to be saved

#### Run git clone:

```
git clone https://github.com/petedanielsmith/spotify-recommendation-system.git
```

#### Navigate in to the new folder:

```
cd spotify-recommendation-system
```

#### Setup a virtual enviroment:

Create a virtual enviroment for the project.

Linux / Mac:

```
python3 -m venv .venv
source .venv/bin/activate
```

Windows CMD:

```
python3 -m venv .venv
.venv\Scripts\activate
```

Windows PowerShell:

```
python3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Install the dependancies:

This will install all the dependancies needed for the project in to the virtual enviroment if it is setup, rather than globally

```
pip install -r requirements.txt
```

#### Select the Kernel

There is a drop down at the top of the notebooks to select your kernal that will run the Python.
If you setup a virtual enviroment then make sure you pick the venv one.

---

</details>

<p>
  <br/>
</p>

## Team Members
- Cosmin Manolescu - [https://www.linkedin.com/in/cosmin-manolescu95/](https://www.linkedin.com/in/cosmin-manolescu95/)
- Duminda Gamage - [https://www.linkedin.com/in/dumindap-gamage/](https://www.linkedin.com/in/dumindap-gamage/)
- Kumudu Saranath Liyanage - 
- Pete Smith - [https://www.linkedin.com/in/petedanielsmith/](https://www.linkedin.com/in/petedanielsmith/)


## Dataset Content

The dataset used in this project can be downloaded from [Kaggle: Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset). It is a dataset of Spotify tracks over a range of 125 different genres. Each track has some audio features associated with it.

**Columns include:**

- `number`: index number.
- `track_id`: The Spotify ID for the track.
- `artists`: The artists' names who performed the track. If there is more than one artist, they are separated by a `;`.
- `album_name`: The album name in which the track appears.
- `track_name`: Name of the track.
- `popularity`: The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently. Artist and album popularity is derived mathematically from track popularity.
- `duration_ms`: The track length in milliseconds.
- `explicit`: Whether or not the track has explicit lyrics (true = yes it does; false = no it does not OR unknown).
- `danceability`: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
- `energy`: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale.
- `key`: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.
- `loudness`: The overall loudness of a track in decibels (dB).
- `mode`: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
- `speechiness`: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
- `acousticness`: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
- `instrumentalness`: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content.
- `liveness`: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
- `valence`: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).
- `tempo`: The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
- `time_signature`: An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of 3/4, to 7/4.
- `track_genre`: The genre in which the track belongs.

## Business Requirements

TODO

## Hypothesis and how to validate?

TODO

## Project Plan

TODO

## The rationale to map the business requirements to the Data Visualisations

TODO

## Analysis techniques used

TODO

## Ethical considerations

This project uses a publicly available Spotify dataset containing track-level metadata and audio features such as popularity, genre, tempo, energy, and danceability. The dataset does not include any personally identifiable information (PII) or individual user listening histories. As a result, the project does not present risks related to user privacy, data protection, or GDPR compliance.

Despite the absence of personal data, ethical considerations remain relevant. The dataset reflects platform-driven popularity and listening trends, which may introduce bias into the recommendation system. Popular artists and genres may be over-represented, while niche or emerging artists may be under-recommended. Recommendations based solely on audio features may also fail to capture cultural, contextual, or subjective aspects of musical preference.

This system is developed for educational and coursework purposes only, with the goal of exploring recommendation techniques rather than influencing user behaviour or commercial outcomes. The limitations of the dataset and model are acknowledged, and the recommendations should be interpreted as illustrative rather than authoritative. Future extensions of the project could include bias evaluation, diversity-aware recommendation strategies, and improved transparency in recommendation logic.

## Dashboard Design

TODO


## Unfixed Bugs

TODO


## Development Roadmap

TODO

## Conclusions

TODO

## Deployment

The dashboard app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) and can be accessed by visiting this URL in a web browser:

[https://spotify-recommendation-system-code-institute.streamlit.app/](https://spotify-recommendation-system-code-institute.streamlit.app/)

It can be ran locally by running:
```
streamlit run dashboard_app/main.py 
```

## Main Data Analysis Libraries

TODO

## Credits 

### Content 

TODO

### Media

TODO

## Acknowledgements (optional)

TODO
