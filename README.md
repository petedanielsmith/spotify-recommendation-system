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
- Kumudu Saranath Liyanage - [https://www.linkedin.com/in/kumudu-s-liyanage/](https://www.linkedin.com/in/kumudu-s-liyanage/)
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

## Hypothesis and how to validate?

* **Hypothesis 1:** All genres have the same average popularity; music preference is evenly distributed.
  * **Selected Test:** **Mann-Whitney U Test**
  * **Visual:** Box Plot comparison.

* **Hypothesis 2:** Duration *does not* impact popularity.
  * **Selected Test:** **Kruskal-Wallis H Test** 
  * **Visual:** Hexbin Density Plot and Binned Bar Chart.

## Project Plan

The prjoject follows the following steps:

1. `Extract` - Extract the data from Kaggle.
2. `Load` - Load the CSV via Pandas.
3. `Transform` - Clean and process the data using Pandas, adding new columns and checking for missing or duplicated values.
4. `Visualise` - Creating charts with Matplotlib and Seaborn to visualise trends and distributions.
5. `Analyse` - Interpret what the visualisations displayed.
6. `Unupervised Learning` - Use K-Means to cluster the data in to similar groups.
7. `Supervised Learning` - Use both Linear Regression and Random Forrest machine learning to create predictive models.
8. `Interactive Dashboard` - Use Streamlit to create an interactive dasboard to display the data and run predictive recommendations.
9. `Document` - Record findings and conclusions.

## The rationale to map the business requirements to the Data Visualisations

| Business Requirement | Data Visualisation(s) | Rationale & Hypothesis Outcome |
| :--- | :--- | :--- |
| **1. Identify trends in music preferences**<br>*(Genre Popularity)* | **Box Plot**<br>*(X=Genre, Y=Popularity)* | **Rationale:** Box plots show not just the *average* popularity, but the *variance* within a genre. This reveals if a genre is consistently popular (tight box, high median) or hit-or-miss (large spread).<br><br>**Hypothesis Outcome:** The analysis reveals that "Pop-Film" and "K-Pop" with high medians, confirming global preference for these styles. |
| **2. Visualise popular songs by time**<br>*(Duration vs. Popularity)* | **Hexbin Plot** or **Scatter Plot with Trendline**<br>*(X=Duration, Y=Popularity)* | **Rationale:** With 100k+ rows, a standard scatter plot will suffer from overplotting. A Hexbin plot groups dense points, clearly showing the "sweet spot" duration where most popular songs exist.<br><br>**Hypothesis Outcome:** The analysis reveals a significant concentration of high-popularity tracks within the 3-to-4-minute range, effectively visualizing the prevailing industry standard. |

## Analysis techniques used

### **Statistical Validation**
* **Non-Parametric Testing:** We prioritized non-parametric tests (**Mann-Whitney U** and **Kruskal-Wallis**) over standard parametric tests (T-Test/ANOVA).
    * *Reasoning:* Our "Normality Checks" (Shapiro-Wilk) confirmed that audio features and popularity scores are highly skewed. Using parametric tests on this data would lead to incorrect P-values and false conclusions.

### **Data Visualization Strategies**
* **Distribution Analysis (Box Plots):** Used to visualize the spread and central tendency of popularity across genres, highlighting outliers and consistency.
* **Density Estimation (Hexbins):** Employed for the Duration vs. Popularity analysis to handle the large dataset size (100,000+ rows). This technique reveals high-density clusters that would be invisible in a standard scatter plot.
* **Binning:** Continuous variables (like Duration) were converted into categorical bins (e.g., "Radio Edit: 3-4 min") to allow for group-based statistical comparison.

## Machine Learning Techniques

### Clustering
TO DO
### Classification

The classification phase focuses on building a robust predictive model to assign new song data or user preferences to one of the 10 identified musical clusters.

* **Objective**: Train and evaluate multiple machine learning algorithms to classify audio features into target clusters.
* **Workflow**:
    1.  **Data Preparation**: The dataset is split into training (80%) and testing (20%) sets, stratified by cluster to ensure class balance.
    2.  **Model Selection**: Three distinct classifiers are trained and compared:
        * **Random Forest Classifier** (`n_estimators=200`)
        * **Gradient Boosting Classifier**
        * **XGBoost Classifier** (optimized with `max_depth=6`, `learning_rate=0.1`)
    3.  **Evaluation**: Models are evaluated based on **Accuracy**. The notebook automatically identifies the best-performing model among the three.
    4.  **Deployment**: The champion model is serialized and saved as `best_spotify_model.pkl` for integration into the application.
* **Features Used**: 11 numerical audio attributes, including `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, and `tempo`.

## Ethical considerations

This project uses a publicly available Spotify dataset containing track-level metadata and audio features such as popularity, genre, tempo, energy, and danceability. The dataset does not include any personally identifiable information (PII) or individual user listening histories. As a result, the project does not present risks related to user privacy, data protection, or GDPR compliance.

Despite the absence of personal data, ethical considerations remain relevant. The dataset reflects platform-driven popularity and listening trends, which may introduce bias into the recommendation system. Popular artists and genres may be over-represented, while niche or emerging artists may be under-recommended. Recommendations based solely on audio features may also fail to capture cultural, contextual, or subjective aspects of musical preference.

This system is developed for educational and coursework purposes only, with the goal of exploring recommendation techniques rather than influencing user behaviour or commercial outcomes. The limitations of the dataset and model are acknowledged, and the recommendations should be interpreted as illustrative rather than authoritative. Future extensions of the project could include bias evaluation, diversity-aware recommendation strategies, and improved transparency in recommendation logic.

## Dashboard Design

The dashboard was designed with accessibility for non-technical users as a priority. Visualizations were carefully selected for clarity to ensure insights are immediately understandable, while **Plotly** was utilized to deliver a visually appealing and interactive user experience. Furthermore, textual explanations and strategic recommendations were embedded directly into the interface to ensure self-service interpretation.

### Dashboard Pages:

#### 1. EDA - Exploratory Data Analysis
This page serves as the analytical foundation of the project, presenting our Hypothesis Testing and Data Quality checks. To accommodate different stakeholders, the analysis is split into two distinct views:

* **Tab 1: Executive Insights (Business)**
    * **Focus:** High-level trends and actionable strategy.
    * **Key Visuals:**
        * **The "Mainstream" Gap:** A box plot comparing the Top 10 vs. Bottom 10 genres, highlighting the massive popularity advantage of mainstream music.
        * **The "Radio Edit" Effect:** A dual-view chart (Bar & Density) visualizing the "sweet spot" for song duration (3–4 minutes).
    * **Outcome:** Provides clear "Business Recommendations" for the recommendation engine, such as implementing a popularity bias for new users and filtering out non-musical content.

* **Tab 2: Data Science Lab (Technical)**
    * **Focus:** Statistical validity and feature engineering.
    * **Key Visuals:**
        * **Normality Checks:** A dynamic table showing Shapiro-Wilk test results to justify non-parametric testing.
        * **Correlation Matrix:** A heatmap identifying multicollinearity (e.g., Energy vs. Loudness).
        * **Formal Hypothesis Testing:** Raw outputs of Mann-Whitney U and Kruskal-Wallis tests (Statistic & P-Value) to mathematically prove that observed trends are not random noise.

#### 2. Clustering
*TODO:

#### 3. Prediction & Recommendation

The project features an interactive web application built with **Streamlit**, serving as the front-end for the recommendation engine.

* **User Interface**:
    * **Feature Tuning**: Users can define their ideal "sound" using sliders for 11 audio features.
    * **Musical Key Mapping**: A user-friendly dropdown allows selection of musical keys (e.g., C, F#, B) which are mapped internally to their integer representations.
* **Prediction Engine**:
    * Upon submission, the app loads the pre-trained `best_spotify_model.pkl`.
    * It predicts the specific **Cluster** the user's input belongs to and maps it to a descriptive genre label (e.g., *Extreme/Metal*, *Electronic/House*, *Acoustic/Piano*) using a predefined `CLUSTER_NAMES` dictionary.
* **Recommendation System**:
    1.  **Filtering**: The dataset is filtered to include only songs from the predicted cluster.
    2.  **Similarity Search**: The app calculates the **Euclidean Distance** between the user's input vector and every song in that cluster.
    3.  **Ranking**: The top 50 songs with the lowest distance (highest similarity) are retrieved.
* **Interactive Results**:
    * **Artist Filter**: Users can search within the recommendations for specific artists.
    * **Grouped Display**: Duplicate track entries are aggregated, displaying unique songs with their associated album and genres.
    * **Visual Validation**: A bar chart acts as a feedback loop, visualizing the difference between the **User's Input** (Red) and the **Average Profile of Recommended Songs** (Blue).


## Unfixed Bugs

TODO


## Development Roadmap

The project is structured into four distinct phases, ensuring a logical flow from raw data to a user-facing application.

### Phase 1: Cleaning, EDA & Hypothesis Testing (Notebook 01)
* Data Cleaning
* Hypothesis Testing
* EDA

### Phase 2: Clustering (Notebook 02)
* Feature Engineering
* Model Selection
* Evaluation 

### Phase 3: Predictions for Song Recommendation (Notebook 03)
* Classification
* Tuning 
* Recommendation Logic 

### Phase 4: Dashboard & Documentation
* Streamlit App
* README 

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

The libraries used for data analysis were:

1. `Pandas` - For data loading, transforming and cleaning.
2. `NumPy` - For data transforming.
3. `Matplotlib` - For overall multi chart layouts.
4. `Seaborn` - For a lot of the individual charts.
5. `Scikit-learn` - For machine learning alogrithms.
6. `Joblib` - For saving and loading models.
7. `Streamlit` - For creating an interactive web dashboard.

## Credits 

### Content 

- [Code institute](https://codeinstitute.net/) - The intial project structure and the LMS (Learning Managment System) from the course.
- [Kaggle](https://www.kaggle.com/) - Providing the data set used.

### Media

- [Google AI - Gemini 3](https://deepmind.google/models/gemini/) - AI generated banner logo for this README file.
- [Google Material Icons](https://fonts.google.com/icons) - Icons in the dashboard.
- [Code Institute](https://codeinstitute.net/) - Code Institute logo.
- [Python](https://www.python.org/) - Python logo image.
- [Pandas](https://pandas.pydata.org/) - Pandas logo image.
- [Matplotlib](https://matplotlib.org/) - Matplotlib logo image.
- [Seaborn](https://seaborn.pydata.org/) - Seaborn logo image.
- [Kaggle](https://www.kaggle.com/) - Kaggle logo image.
- [Scikit-learn](https://scikit-learn.org/stable/) - Scikit-learn logo image.
- [Streamlit](https://docs.streamlit.io/) - Steamlit logo image.

## Acknowledgements (optional)

TODO
