<div align="center">

# 🎵 Spotify Big Data Analytics & Recommender System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![PySpark](https://img.shields.io/badge/PySpark-3.4.1-E25A1C?style=flat&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=flat&logo=mongodb&logoColor=white)](https://www.mongodb.com/atlas)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=flat&logo=plotly&logoColor=white)](https://plotly.com/)

**A full end-to-end Big Data pipeline** — ingesting over **1.2 million songs** and real user behavior data from MongoDB Atlas, engineering audio features, clustering tracks using K-Means with hyperparameter tuning, and serving personalized recommendations via a hybrid content-based + demographic-aware engine.

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Pipeline Phases](#-pipeline-phases)
  - [Phase 1 — Data Ingestion](#phase-1--data-ingestion)
  - [Phase 2 — Feature Engineering & EDA](#phase-2--feature-engineering--eda)
  - [Phase 3 — K-Means Clustering with Hyperparameter Tuning](#phase-3--k-means-clustering-with-hyperparameter-tuning)
  - [Phase 3.1 — Cluster Interpretation & Ethics](#phase-31--cluster-interpretation--ethics)
  - [Phase 3.2 — Hybrid Recommender System](#phase-32--hybrid-recommender-system)
- [ML Results](#-ml-results)
- [Recommender System](#-recommender-system)
- [Visualizations](#-visualizations)
- [Ethical Considerations](#-ethical-considerations)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)

---

## 🌟 Overview

This project demonstrates a **production-grade Big Data analytics pipeline** built entirely within Google Colab, capable of:

- Ingesting **1.2M+ songs** with 22 audio features from a cloud MongoDB Atlas cluster using the **Spark-MongoDB connector**
- Processing real **user behavior survey data** (demographics, listening habits, genre preferences)
- Engineering ML-ready features and performing **K-Means clustering with silhouette-score hyperparameter tuning**
- Surfacing insights via an **interactive Plotly dashboard**
- Delivering personalized song recommendations through a **hybrid engine** combining:
  - 📊 Content-based filtering (audio feature clusters)
  - 👥 Demographic-aware profiling (age group × time-of-day signals)
  - 🔍 Locality-Sensitive Hashing (LSH) for approximate nearest-neighbor song similarity

---

## 🏗 Architecture

```
MongoDB Atlas (Cloud)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Apache Spark (PySpark 3.4.1)                    │
│                                                                     │
│  ┌──────────────┐    ┌────────────────────────────────────────────┐ │
│  │  SongCatalog │    │              UserBehavior                  │ │
│  │  (1.2M rows) │    │              (Survey Data)                 │ │
│  └──────┬───────┘    └──────────────────┬─────────────────────────┘ │
│         │                               │                           │
│         ▼                               ▼                           │
│  ┌─────────────────┐         ┌────────────────────────┐            │
│  │ Feature Engineer│         │ Demographic Aggregation │            │
│  │ + Decade / Tempo│         │ Age × Genre → Cluster  │            │
│  └────────┬────────┘         └────────────┬───────────┘            │
│           │                               │                         │
│           ▼                               │                         │
│  ┌────────────────────┐                   │                         │
│  │ VectorAssembler    │                   │                         │
│  │ + StandardScaler   │                   │                         │
│  └────────┬───────────┘                   │                         │
│           │                               │                         │
│           ▼                               │                         │
│  ┌───────────────────────────────────┐    │                         │
│  │     K-Means Clustering            │    │                         │
│  │  k ∈ {5,6,7,8,10,12} tuned by    │    │                         │
│  │  Silhouette Score → best k=6      │    │                         │
│  └────────────┬──────────────────────┘    │                         │
│               │                           │                         │
│               ▼                           ▼                         │
│  ┌────────────────────────────────────────────────────┐            │
│  │         Hybrid Recommender Engine                  │            │
│  │  Mode 1: Vibe-Based  (Cluster + Tempo + Age-gate) │            │
│  │  Mode 2: Song-Based  (LSH Approximate k-NN)       │            │
│  └────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
Plotly Interactive Dashboard
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Distributed Computing** | Apache Spark (PySpark 3.4.1) |
| **Cloud Database** | MongoDB Atlas + `mongo-spark-connector_2.12:10.3.0` |
| **ML / Feature Engineering** | `pyspark.ml` — VectorAssembler, StandardScaler, KMeans, BucketedRandomProjectionLSH |
| **Clustering Evaluation** | ClusteringEvaluator (Silhouette Score), WSSSE (Inertia) |
| **Data Wrangling** | PySpark SQL Functions — `regexp_replace`, `to_date`, `when`, UDFs |
| **Visualization** | Plotly Express (dark theme, interactive) |
| **Environment** | Google Colab (cloud GPU/CPU runtime) |
| **Storage Format** | Parquet (columnar, Spark-native output) |

---

## 📂 Dataset

### 🎵 Song Catalog (`SpotifyAnalytics.SongCatalog`)

| Feature | Description |
|---|---|
| `name`, `artists`, `album` | Track metadata |
| `year`, `release_date` | Temporal metadata |
| `duration_ms` | Track length (converted to seconds) |
| `danceability` | How suitable for dancing (0.0–1.0) |
| `energy` | Intensity and activity (0.0–1.0) |
| `valence` | Musical positivity/happiness (0.0–1.0) |
| `acousticness` | Confidence of acoustic sound (0.0–1.0) |
| `instrumentalness` | Predicts absence of vocals (0.0–1.0) |
| `loudness` | Overall loudness in dB |
| `speechiness` | Presence of spoken words |
| `tempo` | Beats per minute (BPM) |
| `liveness`, `key`, `mode`, `time_signature` | Musical structure features |
| `explicit` | Content rating flag |

**Scale:** 1,200,000+ tracks spanning **1950 to present day**

### 👤 User Behavior (`SpotifyAnalytics.UserBehavior`)

Real survey data capturing:
- Demographics: `Age` (12-20 / 20-35 / 35-60), `Gender`
- Listening: `music_lis_frequency`, `music_time_slot`, `fav_music_genre`
- Podcast preferences: `pod_lis_frequency`, `pod_host_preference`, `preffered_pod_format`
- Platform: `spotify_subscription_plan`, `spotify_listening_device`, `spotify_usage_period`

---

## 🔄 Pipeline Phases

### Phase 1 — Data Ingestion

Connects to a **MongoDB Atlas cloud cluster** using the official Spark-MongoDB connector, loading both collections directly into Spark DataFrames — no intermediate CSV exports required:

```python
spark = SparkSession.builder \
    .appName("Spotify Data Processing") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
    .getOrCreate()

df_songs = spark.read.format("mongodb") \
    .option("connection.uri", uri) \
    .option("database", "SpotifyAnalytics") \
    .option("collection", "SongCatalog") \
    .load()
```

---

### Phase 2 — Feature Engineering & EDA

**Song Transformations:**
- `duration_ms → duration_sec` (human-readable conversion)
- `artists` field cleaned via `regexp_replace` (removes raw Python list syntax)
- `release_date` parsed with `to_date`
- `decade` column derived: `(year / 10).cast("int") * 10`
- `tempo_category` engineered: `Slow (<90 BPM) / Medium (90–120) / Fast (>120)`

**Aggregations for Dashboard:**

| Aggregation | Purpose |
|---|---|
| Music Trends by Year | Track evolution of energy, danceability, acousticness from 1950 onward |
| Target Demographics | Cross-tab of age group × favourite genre × user count |
| Top Workout Songs (2020s) | Decade=2020 ∩ Fast tempo, ordered by energy desc — useful as cold-start fallback |

---

### Phase 3 — K-Means Clustering with Hyperparameter Tuning

**Feature Vector:** `[danceability, energy, valence, acousticness, tempo]`

> ⚡ **Why StandardScaler?** `tempo` is measured in BPM (60–200), while all other features are normalized 0–1. Without scaling, K-Means distance calculations would be dominated entirely by tempo, rendering the other features irrelevant. `StandardScaler(withStd=True, withMean=True)` brings all features to zero-mean, unit-variance.

**Hyperparameter Search:**

| k | Silhouette Score | Inertia (WSSSE) |
|---|---|---|
| 5 | 0.3820 | 1,788,087.54 |
| **6** | **0.3860** ✅ | **1,614,248.83** |
| 7 | 0.3710 | 1,485,348.29 |
| 8 | 0.3492 | 1,401,055.12 |
| 10 | 0.3499 | 1,247,823.20 |
| 12 | 0.3371 | 1,183,377.36 |

**Optimal:** `k=6` with Silhouette Score = **0.3860**

> A silhouette score of ~0.39 is expected and acceptable for high-dimensional audio data. Musical features inherently overlap across genres (e.g., an acoustic pop track shares properties with both ambient and upbeat clusters). The clusters serve as broad *vibe profiles* rather than strict genre labels — well-suited for a recommendation use case.

---

### Phase 3.1 — Cluster Interpretation & Ethics

Each cluster was labeled by cross-referencing its **actual feature averages** against acoustic intuition:

| Cluster | Label | avg_energy | avg_dance | avg_acoustic | avg_positivity | Size |
|---|---|---|---|---|---|---|
| 0 | 🎸 Upbeat Rock & Alt-Pop | 0.79 | 0.43 | 0.11 | 0.52 | 113,067 |
| 1 | 🎶 Chill Acoustic Groove | 0.32 | 0.58 | 0.78 | 0.50 | 151,201 |
| 2 | 📚 Ambient & Studying | 0.25 | 0.40 | 0.82 | 0.30 | 108,712 |
| 3 | 🌙 Melancholy & Sleep | 0.14 | 0.27 | 0.88 | 0.13 | 159,697 |
| 4 | 🎉 Party & Workout | 0.69 | 0.68 | 0.21 | 0.72 | 208,620 |
| 5 | 🤘 Intense & Dark | 0.72 | 0.45 | 0.10 | 0.28 | 157,702 |

---

### Phase 3.2 — Hybrid Recommender System

A two-mode, demographics-aware recommender:

#### 🎯 Mode 1 — Vibe-Based Recommendations
1. Looks up the user's `(age_group, time_slot)` pair in a demographic cluster preference table derived from actual user survey data
2. Suggests a cluster; user can accept or override
3. Filters by chosen cluster + preferred tempo
4. **Age-gates explicit content** for the 12–20 group

#### 🔍 Mode 2 — Song-to-Song (LSH Approximate Nearest Neighbors)

Uses **BucketedRandomProjectionLSH** (`bucketLength=2.0, numHashTables=3`) to perform sub-linear approximate nearest-neighbor search across the full 1.2M-song catalog:

```python
brp = BucketedRandomProjectionLSH(
    inputCol="features", outputCol="hashes",
    bucketLength=2.0, numHashTables=3
)
neighbours = lsh_model.approxNearestNeighbors(hashed_songs, target["features"], 11)
```

**Quality Metrics Reported:**
- Average Euclidean feature distance (lower = better match)
- % of recommendations within the same cluster as the query song

---

## 📊 ML Results

### Demographic → Cluster Lookup (built from real user data)

| Age Group | Listening Time | Inferred Cluster |
|---|---|---|
| 12–20 | Afternoon | 📚 Cluster 2 — Ambient & Studying |
| 12–20 | Morning | 🎉 Cluster 4 — Party & Workout |
| 12–20 | Night | 🎉 Cluster 4 — Party & Workout |
| 20–35 | Afternoon | 🎶 Cluster 1 — Chill Acoustic Groove |
| 20–35 | Morning | 🎶 Cluster 1 — Chill Acoustic Groove |
| 20–35 | Night | 🎶 Cluster 1 — Chill Acoustic Groove |

12 unique age × time-slot combinations were profiled.

### Sample Recommendation Run

**Input:** Age=12-20 | Time=Afternoon | Tempo=Medium | Vibe=Ambient & Studying

| Song | Artist | Year |
|---|---|---|
| Falling Is Like This | Ani DiFranco | 1994 |
| Migration | The Innocence Mission | 2001 |
| Don't Let Me Cross Over | Sister Sadie | 2016 |
| Night is a Woman | John Gorka | 1991 |
| Rain Check | Ani DiFranco | 2004 |

**Found 4,221 matching tracks** in the Ambient & Studying × Medium tempo intersection.

---

## 📈 Visualizations

The Plotly dashboard renders four interactive charts (dark theme):

| Chart | What It Shows |
|---|---|
| **Audio Trends (Line)** | Evolution of avg. energy, danceability, acousticness from 1950 to present |
| **Demographic Bar Chart** | Top genres by age group — who listens to what |
| **Cluster Vibe Profiles** | Grouped bar chart of avg features across all 6 K-Means clusters |
| **Recommended Songs Fingerprint** | Audio profile (energy, acousticness, valence) of the top 5 recommended songs — visually proves the algorithm matched the target vibe |

---

## ⚖️ Ethical Considerations

This project includes an explicit bias audit embedded in the pipeline:

1. **Popularity Bias** — Dataset is skewed toward mainstream Western music; niche/regional genres are underrepresented.
2. **Demographic Bias** — User survey skews toward 20–35 age group, limiting generalization to other cohorts.
3. **Filter Bubble Risk** — Pure content-based systems trap users in acoustic echo chambers. The hybrid approach partially mitigates this by integrating behavioral signals.
4. **Age-Gating** — Explicit content is programmatically blocked for users in the 12–20 group at the recommender layer.

---

## 🚀 How to Run

### Prerequisites
- Google Colab account (free tier is sufficient)
- MongoDB Atlas cluster with `SpotifyAnalytics` database populated

### Steps

1. **Open in Google Colab** using the badge at the top of this README.

2. **Install PySpark** (first cell):

```bash
!pip install pyspark==3.4.1
```

> Note: You may see a dependency conflict with `dataproc-spark-connect`. This is non-blocking; PySpark 3.4.1 will still function correctly.

3. **Update the MongoDB URI** in the data ingestion cell:

```python
uri = "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/"
```

4. **Run all cells sequentially** (Runtime → Run All).

5. **Interact with the recommender** when prompted:
   - Enter age group: `12-20`, `20-35`, or `35-60`
   - Enter time slot: `Morning`, `Afternoon`, `Evening`, or `Night`
   - Enter tempo preference: `Slow`, `Medium`, or `Fast`
   - Choose Mode 1 (Vibe) or Mode 2 (Song similarity)

---

## 📁 Project Structure

```
spotify-bigdata/
│
├── SpotifyAnalytics.ipynb          # Main Colab notebook (all phases)
│
├── dashboard_data/
│   └── clustered_songs.parquet     # ML pipeline output (1.2M clustered tracks)
│
└── README.md                       # This file
```

---

## 👨‍💻 Author

Built as a Big Data Systems course project, demonstrating:
- Distributed data processing at scale with Apache Spark
- Cloud-native data ingestion from MongoDB Atlas
- Unsupervised ML with rigorous hyperparameter evaluation
- Ethical AI considerations embedded in the pipeline
- Real-time interactive recommendation serving

---

<div align="center">

*Built with ❤️ using PySpark · MongoDB Atlas · Plotly*

</div>
