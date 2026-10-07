# Training a Playlist Curator Based on User Taste

**UE24CS352A – Machine Learning · Mini-Project · Problem No. 26**

| Team member | SRN |
|---|---|
| Rekha Dhorigol | PES1UG24CS370 |
| Sai Manogna B V | PES1UG24CS399 |

## 1. Problem statement

Given a set of playlists and a new, unclassified song, can we place the song in the best-matching playlist, or suggest that a **new playlist** be created?

We use the paper *"Training a Playlist Curator Based on User Taste"* (Awadelkarim & Coelho, Stanford, 2018) as the **baseline**, re-implement its methods ourselves, and then extend them with new features, more models, hyperparameter tuning, feature selection, ablation and segmentation studies, error analysis and an interactive app.

## 2. Dataset

- **Source:** Kaggle – [30000 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs) (`joebeachcapital/30000-spotify-songs`). The notebook downloads it automatically with `kagglehub`, so nothing needs to be committed to the repo.
- **Raw data:** 32,833 tracks, 23 columns, 449 playlists.
- **Our subset:** 13 distinct playlists (3,690 tracks) → balanced and de-duplicated by `track_id` → **676 songs, 52 per playlist**.
- **Target:** the playlist a song belongs to (13 classes; chance accuracy = 7.7 %).

| Feature set | Features |
|---|---|
| **A – paper baseline** | The paper's 10 Spotify audio features: danceability, energy, key, loudness, mode, speechiness, acousticness, liveness, valence, tempo |
| **B** | A + 3 engineered ratios (`energy_dance_ratio`, `acoustic_energy_ratio`, `tempo_energy`) |
| **C – final** | B + `instrumentalness`, `duration_ms`, `track_popularity`, `release_year` |

> The paper also used 2,095 one-hot genre tags. This dataset has no per-track genre tags, so our baseline uses audio features only. `playlist_genre` / `playlist_subgenre` are **not** used as features because they are derived from the playlist (label leakage).

## 3. Approach

1. **Baseline replication (paper):** Perceptron, Polynomial / Sigmoid / RBF SVM and a 1-hidden-layer NN, scaled vs. unscaled, 5-fold CV.
2. **Our additions:**
   - three feature sets (A / B / C) and extra models (Logistic Regression, KNN, Random Forest, HistGradientBoosting)
   - **hyperparameter tuning:** `GridSearchCV` (SVM) and `RandomizedSearchCV` (Random Forest, MLP, HistGB); the scaler itself is tuned (Standard vs Quantile)
   - **parameter tuning:** validation curve for SVM `C`; tuned abstain threshold
   - repeated stratified 5-fold CV (×3)
   - **feature selection:** `SelectKBest` (mutual information) inside the pipeline
   - soft-voting ensemble
   - **ablation study:** accuracy with and without `release_year` / `track_popularity`
   - **segmentation study (idea from the paper):** accuracy vs. the number of playlists (2, 3, 5, 8, 13)
   - error analysis (confusion matrix, PCA), permutation importance, per-playlist feature profiles
   - **"suggest a new playlist"** rule: if the top probability is below a tuned threshold, the app recommends creating a new playlist
   - validation on 2,422 songs from the same playlists that were never in the 676-song dataset
3. **No leakage:** scaling and feature selection happen *inside* sklearn `Pipeline`s; all tuning uses only the 80 % training split; the final model is chosen by cross-validation, not by the test set.

## 4. Key results

| Model | Features | CV acc. | Test acc. | Test macro-F1 | Test top-3 |
|---|---|---|---|---|---|
| Baseline RBF SVM (paper) | A | 0.294 | 0.301 | 0.286 | 0.632 |
| Baseline NN, 1 hidden layer (paper) | A | 0.310 | 0.301 | 0.282 | 0.574 |
| SVM (tuned) | C | 0.470 | 0.426 | 0.426 | 0.794 |
| MLP (tuned) | C | 0.459 | 0.434 | 0.428 | 0.706 |
| HistGradientBoosting (tuned) | C | 0.523 | 0.515 | 0.508 | 0.801 |
| **Random Forest (tuned) – final model** | C | **0.527** | **0.551** | **0.548** | **0.779** |

**Ablation** (tuned Random Forest, repeated CV on the training split):

| Feature set | Features | CV acc. | Test acc. |
|---|---|---|---|
| A – paper's 10 audio features | 10 | 0.358 | 0.324 |
| B – A + engineered ratios | 13 | 0.349 | 0.279 |
| C minus `release_year` and `track_popularity` | 15 | 0.377 | 0.368 |
| C minus `release_year` | 16 | 0.434 | 0.412 |
| **C – all features (final)** | 17 | **0.527** | **0.551** |

**Segmentation** (CV accuracy on random k-playlist subsets, 20 subsets per k; k = 13 is the full problem):

| Playlists (k) | Chance | SVM (tuned) | Random Forest (tuned) |
|---|---|---|---|
| 2 | 0.500 | 0.828 | 0.863 |
| 3 | 0.333 | 0.750 | 0.816 |
| 5 | 0.200 | 0.653 | 0.725 |
| 8 | 0.125 | 0.568 | 0.621 |
| 13 | 0.077 | 0.469 | 0.544 |

- **Out-of-set check:** on 2,422 unseen songs from the same 13 playlists the final model reaches 0.460 accuracy, 0.430 macro-F1 and 0.741 top-3 accuracy.
- **Abstain rule (threshold 0.30):** on the test set the model answers 71 % of songs with 63.5 % accuracy (55.1 % when it answers everything).
- **Most influential feature:** `release_year` (permutation importance 0.27, about 4× the next feature). Removing it lowers the tuned Random Forest from 0.527 to 0.434 CV accuracy, while audio-only features reach 0.358 – so metadata helps a lot, partly because these playlists are era-themed.
- **Fewer playlists is easier:** the Random Forest reaches 0.73 accuracy with 5 playlists and 0.86 with 2, compared with 0.54 for all 13 – closer to what a real user with a handful of playlists would see.
- **Hardest playlists:** *Charts 2020* vs *2020 Hits* and *Neo-Soul* vs *Urban Contemporary* overlap heavily in sound.
- Our baseline is far below the paper's 0.77–0.82 mainly because we have no genre tags and our playlists overlap more.
- The test set is small (136 songs, about ±4 points), so cross-validation scores are the more reliable comparison. The ablation uses hyperparameters tuned on the full feature set C, which slightly favours C.

## 5. Individual contributions

Both members wrote and ran code in the notebook and are responsible for explaining the sections below.

| | Rekha Dhorigol (PES1UG24CS370) | Sai Manogna B V (PES1UG24CS399) |
|---|---|---|
| **Notebook sections** | **7** Feature design (extra metadata features) and train/test split<br>**9** Hyperparameter tuning and SVM validation curve<br>**12** Model comparison, final-model selection and the ablation study<br>**14** Explainability (permutation importance, playlist profiles)<br>**15** Abstain-threshold tuning, final model, saving the model<br>**16** Prediction demo and unseen-song validation | **1–6** Imports, dataset loading, EDA, data cleaning and balancing, preprocessing, feature engineering<br>**8** Paper baseline replication and extra models × feature sets<br>**10** Repeated cross-validation<br>**11** Feature selection (mutual information)<br>**12** Segmentation experiment<br>**13** Error analysis (confusion matrix, PCA) |
| **Other** | Streamlit app, README, integration of the final notebook | Dataset curation, baseline tables and plots |
| **Shared** | Reading and summarising the Stanford paper, testing, project write-up, slides, GitHub repository, review presentation | |

## 6. Repository structure

```
.
├── Playlist_Curator_Main_final.ipynb   # full pipeline: data → baselines → tuning → evaluation → demo
├── README.md
├── docs/
│   ├── summary.pdf                     # project write-up
│   └── slides.pptx                     # review presentation
└── streamlit_app/
    ├── app.py                          # interactive demo
    ├── requirements.txt
    ├── playlist_curator_final.joblib   # saved final model (created in notebook Section 15)
    └── playlist_curator_processed.csv  # processed dataset (created at the end of Section 6)
```

## 7. How to run

### A. Notebook (recommended: Google Colab)

1. Open `Playlist_Curator_Main_final.ipynb` in Google Colab.
2. **Runtime → Run all.** The notebook installs `kagglehub` and downloads the dataset itself (Kaggle access may ask for a login the first time).
3. It creates `playlist_curator_processed.csv` (end of Section 6) and `playlist_curator_final.joblib` (Section 15) and downloads both to your computer automatically. Colab files disappear when the runtime resets, so keep these two files.

Running locally instead: Python 3.10+ and

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib kagglehub jupyter
jupyter notebook Playlist_Curator_Main_final.ipynb
```

(Remove the two `google.colab` `files.download` lines when running outside Colab.) The full notebook takes several minutes because of the hyperparameter searches, repeated CV and the segmentation study. The executed notebook in this repo keeps all outputs, so the results can be read without re-running.

### B. Streamlit app

1. Put `playlist_curator_final.joblib` and `playlist_curator_processed.csv` (from step A) next to `app.py` in `streamlit_app/`.
2. In `streamlit_app/requirements.txt`, set `scikit-learn` to the **same version as the Colab that created the model** (`import sklearn; print(sklearn.__version__)`), otherwise the model may fail to load.
3. Run:

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

Set the sliders for a song's audio features and metadata and click **Suggest playlist**. The app returns the top-3 playlists with probabilities, or recommends creating a new playlist when confidence is below the tuned threshold.

## 8. Limitations and future work

- Small dataset (676 songs, 13 classes) and a small test split; several playlists overlap heavily.
- `release_year` is a strong signal partly because of how these playlists were curated, so the metadata features help more than audio alone.
- No genre or artist tags (the paper's strongest features). Future work: artist and genre embeddings (node2vec / GloVe, as proposed in the paper), larger per-user playlists, and trying the segmentation study on real users' own playlists.

## 9. References

- A. Awadelkarim and K. Coelho, *Training a Playlist Curator Based on User Taste*, Stanford University, 2018. (Baseline for this project. The authors' public repos, `kevin-coelho/playlistr-ml-v1` and `playlistr-ml-py-v1`, were read for background only; all code in this repository is our own.)
- Kaggle: *30000 Spotify Songs*, `joebeachcapital/30000-spotify-songs`.
- scikit-learn, pandas, matplotlib, seaborn, Streamlit.
