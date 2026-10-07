"""Playlist Curator - Streamlit demo.

Loads the model saved by section 15b of the notebook (playlist_curator_final.joblib) and
playlist_curator_processed.csv (saved in cell 31), takes a song's features, and returns the
top-3 playlists - or suggests creating a NEW playlist when confidence is below the tuned threshold.
Run:  streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Playlist Curator", page_icon="🎧")

ARTIFACT_PATH = Path("playlist_curator_final.joblib")
DATA_PATH = Path("playlist_curator_processed.csv")


@st.cache_resource
def load_artifact():
    return joblib.load(ARTIFACT_PATH)


@st.cache_data
def load_reference():
    ref = pd.read_csv(DATA_PATH)
    ref["release_year"] = pd.to_numeric(
        ref["track_album_release_date"].astype(str).str[:4], errors="coerce")
    return ref


if not ARTIFACT_PATH.exists() or not DATA_PATH.exists():
    st.error("Put `playlist_curator_final.joblib` and `playlist_curator_processed.csv` "
             "in the same folder as app.py (both are produced by the notebook).")
    st.stop()

art = load_artifact()
ref = load_reference()
needed = art["features"]
med = ref.median(numeric_only=True)


def build_features(row: dict) -> pd.DataFrame:
    d = pd.DataFrame([row])
    # same engineered features as the notebook (section 5)
    d["energy_dance_ratio"] = d["energy"] / (d["danceability"] + 1e-6)
    d["acoustic_energy_ratio"] = d["acousticness"] / (d["energy"] + 1e-6)
    d["tempo_energy"] = d["tempo"] * d["energy"]
    for c in needed:                                  # anything not supplied -> training median
        if c not in d:
            d[c] = med[c]
    return d[needed]


st.title("🎧 Playlist Curator")
st.caption("Which of the 13 playlists does this song belong to? Or does it need a new one?")

st.subheader("Audio features")
c1, c2 = st.columns(2)
with c1:
    danceability = st.slider("Danceability", 0.0, 1.0, float(med["danceability"]))
    energy = st.slider("Energy", 0.0, 1.0, float(med["energy"]))
    speechiness = st.slider("Speechiness", 0.0, 1.0, float(med["speechiness"]))
    acousticness = st.slider("Acousticness", 0.0, 1.0, float(med["acousticness"]))
    liveness = st.slider("Liveness", 0.0, 1.0, float(med["liveness"]))
with c2:
    valence = st.slider("Valence", 0.0, 1.0, float(med["valence"]))
    tempo = st.slider("Tempo (BPM)", 40.0, 220.0, float(med["tempo"]))
    loudness = st.slider("Loudness (dB)", -40.0, 0.0, float(max(med["loudness"], -40.0)))
    key = st.selectbox("Key (0=C ... 11=B)", list(range(12)), index=int(med["key"]))
    mode = st.selectbox("Mode (0=minor, 1=major)", [0, 1], index=int(med["mode"]))

row = dict(danceability=danceability, energy=energy, speechiness=speechiness,
           acousticness=acousticness, liveness=liveness, valence=valence,
           tempo=tempo, loudness=loudness, key=key, mode=mode)

extras = [f for f in ("instrumentalness", "duration_ms", "track_popularity", "release_year") if f in needed]
if extras:
    with st.expander("Track metadata (used by the final model)", expanded=True):
        if "release_year" in extras:
            row["release_year"] = st.slider("Release year", 1950, 2026, int(med["release_year"]))
        if "track_popularity" in extras:
            row["track_popularity"] = st.slider("Popularity (0-100)", 0, 100, int(med["track_popularity"]))
        if "instrumentalness" in extras:
            row["instrumentalness"] = st.slider("Instrumentalness", 0.0, 1.0, float(med["instrumentalness"]))
        if "duration_ms" in extras:
            mins = st.slider("Duration (minutes)", 0.5, 10.0, round(float(med["duration_ms"]) / 60000, 1))
            row["duration_ms"] = mins * 60000

if st.button("Suggest playlist", type="primary"):
    proba = art["model"].predict_proba(build_features(row))[0]
    top = proba.argsort()[::-1][:3]
    result = pd.DataFrame({"Playlist": [art["class_names"][i] for i in top],
                           "Probability": [float(proba[i]) for i in top]})

    if proba.max() < art["threshold"]:
        st.warning(f"Low confidence (top probability {proba.max():.2f} < threshold "
                   f"{art['threshold']:.2f}): **consider creating a NEW playlist** for this song.")
    else:
        st.success(f"Best match: **{result.iloc[0]['Playlist']}** ({result.iloc[0]['Probability']:.0%})")

    st.table(result.assign(Probability=result["Probability"].map("{:.1%}".format)))
    st.bar_chart(result.set_index("Playlist"))

st.divider()
st.caption(f"Model: {art.get('final_name', 'n/a')} trained on {len(ref)} songs from 13 playlists. "
           "Expect roughly 45-55% top-1 accuracy (about 75% top-3) - several playlists overlap heavily in sound.")
