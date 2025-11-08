from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Any
import os

# Optional transformers
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception as e:
    HF_AVAILABLE = False

app = FastAPI(title="CarbonSwap Recommendation API")

# CONFIG: weights
WEIGHTS = {
    "karbon_terserap": 0.25,
    "annual_survival_rate": 0.20,
    "luas_ha": 0.15,
    "pohon_tertanam": 0.10,
    "orang_terlibat": 0.08,
    "rating_ulasan": 0.07,
    "jumlah_ulasan": 0.05,
    "nlp_sentiment": 0.10
}

CSV_PATH = os.getenv("LOCATIONS_CSV", "location.csv")

# helper parsing functions
def parse_int_like(s):
    if pd.isna(s):
        return 0
    if isinstance(s, (int, float)) and not np.isnan(s):
        return int(s)
    s = str(s).strip()
    if s == "":
        return 0
    s = s.replace(",", "")
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*[kK].*$", s)
    if m:
        return int(float(m.group(1)) * 1000)
    m3 = re.search(r"([0-9]*\.?[0-9]+)", s)
    if m3:
        try:
            return int(float(m3.group(1)))
        except:
            return 0
    return 0

def parse_numeric_allow_k(s):
    if pd.isna(s):
        return 0.0
    if isinstance(s, (int, float)) and not np.isnan(s):
        return float(s)
    st = str(s).strip()
    if st == "":
        return 0.0
    for u in ["Kg", "kg", "CO2eq", "CO2", "mm"]:
        st = st.replace(u, "")
    if "%" in st:
        try:
            return float(st.replace("%", "").strip()) / 100.0
        except:
            pass
    m = re.match(r"^([0-9]*\.?[0-9]+)\s*[kK].*$", st)
    if m:
        return float(m.group(1)) * 1000.0
    st = st.replace(",", "")
    m2 = re.search(r"([0-9]*\.?[0-9]+)", st)
    if m2:
        try:
            return float(m2.group(1))
        except:
            return 0.0
    return 0.0

def safe_to_float(x):
    try:
        return float(x)
    except:
        return 0.0

# NLP sentiment (per-location aggregated)
SENT_MODEL_NAME = os.getenv("HF_SENT_MODEL", "indolem/indobert-base-uncased-sentiment")
sentiment_pipe = None
if HF_AVAILABLE:
    try:
        sentiment_pipe = pipeline("sentiment-analysis", model=SENT_MODEL_NAME)
    except Exception as e:
        # fallback
        sentiment_pipe = None

def avg_sentiment_for_texts(texts: List[str]) -> float:
    vals = []
    if sentiment_pipe is None:
        pos = ["baik","bagus","aktif","mudah","cocok","sukses","terawat","terjaga","support"]
        neg = ["mati","buruk","sulit","abrasi","rusak","terbengkalai","sampah","abrasi"]
        for t in texts:
            if not isinstance(t, str) or t.strip()=="":
                continue
            low = t.lower()
            score = 0.5
            for p in pos:
                if p in low:
                    score += 0.18
            for n in neg:
                if n in low:
                    score -= 0.18
            vals.append(max(0.0, min(1.0, score)))
    else:
        for t in texts:
            if not isinstance(t, str) or t.strip()=="":
                continue
            try:
                out = sentiment_pipe(t[:512])
                label = out[0]['label'].lower()
                sc = float(out[0]['score'])
                if 'pos' in label or 'positive' in label:
                    vals.append(sc)
                else:
                    vals.append(1.0 - sc)
            except:
                vals.append(0.5)
    if len(vals) == 0:
        return 0.5
    return float(sum(vals) / len(vals))

# core scoring function 
def compute_scores(df: pd.DataFrame, include_nlp: bool = True) -> pd.DataFrame:
    # cleaning numeric columns
    dfc = df.copy()
    dfc["luas_ha"] = dfc.get("luas_ha", 0).apply(parse_numeric_allow_k)
    dfc["orang_terlibat"] = dfc.get("orang_terlibat", 0).apply(parse_int_like)
    dfc["pohon_tertanam"] = dfc.get("pohon_tertanam", 0).apply(parse_int_like)
    dfc["karbon_terserap"] = dfc.get("karbon_terserap", 0).apply(parse_numeric_allow_k)
    dfc["annual_survival_rate"] = dfc.get("annual_survival_rate", 0).apply(parse_numeric_allow_k)
    dfc["annual_survival_rate"] = dfc["annual_survival_rate"].apply(lambda v: v/100.0 if v>1 else v)
    dfc["rating_ulasan"] = dfc.get("rating_ulasan", 0).apply(lambda x: safe_to_float(parse_numeric_allow_k(x)))
    dfc["jumlah_ulasan"] = dfc.get("jumlah_ulasan", 0).apply(parse_int_like)

    # NLP
    review_cols = [c for c in dfc.columns if c.lower().startswith("ulasan") or c.lower().startswith("ulasan_")]
    nlp_scores = []
    for idx, row in dfc.iterrows():
        texts = []
        for c in review_cols:
            v = row.get(c)
            if isinstance(v, str) and v.strip():
                texts.append(v.strip())
        nlp_scores.append(avg_sentiment_for_texts(texts))
    dfc["nlp_sentiment"] = nlp_scores

    feat_map = {
        "karbon_terserap": "karbon_terserap",
        "annual_survival_rate": "annual_survival_rate",
        "luas_ha": "luas_ha",
        "pohon_tertanam": "pohon_tertanam",
        "orang_terlibat": "orang_terlibat",
        "rating_ulasan": "rating_ulasan",
        "jumlah_ulasan": "jumlah_ulasan",
        "nlp_sentiment": "nlp_sentiment"
    }

    features = [feat_map[k] for k in WEIGHTS.keys() if feat_map[k] in dfc.columns]

    # fill missing numeric
    for f in features:
        dfc[f] = pd.to_numeric(dfc[f], errors='coerce').fillna(0.0)

    # normalize
    X = dfc[features].astype(float).values
    if X.shape[1] == 0:
        raise ValueError("No features available for scoring.")
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(Xs, columns=features, index=dfc.index)

    # weighted sum
    score = np.zeros(len(dfc))
    for key, w in WEIGHTS.items():
        col = feat_map.get(key)
        if col in df_scaled.columns:
            score += df_scaled[col].values * w

    dfc["skor_total"] = score
    dfc_sorted = dfc.sort_values("skor_total", ascending=False).reset_index(drop=True)
    return dfc_sorted

# endpoint(s)
class RecommendationItem(BaseModel):
    nama_lokasi: str
    jenis_bibit: str
    skor_total: float

@app.get("/recommendations", response_model=List[RecommendationItem])
def get_recommendations(top_k: int = 10, include_nlp: bool = True):
    if not os.path.exists(CSV_PATH):
        return []
    df = pd.read_csv(CSV_PATH)
    df_sorted = compute_scores(df, include_nlp=include_nlp)
    top = df_sorted.head(top_k)[["nama_lokasi", "jenis_bibit", "skor_total"]]
    top["skor_total"] = top["skor_total"].round(4)
    return top.to_dict(orient="records")

# endpoint to return full ranked table
@app.get("/ranked_full")
def get_full_ranked(include_nlp: bool = True):
    if not os.path.exists(CSV_PATH):
        return {"error": "no csv"}
    df = pd.read_csv(CSV_PATH)
    df_sorted = compute_scores(df, include_nlp=include_nlp)
    df_sorted["skor_total"] = df_sorted["skor_total"].round(4)
    return df_sorted.to_dict(orient="records")
