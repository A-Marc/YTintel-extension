from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import re
import pickle
import pandas as pd
import mlflow
import dagshub
import os
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi.responses import StreamingResponse
from io import BytesIO
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime


# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="YT Intel Sentiment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request Schemas
# ----------------------------

class CommentItem(BaseModel):
    text: str
    timestamp: str


class CommentRequest(BaseModel):
    comments: List[str]


class CommentWithTimestampRequest(BaseModel):
    comments: List[CommentItem]

class ChartRequest(BaseModel):
    sentiment_counts: dict


class WordCloudRequest(BaseModel):
    comments: List[str]


class TrendGraphItem(BaseModel):
    timestamp: str
    sentiment: int


class TrendGraphRequest(BaseModel):
    sentiment_data: List[TrendGraphItem]

# ----------------------------
# Globals
# ----------------------------
model = None
vectorizer = None


# ----------------------------
# Text Preprocessing
# ----------------------------
def preprocess_comment(comment: str) -> str:
    comment = comment.lower().strip()
    comment = re.sub(r"\n", " ", comment)
    comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

    stop_words = set(stopwords.words("english")) - {
        "not", "but", "however", "no", "yet"
    }

    comment = " ".join(
        word for word in comment.split() if word not in stop_words
    )

    lemmatizer = WordNetLemmatizer()
    comment = " ".join(
        lemmatizer.lemmatize(word) for word in comment.split()
    )

    return comment


# ----------------------------
# Startup Event
# ----------------------------
@app.on_event("startup")
def load_model():

    global model, vectorizer

    token = os.getenv("DAGSHUB_USER_TOKEN", "").strip()
    if token:
        print(f"📡 Found DAGSHUB_USER_TOKEN (cleaned length: {len(token)}, prefix: {token[:4]}...)")
        os.environ['MLFLOW_TRACKING_USERNAME'] = "prasu202324"
        os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    else:
        print("❌ DAGSHUB_USER_TOKEN not found in environment!")

    # mlflow.set_tracking_uri is already called below
    
    # dagshub.init(
    #     repo_owner="prasu202324",
    #     repo_name="YTintel-extension",
    #     mlflow=True
    # )

    mlflow.set_tracking_uri(
        "https://dagshub.com/prasu202324/YTintel-extension.mlflow/"
    )

    model = mlflow.pyfunc.load_model(
        "models:/yt_chrome_plugin_model/Production"
    )

    with open("./tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    print("✅ Model and vectorizer loaded successfully!")


# ----------------------------
# Routes
# ----------------------------

@app.get("/")
def home():
    return {"message": "Welcome to our FastAPI backend API"}


@app.post("/predict")
def predict(request: CommentRequest):

    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        cleaned = [preprocess_comment(c) for c in request.comments]

        X_sparse = vectorizer.transform(cleaned)

        # 🔥 Convert to DataFrame (required for MLflow signature)
        X_df = pd.DataFrame(
            X_sparse.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        preds = model.predict(X_df).tolist()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return [
        {"comment": c, "sentiment": str(p)}
        for c, p in zip(request.comments, preds)
    ]


@app.post("/predict_with_timestamps")
def predict_with_timestamps(request: CommentWithTimestampRequest):

    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        comments = [item.text for item in request.comments]
        timestamps = [item.timestamp for item in request.comments]

        preprocessed = [preprocess_comment(c) for c in comments]

        X_sparse = vectorizer.transform(preprocessed)

        # 🔥 Convert to DataFrame (CRITICAL FIX)
        X_df = pd.DataFrame(
            X_sparse.toarray(),
            columns=vectorizer.get_feature_names_out()
        )

        predictions = model.predict(X_df).tolist()
        predictions = [str(p) for p in predictions]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    return [
        {
            "comment": comment,
            "sentiment": sentiment,
            "timestamp": timestamp
        }
        for comment, sentiment, timestamp
        in zip(comments, predictions, timestamps)
    ]

@app.post("/generate_chart")
def generate_chart(request: ChartRequest):

    try:
        counts = request.sentiment_counts

        labels = ["Positive", "Neutral", "Negative"]
        values = [
            counts.get("1", 0),
            counts.get("0", 0),
            counts.get("-1", 0),
        ]

        plt.figure()
        plt.pie(values, labels=labels, autopct="%1.1f%%")
        plt.title("Sentiment Distribution")

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()

        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/generate_wordcloud")
def generate_wordcloud(request: WordCloudRequest):

    try:
        combined_text = " ".join(request.comments)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(combined_text)

        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()

        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_trend_graph")
def generate_trend_graph(request: TrendGraphRequest):

    try:
        # Convert to DataFrame
        data = pd.DataFrame([
            {
                "timestamp": datetime.fromisoformat(item.timestamp.replace("Z", "")),
                "sentiment": item.sentiment
            }
            for item in request.sentiment_data
        ])

        # Sort by time
        data = data.sort_values("timestamp")

        plt.figure()
        plt.plot(data["timestamp"], data["sentiment"])
        plt.title("Sentiment Trend Over Time")
        plt.xlabel("Time")
        plt.ylabel("Sentiment")
        plt.xticks(rotation=45)

        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        plt.close()

        return StreamingResponse(buffer, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))