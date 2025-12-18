from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("data/shl_assessments.csv")

# Combine text fields for similarity
df["combined"] = df["skills"] + " " + df["job_role"]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["combined"])

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    user_text = " ".join(data.get("skills", [])) + " " + data.get("job_role", "")

    user_vec = vectorizer.transform([user_text])
    scores = cosine_similarity(user_vec, tfidf_matrix).flatten()

    df["score"] = scores
    top = df.sort_values("score", ascending=False).head(3)

    return jsonify({
        "recommended_assessments": top[[
            "assessment_name", "difficulty", "duration"
        ]].to_dict(orient="records")
    })

if __name__ == "__main__":
    app.run(debug=True)
