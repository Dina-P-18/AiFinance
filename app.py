from flask import Flask, request, jsonify
from PIL import Image
import re
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.linear_model import LinearRegression
from datetime import datetime
import pytesseract
import numpy as np
import os
import json

# ---------------- Tesseract ----------------
# ❌ REMOVE local path (Render doesn't have D: drive)
# pytesseract.pytesseract.tesseract_cmd = r"D:\Pro Files\tesseract.exe"

# ---------------- Firebase ----------------
if not firebase_admin._apps:
    firebase_key = json.loads(os.environ["FIREBASE_KEY"])
    cred = credentials.Certificate(firebase_key)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ---------------- Flask ----------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Server running"

# ---------------- Helpers ----------------
def normalize_amount(text):
    if not text:
        return 0
    text = re.sub(r"[^\d.]", "", text)
    try:
        return float(text)
    except:
        return 0

def extract_amount(raw):
    matches = re.findall(r"\d+\.\d+|\d+", raw)
    return float(matches[-1]) if matches else 0

def extract_date(raw):
    match = re.search(r"\d{2}/\d{2}/\d{2,4}", raw)
    return match.group(0) if match else ""

def extract_merchant(raw):
    lines = raw.split("\n")
    for line in lines:
        if len(line.strip()) > 3:
            return line.strip().title()
    return "Unknown"

def guess_category(text):
    t = text.lower()
    rules = [
        ("grocery|mart|milk|veg", "Groceries"),
        ("hotel|restaurant|food|cafe", "Food"),
        ("petrol|diesel|fuel", "Transport"),
        ("electric|water|wifi", "Bills"),
        ("hospital|medical|pharmacy", "Health"),
        ("movie|cinema", "Entertainment"),
    ]
    for r, c in rules:
        if re.search(r, t):
            return c
    return "Other"

# ---------------- Extract API ----------------
@app.route("/extract", methods=["POST"])
def extract():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "file missing"}), 400

        img = Image.open(file).convert("RGB")
        img = img.resize((800, 800))  # safe size

        raw = pytesseract.image_to_string(img)

        amount = extract_amount(raw)
        date = extract_date(raw)
        merchant = extract_merchant(raw)
        category = guess_category(raw)

        return jsonify({
            "merchant": merchant,
            "amount": amount,
            "date": date,
            "category": category
        })

    except Exception as e:
        return jsonify({
            "merchant": "Unknown",
            "amount": 0,
            "date": "",
            "category": "Other",
            "error": str(e)
        }), 200

# ---------------- Predict API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.json or {}
        user_id = body.get("user_id")

        if not user_id:
            return jsonify({"error": "user_id required"}), 400

        docs = db.collection("users") \
            .document(user_id) \
            .collection("transactions") \
            .where("isExpense", "==", True) \
            .stream()

        rows = []

        for d in docs:
            x = d.to_dict()

            if "amount" not in x or "date" not in x or "categoryName" not in x:
                continue

            dt = x["date"]
            if not isinstance(dt, datetime):
                dt = dt.to_datetime()

            rows.append({
                "date": dt,
                "amount": float(x["amount"]),
                "category": x["categoryName"]
            })

        if not rows:
            return jsonify({"category_predictions": {}})

        df = pd.DataFrame(rows)
        df["month"] = df["date"].dt.to_period("M")

        predictions = {}

        for cat in df["category"].unique():
            cat_df = df[df["category"] == cat]

            monthly = (
                cat_df.groupby("month")["amount"]
                .sum()
                .sort_index()
                .reset_index()
            )

            monthly["month"] = monthly["month"].dt.to_timestamp()

            if len(monthly) < 3:
                predictions[cat] = round(float(monthly["amount"].mean()), 2)
                continue

            monthly["idx"] = range(len(monthly))

            X = monthly[["idx"]]
            y = monthly["amount"]

            lr = LinearRegression()
            lr.fit(X, y)

            pred = lr.predict([[monthly["idx"].max() + 1]])[0]
            predictions[cat] = round(max(pred, 0), 2)

        return jsonify({"category_predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 200

# ---------------- Main ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)