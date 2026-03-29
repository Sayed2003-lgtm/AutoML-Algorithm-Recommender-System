# 🤖 Hybrid AutoML Algorithm Recommender

> Upload any CSV dataset → Get the best ML algorithm recommended instantly — no training required.

---

## 🚀 What it does

This system **automatically analyzes any dataset** and suggests the best machine learning algorithm using a **3-layer hybrid approach**:

| Layer | Method | Weight |
|-------|--------|--------|
| 🧠 Meta-Learning | Rule-based scoring from dataset characteristics | 60% |
| 📐 Similarity Search | Matches against 12+ reference dataset profiles | 40% |
| ⚖️ Smart Fusion | Weighted ensemble gives final ranked decision | — |

**Result**: A confident, data-driven algorithm recommendation — not guesswork.

---

## ✨ Features

- 📂 Upload any CSV dataset
- 🎯 Select your target column
- 🔍 Auto-detects Classification vs Regression
- 📊 Dataset meta-feature extraction (shape, missing %, imbalance, correlations)
- 🏆 Ranked list of algorithms with confidence scores
- 💡 Actionable insights about your dataset
- ⚡ No model training — instant results

---

## 🛠️ Tech Stack

- **Python** — core logic
- **Scikit-learn** — meta-feature extraction
- **Pandas / NumPy** — data processing
- **Streamlit** — interactive web UI

---

## ▶️ Run Locally

```bash
# 1. Clone / download the project
cd automl_recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
automl_recommender/
├── app.py            # Streamlit UI
├── recommender.py    # Core hybrid recommendation engine
├── requirements.txt  # Dependencies
└── README.md
```

---

## 🔮 Future Roadmap

- [ ] Full AutoML pipeline with hyperparameter tuning
- [ ] FAISS-based vector similarity search
- [ ] FastAPI backend for REST API
- [ ] Real-time model training & evaluation
- [ ] Export recommendations as PDF report

---

## 👤 Author

Built as a portfolio project demonstrating **meta-learning**, **similarity search**, and **hybrid AI decision systems**.
