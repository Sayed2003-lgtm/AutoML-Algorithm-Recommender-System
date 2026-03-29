Live Demo

 Try it here:(https://automl-algorithm-recommender-system-ggysh2dpb5d3g2k29qslzu.streamlit.app/)

 Overview

This project is a Hybrid AutoML Algorithm Recommender System that intelligently suggests the best machine learning algorithm for any dataset — without training multiple models manually.

It combines:

 Dataset meta-feature analysis
 Rule-based meta-learning
 Dataset similarity matching
 Weighted hybrid decision system
 Key Features

 Upload any CSV dataset
 Automatic task detection (Classification / Regression)
 Smart algorithm recommendations
 Confidence score with ranking
 Explainable AI (why this algorithm?)
 Dataset insights & visualization

 How It Works
1️ Dataset Analysis
Extracts meta-features like:
Number of rows & features
Missing values %
Numerical vs categorical ratio
Class imbalance
Feature correlations
2️ Rule-Based Meta-Learning
Uses ML intuition:
Small dataset → Logistic Regression / KNN
Large dataset → Random Forest / XGBoost
Imbalanced data → XGBoost
High dimensions → SVM
3️ Similarity-Based Recommendation
Compares dataset with known dataset profiles
Uses similarity scoring to boost suitable algorithms
4️ Hybrid Decision System

Final score is computed using:

Final Score = 60% Meta-Learning + 40% Similarity
 Output Example
 Best Algorithm: Gradient Boosting (XGBoost)
 Confidence Score: 89%
 Full ranking of all algorithms
 Insights about dataset
 Tech Stack
Python
Pandas, NumPy
Scikit-learn
Streamlit
Matplotlib
