# Reddit-Comment-Removal-Prediction

## Objective
Developed an ML-powered content moderation system to predict whether a Reddit comment is likely to be removed based on its textual features.

## Key Features
- Processes 200K+ Reddit comments with NLP techniques.
- Implements multiple classification models (Logistic Regression, XGBoost, Deep Learning).
- Optimizes models with hyperparameter tuning and overfitting mitigation.
- Deploy a real-time prediction demo using Gradio.

## Dataset
- **Source:** Reddit 200K dataset
- **Preprocessing:**
  - Removed stopwords, punctuation, URLs, emails, and numbers.
  - Applied TF-IDF vectorization (5,000 features).
  - Engineered custom features (comment length, word count, exclamation marks).

## Model Development & Optimization
- **Models:**
  - Logistic Regression (Baseline Model)
  - XGBoost (Optimized with GridSearchCV)
  - Deep Learning (TensorFlow with Dense Layers & Dropout)
- **Optimization:**
  - Hyperparameter tuning (max_depth, n_estimators, etc.)
  - Early stopping for deep learning to prevent overfitting.

- **Metrics Used:** Precision, Recall, Confusion Matrix, Classification Report

## Deployment & Accessibility
- **Demo:** Gradio web-based application for real-time predictions.
- **Model Saving:** Joblib serialization for XGBoost, TF-IDF vectorizer, and NLP preprocessor.
- **Hosting:** Google Colab with GPU acceleration for fast inference.

## Results & Impact
✅ Automates content moderation by predicting comment removal likelihood.
✅ Reduces manual effort in filtering harmful/irrelevant content.
✅ Scalable and interactive tool for social media platforms.

## Technical Stack
- **Programming Language:** Python
- **Libraries:** Scikit-learn, XGBoost, TensorFlow, Spacy, TF-IDF, NLP, GridSearchCV, Gradio
- **Tools:** Google Colab, Joblib

## Large Dataset Download  
The dataset is too large for GitHub. You can download it from the link below:  

🔗 [Download the dataset](https://drive.google.com/drive/folders/1_Ca1nzzuupRkUbfEb1y5el6RpMvxlZiv?usp=share_link)  
