# Human Cognitive Performance Analysis Dashboard

An interactive dashboard built with Streamlit for analyzing, visualizing, and predicting human cognitive performance based on lifestyle and physiological factors. It supports deep data exploration, insightful visualizations, and ML-based cognitive score predictions.

---

## 🧠 Features

- Explore and inspect the human cognitive performance dataset
- Visualize feature correlations and distributions
- Predict cognitive scores using multiple ML models
- Compare performance across regressors like XGBoost, CatBoost, RandomForest, etc.
- Interactive input for real-time predictions
- Summary of model metrics (MSE and R² Score)

---

## 📊 Dataset

The dataset used in this project is `human_cognitive_performance.csv` and contains the following features:

- Age  
- Gender  
- Exercise Frequency  
- Sleep Duration  
- Stress Level  
- Daily Screen Time  
- Caffeine Intake  
- Reaction Time  
- Memory Test Score  
- AI Predicted Score  
- Cognitive Score

---

## 🛠️ Tech Stack

| Tool       | Purpose                          |
|------------|----------------------------------|
| Python     | Core programming language        |
| Streamlit  | Interactive dashboard            |
| Pandas     | Data handling                    |
| Seaborn    | Data visualization               |
| Matplotlib | Plotting                         |
| scikit-learn | ML models & preprocessing      |
| Joblib     | Model serialization              |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/pr1yanka-dhage/Human-Cognitive-Performance.git
cd Human-Cognitive-Performance
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🧪 Model Details

This app includes the following machine learning regressors:

- XGBoost Regressor
- Random Forest Regressor
- CatBoost Regressor

All models are pre-trained and stored as `.pkl` files. You can choose between them on the dashboard for prediction.

---

## 📈 Dashboard Sections

### ▶️ Upload CSV File

- Upload your custom CSV with matching features to analyze new data.

### 🔍 Data Preview

- Displays the raw uploaded dataset as a table.

### 📌 Correlation Heatmap

- Shows a heatmap of correlation among variables.

### 📊 Feature Distributions

- Visualize distributions of each feature (e.g., age, stress level, memory score).

### 🧮 Cognitive Score Prediction

- Choose model → Input feature values → Predict score.
- Displays predicted score and compares it to the actual cognitive score if present.

### 📉 Model Evaluation

- Shows:
  - Mean Squared Error (MSE)
  - R² Score

---

## 📁 Project Structure

```
Human-Cognitive-Performance/
│
├── app.py                   # Main Streamlit dashboard script
├── models/
│   ├── xgb_model.pkl
│   ├── rf_model.pkl
│   └── catboost_model.pkl
├── human_cognitive_performance.csv
├── requirements.txt
└── README.md                # You're reading it!
```

---

## 📜 License

This project is under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🙌 Acknowledgments

Inspired by efforts in combining cognitive science with machine learning for performance prediction and enhancement.

**Made with ❤️ by [@pr1yanka-dhage](https://github.com/pr1yanka-dhage)**
