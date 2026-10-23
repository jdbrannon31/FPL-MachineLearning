# ⚽ FPL Autonomous Machine Learning Pipeline

An automated, end-to-end machine learning pipeline that predicts Fantasy Premier League (FPL) player performance using historical statistical features and real-time natural language processing (NLP). 

This repository operates as a self-sustaining MLOps system. Every Friday, a CI/CD pipeline wakes up, ingests the latest Premier League data, processes injury reports using Google Gemini, runs a Random Forest regression model, and commits the updated target projections back to the repository.

## 🏗️ System Architecture

1. **Data Ingestion (`ingestion.py`)**: Fetches live data from the official FPL API, including player rosters, historical gameweek features, and upcoming fixture difficulty.
2. **NLP Injury Analysis (`nlp_news.py`)**: Utilizes the Google GenAI SDK (Gemini 3.8 Flash) to read unstructured injury updates (e.g., *"Hamstring injury - Expected back Oct 24"*) and output a strict mathematical health multiplier.
3. **Predictive Modeling (`model.py` & `predict.py`)**: A Random Forest Regressor trained on 3-game rolling averages, home/away status, and opponent strength to project expected points and calculate Value per Million (£) ROI.
4. **Automated CI/CD (`.github/workflows/fpl_pipeline.yml`)**: A GitHub Actions workflow scheduled via cron job to execute the entire pipeline on an Ubuntu virtual server and automatically commit the new `fpl_predictions.csv` weekly.

## 🛠️ Tech Stack
* **Language:** Python 3.10
* **Machine Learning:** Scikit-Learn (Random Forest), Joblib
* **Data Processing:** Pandas, NumPy
* **AI / NLP:** Google Gemini API (`google-genai`)
* **DevOps / CI/CD:** GitHub Actions, YAML

## 🚀 Local Setup

To run this pipeline locally for testing or evaluation:

1. Clone the repository:
   ```bash
   git clone https://github.com/jdbrannon31/FPL-MachineLearning.git
   cd FPL-MachineLearning/fpl_ai_backend
   ```
2. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```
3. Set your Gemini API Key as an environment variable:
   * **Windows:** `setx GEMINI_API_KEY "your_key_here"`
   * **Linux/Mac:** `export GEMINI_API_KEY="your_key_here"`
4. Run the prediction pipeline:
   ```bash
   python predict.py
   ```
5. Evaluate against actual weekend results:
   ```bash
   python evaluate.py
   ```

## 📊 Next Steps & Future Features
* **Advanced Metrics:** Integrating Expected Goals (xG) and Expected Assists (xA) to handle high-variance outliers (e.g., massive 15-point hauls).
* **Expected Minutes:** Building a dedicated feature to predict playing time and avoid "Surprise Start" anomalies.
* **Web Dashboard:** Deploying the prediction outputs to an interactive Streamlit frontend.
