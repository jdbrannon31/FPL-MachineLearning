import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_and_save_model():
    print("Loading historical training data...")
    history_df = pd.read_csv('fpl_historical_features.csv')

    # Filter out rows with missing targets or missing form
    trainable_df = history_df.dropna(subset=['target_next_week_points', '3_game_avg_points', 'opponent_strength']).copy()

    # Define the V3 features (excluding 'minutes')
    features = ['3_game_avg_points', 'home_game', 'opponent_strength']
    X = trainable_df[features]
    y = trainable_df['target_next_week_points']

    print("Training Random Forest V3...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the trained model to the local directory
    joblib.dump(model, 'fpl_model_v3.pkl')
    print("Model saved successfully as fpl_model_v3.pkl!")

if __name__ == "__main__":
    train_and_save_model()