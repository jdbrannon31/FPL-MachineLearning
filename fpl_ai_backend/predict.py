import pandas as pd
import joblib

def generate_predictions():
    print("Loading data and model...")
    model = joblib.load('fpl_model_v3.pkl')
    roster_df = pd.read_csv('fpl_cleaned_roster.csv')
    history_df = pd.read_csv('fpl_historical_features.csv')
    fixtures_df = pd.read_csv('fpl_next_fixtures.csv')

    print("Processing latest player form...")
    # Get the single most recent valid rolling average for each player
    latest_stats = (
        history_df.dropna(subset=['3_game_avg_points'])
        .sort_values(by=['element', 'round'])
        .groupby('element')
        .last()
        .reset_index()
    )
    
    # Merge roster with latest form
    future_df = pd.merge(roster_df, latest_stats[['element', '3_game_avg_points']], left_on='id', right_on='element', how='inner')
    
    # Merge with upcoming fixtures based on the player's team
    future_df = pd.merge(future_df, fixtures_df, left_on='team', right_on='team_id', how='inner')
    
    print("Running predictions...")
    # Predict using the exact 3 features the model was trained on
    X_predict = future_df[['3_game_avg_points', 'home_game', 'opponent_strength']]
    future_df['projected_points'] = model.predict(X_predict)
    
    # Calculate ROI
    future_df['value_per_million'] = future_df['projected_points'] / future_df['price']
    
    print("\n--- TOP BUDGET TARGETS (Under £7.5m) ---")
    affordable = future_df[future_df['price'] <= 7.5]
    top_targets = affordable.sort_values(by='value_per_million', ascending=False)
    
    display_cols = ['web_name', 'opponent_name', 'price', 'projected_points', 'value_per_million']
    print(top_targets[display_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    generate_predictions()