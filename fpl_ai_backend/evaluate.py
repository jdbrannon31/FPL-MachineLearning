import pandas as pd
import requests

def evaluate_weekend_performance():
    print("Fetching actual weekend results from FPL API...")
    
    # 1. Pull the live FPL data
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = requests.get(url)
    data = response.json()
    
    # Extract the players and their actual points from the most recent weekend
    elements = data['elements']
    actuals_df = pd.DataFrame(elements)[['id', 'web_name', 'event_points']]
    actuals_df.rename(columns={'event_points': 'actual_points'}, inplace=True)
    
    # 2. Load your model's predictions
    try:
        predictions_df = pd.read_csv('fpl_predictions.csv')
    except FileNotFoundError:
        print("Error: fpl_predictions.csv not found. Run predict.py first to generate predictions.")
        return

    # 3. Merge the datasets on the player ID
    eval_df = pd.merge(predictions_df, actuals_df, left_on='id', right_on='id', how='inner')
    
    # 4. Calculate the error (Residuals)
    # Positive error = Player overperformed the model
    # Negative error = Player underperformed the model
    eval_df['prediction_error'] = eval_df['actual_points'] - eval_df['projected_points']
    
    # Clean up the display dataframe
    display_cols = ['web_name_x', 'projected_points', 'actual_points', 'prediction_error']
    eval_df = eval_df[display_cols].rename(columns={'web_name_x': 'player'})
    
    # Sort by the absolute error to find the biggest outliers
    eval_df['abs_error'] = eval_df['prediction_error'].abs()
    outliers = eval_df.sort_values(by='abs_error', ascending=False)
    
    print("\n--- BIGGEST WEEKEND OUTLIERS ---")
    print(outliers.drop(columns=['abs_error']).head(15).to_string(index=False))

if __name__ == "__main__":
    evaluate_weekend_performance()