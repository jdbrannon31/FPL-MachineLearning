import requests
import pandas as pd
import numpy as np

def fetch_and_clean_data():
    print("1. Fetching static data from FPL API...")
    bootstrap_res = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/').json()
    elements_df = pd.DataFrame(bootstrap_res['elements'])
    teams_df = pd.DataFrame(bootstrap_res['teams'])
    
    # Mapping dicts
    team_mapping = dict(zip(teams_df['id'].astype(int), teams_df['name']))
    opponent_strength_map = dict(zip(teams_df['id'].astype(int), teams_df['strength']))
    player_team_map = dict(zip(elements_df['id'].astype(int), elements_df['team'].astype(int)))

    # Save cleaned roster
    roster_df = elements_df[['id', 'web_name', 'team', 'element_type', 'now_cost']].copy()
    roster_df['team_name'] = roster_df['team'].map(team_mapping)
    roster_df['price'] = roster_df['now_cost'] / 10
    roster_df.to_csv('fpl_cleaned_roster.csv', index=False)
    print("   -> Saved fpl_cleaned_roster.csv")

    # 2. Fetch Gameweek Histories for active players
    print("2. Fetching player gameweek histories (this may take 30-60 seconds)...")
    history_records = []
    
    for element_id in elements_df['id']:
        summary_url = f'https://fantasy.premierleague.com/api/element-summary/{element_id}/'
        res = requests.get(summary_url)
        if res.status_code == 200:
            gw_history = res.json().get('history', [])
            for row in gw_history:
                history_records.append({
                    'element': element_id,
                    'round': row['round'],
                    'total_points': row['total_points'],
                    'minutes': row['minutes'],
                    'was_home': int(row['was_home']),
                    'opponent_team': row['opponent_team']
                })

    history_df = pd.DataFrame(history_records)

    # 3. Feature Engineering: Rolling 3-game average & target shift
    print("3. Engineering features (rolling averages & target shifts)...")
    history_df.sort_values(by=['element', 'round'], inplace=True)
    
    # 3-game rolling average of points scored
    history_df['3_game_avg_points'] = (
        history_df.groupby('element')['total_points']
        .rolling(window=3, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Opponent strength mapping
    history_df['opponent_strength'] = history_df['opponent_team'].map(opponent_strength_map).fillna(3)
    history_df['home_game'] = history_df['was_home']

    # Target: points in the next gameweek
    history_df['target_next_week_points'] = (
        history_df.groupby('element')['total_points']
        .shift(-1)
    )

    # Save training-ready data
    history_df.to_csv('fpl_historical_features.csv', index=False)
    print("   -> Saved fpl_historical_features.csv")

    # 4. Fetch Next Fixtures Map
    print("4. Fetching upcoming fixture mappings...")
    fixtures_res = requests.get('https://fantasy.premierleague.com/api/fixtures/?future=1').json()
    valid_events = [f['event'] for f in fixtures_res if f.get('event') is not None]
    
    if valid_events:
        next_event = min(valid_events)
        next_fixtures = [f for f in fixtures_res if f.get('event') == next_event]
        
        fixture_rows = []
        for match in next_fixtures:
            fixture_rows.append({'team_id': int(match['team_h']), 'opponent_id': int(match['team_a']), 'home_game': 1})
            fixture_rows.append({'team_id': int(match['team_a']), 'opponent_id': int(match['team_h']), 'home_game': 0})
        
        fixtures_df = pd.DataFrame(fixture_rows)
        fixtures_df['opponent_name'] = fixtures_df['opponent_id'].map(team_mapping)
        fixtures_df['opponent_strength'] = fixtures_df['opponent_id'].map(opponent_strength_map)
        fixtures_df.to_csv('fpl_next_fixtures.csv', index=False)
        print("   -> Saved fpl_next_fixtures.csv")

    print("\nIngestion pipeline complete! All static datasets ready.")

if __name__ == "__main__":
    fetch_and_clean_data()