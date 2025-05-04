import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Venue mapping 
venue_mapping = {
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",
    "Brabourne Stadium, Mumbai": "Brabourne Stadium",
    "Dr DY Patil Sports Academy, Mumbai": "Dr DY Patil Sports Academy",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "Dr YS Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    "Eden Gardens, Kolkata": "Eden Gardens",
    "Himachal Pradesh Cricket Association Stadium": "Himachal Pradesh Cricket Association Stadium",
    "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium",
    "Maharashtra Cricket Association Stadium, Pune": "Maharashtra Cricket Association Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium",
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",
    "Wankhede Stadium, Mumbai": "Wankhede Stadium"
}

# Load your matches data
matches = pd.read_csv('IPL data (2008-2024)/matches.csv')

# Standardize venue names
matches['venue_std'] = matches['venue'].replace(venue_mapping).str.strip()

# Get unique, sorted, standardized venues for dropdown
valid_venues = sorted(matches['venue_std'].dropna().unique())

# Load model bundle and teams
bundle = joblib.load('ipl_match_predictor.pkl')
model = bundle['model']
scaler = bundle['scaler']
target_encoder = bundle['target_encoder']
team_mapping = bundle['team_mapping']
feature_names = bundle.get('feature_names', [
    "team1", "team2", "venue", "toss_winner", "toss_decision", 
    "avg_runs_last_5", "head_to_head_win_pct", "venue_win_pct", 
    "toss_win_pct", "team1_recent_form", "team2_recent_form",
    "is_weekend", "team1_momentum", "team2_momentum", 
    "team1_player_impact", "team2_player_impact", 
    "team1_venue_performance", "team2_venue_performance"
])
valid_teams = sorted(team_mapping.keys())
allowed_decisions = ["Bat", "Bowl"]

@app.route('/')
def home():
    return render_template('index.html',
                           teams=valid_teams,
                           venues=valid_venues,
                           toss_decisions=allowed_decisions)

# --- Feature Calculation Functions (date-independent, unchanged) ---
# ... (your feature functions here, unchanged, but use 'venue_std' for venue lookups)

def avg_runs_last_5(team):
    team_matches = matches[(matches['team1'] == team) | (matches['team2'] == team)].head(5)
    if team_matches.empty:
        return 0.5
    # If you use deliveries, ensure correct path and logic
    return 0.5  # Placeholder if not using deliveries

def head_to_head_win_pct(team1, team2):
    h2h_matches = matches[((matches['team1'] == team1) & (matches['team2'] == team2)) | 
                        ((matches['team1'] == team2) & (matches['team2'] == team1))]
    if h2h_matches.empty:
        return 0.5
    return h2h_matches['winner'].eq(team1).mean()

def venue_win_pct(team, venue):
    venue_matches = matches[((matches['team1'] == team) | (matches['team2'] == team)) & 
                          (matches['venue_std'] == venue)]
    return venue_matches['winner'].eq(team).mean() if not venue_matches.empty else 0.5

def toss_win_pct(team):
    toss_matches = matches[matches['toss_winner'] == team]
    return toss_matches['winner'].eq(team).mean() if not toss_matches.empty else 0.5

def recent_form(team, n=10):
    team_matches = matches[(matches['team1'] == team) | (matches['team2'] == team)].head(n)
    if team_matches.empty:
        return 0.5
    weights = np.linspace(1, 2, len(team_matches))
    return np.average(team_matches['winner'] == team, weights=weights)

def momentum(team, n=15):
    team_matches = matches[(matches['team1'] == team) | (matches['team2'] == team)].head(n)
    if team_matches.empty:
        return 0.5
    weights = np.exp(np.linspace(0, 2, len(team_matches)))
    return np.average(team_matches['winner'] == team, weights=weights)

def player_impact(team, n=10):
    team_matches = matches[(matches['team1'] == team) | (matches['team2'] == team)].head(n)
    if team_matches.empty:
        return 0.5
    return (team_matches['player_of_match'].notna() & 
            (team_matches['winner'] == team)).mean()

def prepare_prediction_features(team1, team2, venue, toss_winner, toss_decision):
    features = {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "avg_runs_last_5": avg_runs_last_5(team1),
        "head_to_head_win_pct": head_to_head_win_pct(team1, team2),
        "venue_win_pct": venue_win_pct(team1, venue),
        "toss_win_pct": toss_win_pct(toss_winner),
        "team1_recent_form": recent_form(team1),
        "team2_recent_form": recent_form(team2),
        "is_weekend": 0,
        "team1_momentum": momentum(team1),
        "team2_momentum": momentum(team2),
        "team1_player_impact": player_impact(team1),
        "team2_player_impact": player_impact(team2),
        "team1_venue_performance": venue_win_pct(team1, venue),
        "team2_venue_performance": venue_win_pct(team2, venue)
    }
    input_df = pd.DataFrame([features])
    input_df[["team1", "team2", "venue", "toss_winner", "toss_decision"]] = \
        target_encoder.transform(input_df[["team1", "team2", "venue", "toss_winner", "toss_decision"]])
    input_df = input_df[feature_names]
    return scaler.transform(input_df)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        team1 = data['team1']
        team2 = data['team2']
        if team1 == team2:
            return render_template('result.html',
                                 predicted_team="Error: Select different teams!",
                                 confidence="0%")
        features = prepare_prediction_features(
            team1, team2,
            data['venue'],
            data['toss_winner'],
            data['toss_decision']
        )
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0][pred]
        confidence = f"{proba * 100:.2f}%"
        predicted_team = team1 if pred == 1 else team2
        return render_template('result.html',
                             predicted_team=predicted_team,
                             confidence=confidence)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
