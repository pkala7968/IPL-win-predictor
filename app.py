from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

# ------------------------------------------
# Load saved model, encoders, and label encoders
# ------------------------------------------
try:
    saved_data = joblib.load("ipl_match_predictor.pkl")
    model = saved_data["model"]
    ordinal_encoder = saved_data["ordinal_encoder"]
    label_encoders = saved_data["label_encoders"]
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Retrieve the LabelEncoders
toss_winner_encoder = label_encoders["toss_winner"]
toss_decision_encoder = label_encoders["toss_decision"]

# Get categories from OrdinalEncoder
teams = ordinal_encoder.categories_[0]  # team1 and team2
venues = ordinal_encoder.categories_[2]  # venue
toss_decisions = toss_decision_encoder.classes_  # Toss decision options

# ------------------------------------------
# Initialize Flask app
# ------------------------------------------
app = Flask(__name__)

# ------------------------------------------
# Home page to render HTML form
# ------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", teams=teams, venues=venues, toss_decisions=toss_decisions)

# ------------------------------------------
# Prediction endpoint
# ------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the form
        team1 = request.form["team1"]
        team2 = request.form["team2"]
        venue = request.form["venue"]
        toss_winner = request.form["toss_winner"]
        toss_decision = request.form["toss_decision"]
        team1_avg_runs_last_5 = float(request.form["team1_avg_runs_last_5"])
        team2_avg_runs_last_5 = float(request.form["team2_avg_runs_last_5"])

        # ------------------------------------------
        # Encode team1, team2, and venue using OrdinalEncoder with column names
        # ------------------------------------------
        ordinal_feature_names = ["team1", "team2", "venue"]

        # Create a DataFrame for OrdinalEncoder with correct column names
        encoded_input_df = pd.DataFrame([[team1, team2, venue]], columns=ordinal_feature_names)

        # Apply OrdinalEncoder transformation
        encoded_input = ordinal_encoder.transform(encoded_input_df)

        # ------------------------------------------
        # Encode toss_winner and toss_decision using LabelEncoders
        # ------------------------------------------
        toss_winner_encoded = toss_winner_encoder.transform([toss_winner])[0]
        toss_decision_encoded = toss_decision_encoder.transform([toss_decision])[0]

        # ------------------------------------------
        # Prepare final input for prediction
        # ------------------------------------------
        example_input = [
            [
                encoded_input[0][0],  # team1_encoded
                encoded_input[0][1],  # team2_encoded
                encoded_input[0][2],  # venue_encoded
                toss_winner_encoded,  # toss_winner_encoded
                toss_decision_encoded,  # toss_decision_encoded
                team1_avg_runs_last_5,  # team1 average runs
            ]
        ]

        # ------------------------------------------
        # Fix: Convert input to DataFrame with correct feature names
        # ------------------------------------------
        feature_names = ["team1", "team2", "venue", "toss_winner", "toss_decision", "avg_runs_last_5"]
        example_input_df = pd.DataFrame(example_input, columns=feature_names)

        # ------------------------------------------
        # Make prediction using DataFrame
        # ------------------------------------------
        prediction = model.predict(example_input_df)

        # ------------------------------------------
        # Decode Prediction Properly
        # ------------------------------------------
        if prediction[0] < len(ordinal_encoder.categories_[0]):
            predicted_winner = ordinal_encoder.categories_[0][prediction[0]]
        else:
            predicted_winner = "Unknown"  # Handle unknown prediction gracefully

        # ------------------------------------------
        # Return Prediction
        # ------------------------------------------
        return jsonify({"predicted_winner": predicted_winner})

    except Exception as e:
        # Handle any errors during prediction
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


# ------------------------------------------
# Run Flask app
# ------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
