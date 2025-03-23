from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

# Load the saved model, encoders, and label encoders
saved_data = joblib.load("ipl_match_predictor.pkl")
model = saved_data["model"]
ordinal_encoder = saved_data["ordinal_encoder"]
label_encoders = saved_data["label_encoders"]  # Load the label_encoders dictionary

# Retrieve the LabelEncoder for toss_decision
toss_decision_encoder = label_encoders["toss_decision"]

# Verify the order of columns in the OrdinalEncoder
for i, column_categories in enumerate(ordinal_encoder.categories_):
    print(f"Column {i}: {column_categories}")

# Get the list of teams and venues from the OrdinalEncoder
teams = ordinal_encoder.categories_[0]  # team1
venues = ordinal_encoder.categories_[2]  # venue

# Get the list of toss decisions from the LabelEncoder
toss_decisions = toss_decision_encoder.classes_

# Initialize Flask app
app = Flask(__name__)

# Home page
@app.route("/")
def home():
    return render_template("index.html", teams=teams, venues=venues, toss_decisions=toss_decisions)

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the form
    team1 = request.form["team1"]
    team2 = request.form["team2"]
    venue = request.form["venue"]
    toss_winner = request.form["toss_winner"]
    toss_decision = request.form["toss_decision"]
    team1_avg_runs_last_5 = float(request.form["team1_avg_runs_last_5"])
    team2_avg_runs_last_5 = float(request.form["team2_avg_runs_last_5"])

    # Encode the input data using the OrdinalEncoder
    encoded_input = ordinal_encoder.transform([[team1, team2, venue]])

    # Encode the toss_decision using the LabelEncoder
    toss_decision_encoded = toss_decision_encoder.transform([toss_decision])[0]

    # Prepare the input for prediction
    example_input = [
        [
            encoded_input[0][0],  # team1_encoded
            encoded_input[0][1],  # team2_encoded
            encoded_input[0][2],  # venue_encoded
            toss_decision_encoded,  # toss_decision_encoded
            team1_avg_runs_last_5,
            team2_avg_runs_last_5,
        ]
    ]

    # Make a prediction
    prediction = model.predict(example_input)

    # Decode the predicted winner
    predicted_winner = ordinal_encoder.categories_[0][prediction[0]]  # Use Column 0 (teams)

    # Return the prediction
    return jsonify({"predicted_winner": predicted_winner})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)