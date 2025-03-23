from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

# Load the saved model and encoders
saved_data = joblib.load("ipl_match_predictor.pkl")
model = saved_data["model"]
ordinal_encoder = saved_data["ordinal_encoder"]

# Get the list of teams from the OrdinalEncoder
# Assuming 'team1' is the first column in the encoder
teams = ordinal_encoder.categories_[0]

# Initialize Flask app
app = Flask(__name__)

# Home page
@app.route("/")
def home():
    return render_template("index.html", teams=teams)

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

    # Encode the input data
    encoded_input = ordinal_encoder.transform([[team1, team2, venue, toss_winner, toss_decision]])

    # Prepare the input for prediction
    example_input = [
        [
            encoded_input[0][0],  # team1_encoded
            encoded_input[0][1],  # team2_encoded
            encoded_input[0][2],  # venue_encoded
            encoded_input[0][3],  # toss_winner_encoded
            encoded_input[0][4],  # toss_decision_encoded
            team1_avg_runs_last_5,
            team2_avg_runs_last_5,
        ]
    ]

    # Make a prediction
    prediction = model.predict(example_input)

    # Decode the predicted winner
    predicted_winner = ordinal_encoder.categories_[-1][prediction[0]]  # Assuming winner is the last column

    # Return the prediction
    return jsonify({"predicted_winner": predicted_winner})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)