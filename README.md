# IPL Match Prediction Project

This project is a Flask-based web application that predicts the winner of an IPL (Indian Premier League) cricket match based on various input features such as teams, venue, toss winner, toss decision, and average runs scored by each team in their last 5 matches.

---

## 📚 Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Technologies Used](#technologies-used)  
4. [Setup Instructions](#setup-instructions)  
5. [How to Use](#how-to-use)  
6. [API Endpoints](#api-endpoints)  
7. [Future Improvements](#future-improvements)  
8. [License](#license)  
9. [Acknowledgments](#acknowledgments)  

---

## 📊 Project Overview
The IPL Match Prediction project uses a machine learning model to predict the winner of an IPL match. The model is trained on historical IPL match data and takes into account features such as:
- Teams playing (`team1` and `team2`)  
- Venue of the match  
- Toss winner and toss decision  
- Average runs scored by each team in their last 5 matches  

The Flask app provides a user-friendly interface to input match details and displays the predicted winner.

---

## ✨ Features
- **User-Friendly Interface:** A simple web interface to input match details.  
- **Machine Learning Model:** A pre-trained model to predict the match winner.  
- **Real-Time Prediction:** Instant prediction results based on user input.  
- **Dynamic Dropdowns:** Dropdowns for teams, venues, and toss decisions are dynamically populated.  

---

## 🛠️ Technologies Used
- **Python:** Primary programming language.  
- **Flask:** Web framework for building the application.  
- **Scikit-learn:** Machine learning library for training the model.  
- **Joblib:** For saving and loading the trained model and encoders.  
- **HTML/CSS:** For the front-end interface.  
- **NumPy:** For numerical computations.  

---

## ⚙️ Setup Instructions
Follow these steps to set up and run the project on your local machine.

### Prerequisites
- Python 3.7 or higher  
- Pip (Python package installer)  

### Steps
1. **Clone the Repository:**
```bash
git clone https://github.com/your-username/ipl-match-prediction.git
cd ipl-match-prediction
```

2. **Create a Virtual Environment:**
```bash
python -m venv venv
# On Linux/Mac
source venv/bin/activate  
# On Windows
venv\Scripts\activate
```

3. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the Flask App:**
```bash
python app.py
```

5. **Access the Application:**
Open your browser and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

---

## 🚀 How to Use
### 1. Home Page
- Open the application in your browser.  
- You will see a form with dropdowns for teams, venues, and toss decisions.  

### 2. Input Match Details
- Select `team1`, `team2`, `venue`, `toss_winner`, and `toss_decision` from the dropdowns.  
- Enter the average runs scored by `team1` and `team2` in their last 5 matches.  

### 3. Get Prediction
- Click the **"Predict"** button.  
- The predicted winner will be displayed on the screen.  

---

## 🔥 API Endpoints
### 1. Home Page
- **URL:** `/`  
- **Method:** `GET`  
- **Description:** Renders the home page with the input form.  

### 2. Prediction Endpoint
- **URL:** `/predict`  
- **Method:** `POST`  
- **Description:** Accepts form data and returns the predicted winner in JSON format.  

#### Request Body:
```json
{
  "team1": "Chennai Super Kings",
  "team2": "Mumbai Indians",
  "venue": "Wankhede Stadium",
  "toss_winner": "Chennai Super Kings",
  "toss_decision": "bat",
  "team1_avg_runs_last_5": 160.0,
  "team2_avg_runs_last_5": 155.0
}
```

#### Response:
```json
{
  "predicted_winner": "Chennai Super Kings"
}
```

---

## 📈 Future Improvements
- **Add More Features:** Include additional features like player statistics, weather conditions, and head-to-head records.  
- **Improve Model Accuracy:** Train the model on a larger and more recent dataset.  
- **Deploy to Cloud:** Deploy the application to a cloud platform like Heroku or AWS for public access.  
- **User Authentication:** Add user authentication to save prediction history.  

---

## 📜 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments
- **Dataset:** Kaggle IPL Dataset  
- **Flask Documentation:** [Flask Official Docs](https://flask.palletsprojects.com/)  
- **Scikit-learn Documentation:** [Scikit-learn Official Docs](https://scikit-learn.org/)  

