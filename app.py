import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Load the trained model and scaler
with open("fish_market_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extract features from request
    features = np.array([data['Species'], data['Length1'], data['Length2'], data['Length3'], data['Height'], data['Width']]).reshape(1, -1)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    
    return jsonify({'Predicted Weight': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
