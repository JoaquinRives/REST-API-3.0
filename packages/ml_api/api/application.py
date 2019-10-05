from flask import Flask, request
from flask import Blueprint, request, jsonify
import regression_model
import regression_model.predict as pred
import pandas as pd
from sklearn.externals import joblib
import pickle

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        return 'ok'

@app.route('/v1/predict/regression', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.get_json()
        
        result = pred.make_prediction(input_data=json_data)

        return str(result)