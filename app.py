from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

#loading model
model = pickle.load(open("model.pkl", "rb"))

#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods =['POST'])  #helps in prediction
def predict():
    features_data = request.form['feature']
    features_data_list = features_data.split(',')
    np_features = np.asarray(features_data_list, dtype=np.float32)
    prediction = model.predict(np_features.reshape(1, -1))

    output = ["cancerous" if prediction[0]==1 else "not cancerous"]
    return render_template('index.html', message = output)

#python main
if __name__ == "__main__":
    app.run(debug=True)
