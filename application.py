from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 


application=Flask(__name__)
app=application

## import rigde regression and standard scaler pickle
try:
    ridge_model=pickle.load(open('ridge.pkl','rb'))
    standard_scaler=pickle.load(open('scaler.pkl','rb'))
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")


@app.route("/")
def index():
    return render_template('home.html', results=None, error=None)

@app.route("/predict_datapoint", methods=['POST'])
def predict_datapoint():
    # this endpoint only accepts POST; GET requests will return a 405 automatically
    try:
        temperature = float(request.form.get('Temperature'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        # "Classes" and "Region" are not used for prediction, but we read them to satisfy form
        classes = int(request.form.get('Classes'))
        region = int(request.form.get('Region'))
        
        # Prepare features for prediction
        features = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
        
        # Scale features
        scaled_features = standard_scaler.transform(features)
        
        # Make prediction
        prediction = ridge_model.predict(scaled_features)[0]
        
        return render_template('home.html', results=prediction, error=None)
    except Exception as e:
        return render_template('home.html', results=None, error=str(e))

if __name__=="__main__":
    print("Starting Flask app...")
    print("Access your app at: http://localhost:5000/")
    app.run(host="0.0.0.0", debug=True, use_reloader=False)
