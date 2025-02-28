from flask import Flask, request, jsonify 
import pickle 
import numpy as np 
import os 

app = Flask(__name__) 

MODEL_PATH = "models/best_xgb_model.pkl" 

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Best model not found. Run model_selection.py") 

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load() 
    model = model_data["model"] 

@app.route("/") 
def home():
    return jsonify(
        {
            "message" : "XGBoost model API is running" 
        }
    )

@app.route("/predict", methods = ["POST"]) 
def predict():
    try:
        data = request.get_json() 
        features = np.array(data["features"]).reshape(1, -1) 
        prediction = model.predict(features)
        return jsonify(
            {
                "Predictions" : float(1 if prediction[0] > 0.5 else 0) 
            }
        )
    except Exception as e:
        return jsonify(
            {
                'Error' : str(e)
            },
            400 
        )
    
if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000) 