import sys
import os 
import pytest 
import json 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from deployment.app import app 

# Automatically detect the test functions when named with test as prefix
@pytest.fixture
def client():
    """Creates a test client for Flask application""" 
    app.config["Testing"] = True 
    with app.test_client() as client:
        yield client 

def test_home(client):
    """Tests if the home route returns a success message""" 
    response = client.get("/") 
    assert response.status_code == 200 
    assert response.get_json() == {
        'message': 'XGBoost model API is running'
    }

def test_predict_valid_input(client):
    """Test prediction with valid input""" 
    sample_data = {
        "features" : [
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
            1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
            25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ]
    }
    response = client.post("/predict", data = json.dumps(sample_data), content_type = "application/json")

    assert response.status_code == 200 
    json_data = response.get_json() 
    assert "Predictions" in json_data
    # assert isinstance(json_data["Predictions"], float)

def test_predict_invalid_input(client):
    """Test prediction with invalid input"""
    sample_data = {
        "Wrong_key" : [
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
            1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
            25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ]
    }
    response = client.post("/predict", data = json.dumps(sample_data), content_type = "application/json") 
    assert response.status_code == 400 
    json_data = response.get_json() 
    assert 'Error' in json_data 