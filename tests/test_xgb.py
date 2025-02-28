import pytest
import os 
import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import XG_Boost 

@pytest.fixture
def xgb_model():
    """Fixture to create a XGB model"""
    return XG_Boost(learning_rate = 0.02, max_depth = 100)

def test_train_evaluate_model(xgb_model):
    _, filename = xgb_model.train_evaluate_model() 

    assert os.path.exists(filename) 