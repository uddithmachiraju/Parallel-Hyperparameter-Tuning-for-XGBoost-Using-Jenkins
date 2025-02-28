import os 
import pickle 

def select_best_model(path):
    best_loss = float("inf") 
    file = "" 

    for model_file in os.listdir(path):
        with open(f"models/{model_file}", "rb") as f:
            data = pickle.load(f) 
            if data["loss"] < best_loss:
                best_loss = data["loss"] 
                file = model_file 

    os.rename(f"models/{file}", "models/best_xgb_model.pkl") 
    print(f"Best XGBoost model is {file} with loss as {best_loss}") 

if __name__ == "__main__":
    select_best_model(path = "models/")