import os 
import xgboost as xgb
import argparse 
import pickle 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, accuracy_score 

class XG_Boost:
    def __init__(self, learning_rate, max_depth):
        self.learning_rate = learning_rate 
        self.max_depth = max_depth 
        self.data = load_breast_cancer() 
        self.model = xgb.XGBRegressor(
            learning_rate = self.learning_rate, 
            max_depth = self.max_depth, n_estimators = 100
        )
        
    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.data.data, self.data.target, test_size = 0.2, shuffle = True
        )
        return X_train, X_test, y_train, y_test
        
    def train_evaluate_model(self):
        # Training the model 
        X_train, X_test, y_train, y_test = self.split_data() 
        self.model.fit(X_train, y_train) 

        # Evaluate the model
        predictions = self.model.predict(X_test) 
        loss = mean_squared_error(y_test, predictions)
        predictions = (predictions >= 0.5).astype(int)  
        accuracy = accuracy_score(y_test, predictions) 
        print("Accuracy of the model:", accuracy) 

        # Ensure the 'models' directory exists
        os.makedirs("models", exist_ok=True)

        # Define the filename
        filename = f"models/xgb_lr_{self.learning_rate}_depth_{self.max_depth}.pkl"

        # Save the model
        with open(filename, "wb+") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "loss": loss,
                    "learning_rate": self.learning_rate,
                    "max_depth": self.max_depth,
                    "accuracy" : accuracy
                },
                f
            )

        print(f'Trained XGBoost with lr = {self.learning_rate} and depth = {self.max_depth}')

        return loss, filename 

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--lr", type = float, required = True, help = "Learning Rate")
    parser.add_argument("--depth", type = int, required = True, help = "max Depth of trees") 
    args = parser.parse_args() 
    print(args) 

    model = XG_Boost(args.lr, args.depth) 
    print(model.train_evaluate_model())