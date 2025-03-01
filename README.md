# Parallel Hyperparameter Tuning for XGBoost using Jenkins

This project automates **parallel hyperparameter tuning** for **XGBoost** using **Jenkins**.  
It trains multiple models with different hyperparameters **simultaneously**, selects the best model, and deploys it using **Flask & Docker**.

---

## 📂 Project Structure

Parallel_Hyperparameter_Tuning/ 
│── .jenkins/                      # Jenkins pipeline setup
│   ├── Jenkinsfile                # Jenkins CI/CD pipeline script
│── config/                        # Configuration files
│   ├── requirements.txt           # Required Dependencies 
│── src/                           # Core ML training & selection scripts
│   ├── model.py                   # Train XGBoost model
│   ├── model_selection.py         # Select the best model based on loss 
│── deployment/                    # Deployment files
│   ├── app.py                     # Flask API for serving predictions
│   ├── Dockerfile                 # Docker container setup
│── tests/                         # Unit tests for ML training & API
│   ├── test_app.py                # Tests XGBoost model training
│   ├── test_xgb.py                # Tests model selection logic
│── README.md                      # Project documentation
|── .gitignore                     # Git Ignore Files 

## ⚡ Features
**Parallel hyperparameter tuning** using Jenkins  
**XGBoost model training with different hyperparameters**  
**Automatic best model selection**  
**Flask API to serve predictions**  
**Dockerized deployment for easy scalability**