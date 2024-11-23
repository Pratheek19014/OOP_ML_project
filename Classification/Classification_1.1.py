import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor


class CarEvaluationClassifier:
    def __init__(self, DataFrame):
        self.models = {
            "RandomForest": RandomForestClassifier(n_estimators=40),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "SVC": SVC(),
            "kNN": KNeighborsClassifier()
        }
        self.current_model = None
        self.encoder = LabelEncoder()
        ##self.scaler = StandardScaler()
        ##vif_data = pd.DataFrame()

        # Encode all columns
        for column in DataFrame.columns:
            DataFrame[column] = DataFrame[column].astype(str)  # Convert values to strings
            DataFrame[column] = self.encoder.fit_transform(DataFrame[column])

        self.data = DataFrame.iloc[:, :-1]
        self.target = DataFrame.iloc[:, -1]
        ##vif_data["feature"] = self.data.columns
        ##vif_data["VIF"] = [variance_inflation_factor(self.data.values, i) for i in range(self.data.shape[1])]

    def prepare_data(self, test_size=0.2):
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
            self.data, self.target, test_size=test_size, random_state=42
        )
        ##self.data_train_scaled = self.scaler.fit_transform(classifier.data_train)
        ##self.data_test_scaled = self.scaler.transform(classifier.data_test)

    def set_model(self, model_name):
        if model_name in self.models:
            self.current_model = self.models[model_name]
            print(f"Model set to: {model_name}")
        else:
            raise ValueError(f"Invalid model name: {model_name}. Choose from {list(self.models.keys())}.")

    def hyperparameter_tuning(self, param_grid):
        if self.current_model is None:
            raise ValueError("No model selected. Use the 'set_model' method to select a model first.")

        print("Starting hyperparameter tuning...")
        self.grid_search = GridSearchCV(estimator=self.current_model, param_grid=param_grid, cv=5, scoring='accuracy')
        self.grid_search.fit(self.data_train, self.target_train)

        # Update the model with the best parameters
        self.current_model = self.grid_search.best_estimator_
        print("Best Parameters found:", self.grid_search.best_params_)

    def train(self):
        if self.current_model is None:
            raise ValueError("No model selected. Use the 'set_model' method to select a model first.")
        self.current_model.fit(self.data_train, self.target_train)
        print("Training complete.")

    def evaluate(self):
        if self.current_model is None:
            raise ValueError("No model selected. Use the 'set_model' method to select a model first.")
        self.target_pred = self.current_model.predict(self.data_test)
        accuracy = accuracy_score(self.target_test, self.target_pred)
        report = classification_report(self.target_test, self.target_pred)
        print("Model Accuracy:", accuracy)
        print("\nClassification Report:\n", report)

    def display_confusion_matrix(self):
        cm = confusion_matrix(self.target_test, self.target_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.show()

    def calculate_mse(self):
        mse = mean_squared_error(self.target_test, self.target_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        return mse

# Load the data
filepath = "E:/TH koeln_AIT/Courses/Oop/Project/Classification & Regression/Oop_Project_ML/Data/car.xlsx"
data = pd.read_excel(filepath, header=None)

# Create the classifier instance
classifier = CarEvaluationClassifier(data)

# Prepare the data
classifier.prepare_data()

# User selects a model
print("Available models: RandomForest, LogisticRegression, SVC, kNN")
selected_model = input("Enter the model name you want to use: ")

# Example Usage
try:
    classifier.set_model(selected_model)
    if selected_model == "RandomForest":
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "random_state": [42]
        }
    elif selected_model == "LogisticRegression":
        param_grid = {
            "C": [1.0, 10.0, 100.0],
            "penalty": ["l2", "l1", "elasticnet"],
            "solver": ["lbfgs", "liblinear", "newton-cg"],
            "max_iter": [2000, 5000],
            "class_weight": [None, "balanced"]
        }
    elif selected_model == "SVC":
        param_grid = {
            "C": [0.1, 1.0, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
    elif selected_model == "kNN":
        param_grid = {
            'n_neighbors': range(1, 20),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2]
        }
    else:
        raise ValueError("Invalid model name entered.")

    # Perform hyperparameter tuning
    classifier.hyperparameter_tuning(param_grid)
    

    # Train and evaluate
    classifier.train()
    classifier.evaluate()
    classifier.display_confusion_matrix()
    classifier.calculate_mse()

except ValueError as e:
    print(e)