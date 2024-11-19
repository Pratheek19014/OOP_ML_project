import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


class CarEvaluationClassifier:
    def __init__(self, DataFrame):
        self.models = {
            "RandomForest": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "SVC": SVC()
        }
        self.current_model = None
        self.encoder = LabelEncoder()

        # Encode all columns
        for column in DataFrame.columns:
            DataFrame[column] = DataFrame[column].astype(str)  # Convert values to strings
            DataFrame[column] = self.encoder.fit_transform(DataFrame[column])

        self.X = DataFrame.iloc[:, :-1]
        self.y = DataFrame.iloc[:, -1]

    def prepare_data(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )

    def set_model(self, model_name):
        if model_name in self.models:
            self.current_model = self.models[model_name]
            print(f"Model set to: {model_name}")
        else:
            raise ValueError(f"Invalid model name: {model_name}. Choose from {list(self.models.keys())}.")

    def update_hyperparameters(self, hyperparameters):
        """
        Update the hyperparameters of the current model.

        Parameters:
            hyperparameters (dict): A dictionary containing hyperparameters and their values.
        """
        if self.current_model is None:
            raise ValueError("No model selected. Use the 'set_model' method to select a model first.")

        model_class = type(self.current_model)
        try:
            self.current_model = model_class(**hyperparameters)
            print(f"Hyperparameters updated for {model_class.__name__}: {hyperparameters}")
        except TypeError as e:
            print(f"Error updating hyperparameters: {e}")

    def train(self):
        if self.current_model is None:
            raise ValueError("No model selected. Use the 'set_model' method to select a model first.")
        self.current_model.fit(self.X_train, self.y_train)
        print("Training complete.")

    def evaluate(self):
        if self.current_model is None:
            raise ValueError("No model selected. Use the 'set_model' method to select a model first.")
        y_pred = self.current_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred)
        print("Model Accuracy:", accuracy)
        print("\nClassification Report:\n", report)


# Load the data
data = pd.read_excel('car.xlsx', header=None)

# Create the classifier instance
classifier = CarEvaluationClassifier(data)

# Prepare the data
classifier.prepare_data()

# User selects a model
print("Available models: RandomForest, LogisticRegression, SVC")
selected_model = input("Enter the model name you want to use: ")
"""""
try:
    classifier.set_model(selected_model)
    classifier.train()
    classifier.evaluate()
except ValueError as e:
    print(e)
"""""
# Example Usage
if selected_model == "RandomForest":
    # Set Random Forest model
    classifier.set_model("RandomForest")

    # Update hyperparameters for Random Forest
    classifier.update_hyperparameters({
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    })

    # Train and evaluate Random Forest
    classifier.train()
    classifier.evaluate()
elif selected_model == "LogisticRegression":

    # Set Logistic Regression model
    classifier.set_model("LogisticRegression")

    # Update hyperparameters for Logistic Regression
    classifier.update_hyperparameters({
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 200
    })

    # Train and evaluate Logistic Regression
    classifier.train()
    classifier.evaluate()
else:
    # Set SVC model
    classifier.set_model("SVC")

    # Update hyperparameters for SVC
    classifier.update_hyperparameters({
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale"
    })

    # Train and evaluate SVC
    classifier.train()
    classifier.evaluate()
