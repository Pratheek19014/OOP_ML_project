from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import json
import requests

# Global variables
classifier_instance = None


class CarEvaluationClassifier:
    def __init__(self, DataFrame):
        self.models = {
            "RandomForest": RandomForestClassifier(),
            "SVC": SVC(),
            "kNN": KNeighborsClassifier()
        }
        self.current_model = None
        self.encoder = LabelEncoder()

        # Encode all columns
        for column in DataFrame.columns:
            DataFrame[column] = DataFrame[column].astype(str)
            DataFrame[column] = self.encoder.fit_transform(DataFrame[column])

        self.data = DataFrame.iloc[:, :-1]
        self.target = DataFrame.iloc[:, -1]

    def prepare_data(self, test_size=0.2):
        self.data_train, self.data_test, self.target_train, self.target_test = train_test_split(
            self.data, self.target, test_size=test_size, random_state=42
        )

    def set_model(self, model_name):
        if model_name in self.models:
            self.current_model = self.models[model_name]
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    def hyperparameter_tuning(self, param_grid):
        grid_search = GridSearchCV(estimator=self.current_model, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.data_train, self.target_train)
        self.current_model = grid_search.best_estimator_

    def train(self):
        if self.current_model is None:
            raise ValueError("No model selected. Use the 'set_model' method to select a model first.")
        self.current_model.fit(self.data_train, self.target_train)

    def evaluate(self):
        target_pred = self.current_model.predict(self.data_test)
        accuracy = accuracy_score(self.target_test, target_pred)
        mse = mean_squared_error(self.target_test, target_pred)
        cm = confusion_matrix(self.target_test, target_pred)
        return {
            "accuracy": accuracy,
            "mse": mse,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(self.target_test, target_pred, output_dict=True),
        }


@csrf_exempt
def upload_dataset(request):
    if request.method == "POST":
        url = request.POST.get("url")
        if not url:
            return JsonResponse({"error": "Dataset URL not provided"}, status=400)

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = pd.read_csv(pd.compat.StringIO(response.text))

            global classifier_instance
            classifier_instance = CarEvaluationClassifier(data)
            classifier_instance.prepare_data()

            return JsonResponse({"message": "Dataset loaded and split successfully."})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)


@csrf_exempt
def set_model(request):
    if request.method == "POST":
        if not classifier_instance:
            return JsonResponse({"error": "Dataset not loaded. Please upload the dataset first."}, status=400)

        data = json.loads(request.body)
        model_name = data.get("model_name")

        try:
            classifier_instance.set_model(model_name)
            return JsonResponse({"message": f"Model {model_name} selected successfully."})
        except ValueError as e:
            return JsonResponse({"error": str(e)}, status=400)


@csrf_exempt
def train_model(request):
    if request.method == "POST":
        if not classifier_instance or not classifier_instance.current_model:
            return JsonResponse({"error": "Model not set. Please select a model first."}, status=400)

        classifier_instance.train()
        return JsonResponse({"message": "Model trained successfully."})


@csrf_exempt
def evaluate_model(request):
    if request.method == "GET":
        if not classifier_instance or not classifier_instance.current_model:
            return JsonResponse({"error": "Model not trained. Please train the model first."}, status=400)

        results = classifier_instance.evaluate()
        return JsonResponse(results)
