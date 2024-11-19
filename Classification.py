from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#Load the given dataset to a variable called df
cols = ['Buying', 'Maintenence', 'Doors', 'Persons', 'Lug_Boot', 'Safety']

# high=0: med=1: high=2: vhigh=3: more=5:
# big=0: med=1: small=2
# high=0: low=1: med=2
# acc=0: good=1: unacc=2: vgood=3
df = pd.read_excel('car.xlsx', header=None)#, header=None, names = cols
df.iloc[:, 2] = np.where(df.iloc[:, 2] == "more", 5, np.nan)
df.iloc[:, 3] = np.where(df.iloc[:, 3] == "more", 5, np.nan)
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
    #print(column)
#print(df.head())

X = df[df.columns[:-1]].values
Y = df[df.columns[-1]].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))


# new program
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class Classifier:
    def __init__(self, classifier=LogisticRegression(), test_size=0.2, random_state=42):
        """
        Initializes the classifier with the model, test size, and random state.

        Parameters:
            classifier: Any scikit-learn classifier (default: LogisticRegression).
            test_size (float): Proportion of the dataset for testing (default: 0.2).
            random_state (int): Seed for reproducibility (default: 42).
        """
        self.classifier = classifier
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None

    def load_and_prepare_data(self, file_path, target_column):
        """
        Loads and splits the data into training and testing sets.

        Parameters:
            file_path (str): Path to the CSV file containing the dataset.
            target_column (str): Name of the target column in the dataset.

        Returns:
            X_train, X_test, y_train, y_test: Split and scaled data.
        """
        data = pd.read_csv(file_path)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Trains the classifier on the training data."""
        self.model = self.classifier.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluates the model and returns accuracy and classification report."""
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        return accuracy, report

    def fit_and_evaluate(self, file_path, target_column):
        """
        Complete workflow: loads data, trains the model, and evaluates it.

        Parameters:
            file_path (str): Path to the dataset CSV file.
            target_column (str): Target column name for classification.
        """
        X_train, X_test, y_train, y_test = self.load_and_prepare_data(file_path, target_column)
        self.train(X_train, y_train)
        accuracy, report = self.evaluate(X_test, y_test)
        print(f"Model: {self.classifier.__class__.__name__}")
        print("Accuracy:", accuracy)
        print("Classification Report:\n", report)


# Example usage
if __name__ == "__main__":
    # Path to your dataset
    file_path = "your_dataset.csv"  # Replace with the path to your CSV file
    target_column = "target"  # Replace with the target column name in your dataset

    # Logistic Regression Classifier
    lr_classifier = Classifier(classifier=LogisticRegression())
    lr_classifier.fit_and_evaluate(file_path, target_column)

    # Random Forest Classifier
    rf_classifier = Classifier(classifier=RandomForestClassifier())
    rf_classifier.fit_and_evaluate(file_path, target_column)

    # Support Vector Machine Classifier
    svm_classifier = Classifier(classifier=SVC())
    svm_classifier.fit_and_evaluate(file_path, target_column)
