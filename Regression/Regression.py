import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class RegressionModel:
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initializes the regression model with test size and random state.

        Parameters:
            test_size (float): Proportion of dataset to use as test set (default: 0.2)
            random_state (int): Seed for reproducibility (default: 42)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None

    def load_and_prepare_data(self, file_path, target_column):
        """
        Loads data from a CSV file, preprocesses non-numeric data, converts target column to numeric,
        and returns training and test data.

        Parameters:
            file_path (str): Path to the dataset (CSV format).
            target_column (str): Name of the target column.

        Returns:
            X_train, X_test, y_train, y_test: Split and scaled data.
        """
        data = pd.read_csv(file_path)

        # Drop or convert non-numeric columns
        data = self.preprocess_data(data, target_column)

        # Check if target column exists
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset.")

        # Convert target column to numeric, dropping rows where conversion fails
        data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
        data = data.dropna(subset=[target_column])  # Drop rows with NaN in the target column

        # Separate features and target variable
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Drop any rows with missing values in X or y
        X = X.dropna()
        y = y[X.index]  # Keep only the rows that remain in X

        # Check if the dataset is empty after preprocessing
        if X.empty or y.empty:
            raise ValueError(
                "The dataset is empty after preprocessing. Check for non-numeric data in the target column.")

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.test_size,
                                                            random_state=self.random_state)

        # Standardize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def preprocess_data(self, data, target_column):
        """
        Preprocesses the dataset by handling non-numeric columns.

        Parameters:
            data (pd.DataFrame): Input dataset.
            target_column (str): Name of the target column.

        Returns:
            pd.DataFrame: Processed dataset with only numeric columns.
        """
        # Drop non-numeric columns (e.g., Date/Time)
        non_numeric_columns = data.select_dtypes(include=['object']).columns
        data = data.drop(columns=non_numeric_columns.difference([target_column]))

        # Ensure target column is numeric, convert if necessary
        if data[target_column].dtype == 'object':
            data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
            data = data.dropna(subset=[target_column])

        return data

    def set_model(self, model):
        """
        Sets the regression model.

        Parameters:
            model: A scikit-learn regression model
        """
        self.model = model

    def train(self, X_train, y_train):
        """
        Trains the regression model on the training data.

        Parameters:
            X_train: Features for training.
            y_train: Target variable for training.
        """
        if self.model is None:
            raise ValueError("Model not set. Use the set_model() method to set a model before training.")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the trained model on the test data, returning performance metrics.

        Parameters:
            X_test: Features for testing.
            y_test: Target variable for testing.

        Returns:
            metrics (dict): Dictionary containing R² and Mean Squared Error.
        """
        predictions = self.model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)

        metrics = {
            'R²': r2,
            'Mean Squared Error': mse
        }
        return metrics

    def fit_and_evaluate(self, file_path, target_column, model):
        """
        Complete workflow: loads data, sets the model, trains it, and evaluates.

        Parameters:
            file_path (str): Path to the dataset (CSV format).
            target_column (str): Target column name for regression.
            model: A scikit-learn regression model

        Returns:
            metrics (dict): Dictionary of performance metrics.
        """
        X_train, X_test, y_train, y_test = self.load_and_prepare_data(file_path, target_column)
        self.set_model(model)
        self.train(X_train, y_train)
        metrics = self.evaluate(X_test, y_test)
        print(f"Model: {self.model.__class__.__name__}")
        print("Evaluation Metrics:", metrics)
        return metrics


# Usage example
if __name__ == "__main__":
    # Load the dataset
    file_path = "Weather Data_Small.csv"  # Path to uploaded dataset
    target_column = "Weather"  # Replace with actual target column name from the dataset

    # Instantiate the regression model class
    regression_model = RegressionModel()

    # Example with Linear Regression
    from sklearn.linear_model import LinearRegression

    lr_model = LinearRegression()
    regression_model.fit_and_evaluate(file_path, target_column, lr_model)
