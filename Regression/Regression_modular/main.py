"""
Main class for regression.
Added features:
Loading and splitting data
Models
R2 score
Model selection via user input
To be added:
Exception handling
Unit testing
"""

from data_handler import DataHandler
from regression_models import RegressionModels
from metrics import evaluate_model
from visualization_trial import regression_plot, residual_plot, polynomial_plot, ridge_plot, lasso_plot
import matplotlib.pyplot as plt

def main():
    # Configuration
    file_path = "data\RegressionPredictionData.csv"  # Relative path
    target_variable = "pH-Wert"  # Replace with actual target column name
    test_size = 0.2

    # Initialize dataHandler class
    data_handler = DataHandler(file_path=file_path, target_variable=target_variable)

    # Load and split data
    x, y = data_handler.load_data()
    x_train, x_test, y_train, y_test = data_handler.split_data(test_size)

    # Initialize RegressionModels class
    regression_models = RegressionModels()
    
    while True:
        # Menu for model selection
        print("Select the regression method:")
        print("1. Linear Regression")
        print("2. Polynomial Regression")
        print("3. Ridge Regression")
        print("4. Lasso Regression")

        choice = int(input("Enter your choice (1/2/3/4): "))

        if choice == 1:
            # Linear Regression
            model = regression_models.train_linear_regression(x_train, y_train)
            r2, y_pred = evaluate_model(model, x_test, y_test)
            print(f"R2 Score (Linear Regression): {r2}")

            # Plot regression and residuals in one window
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            regression_plot(x_test["Datum"], y_test, "Datum", "pH-Wert", data_handler.data, ax=axs[0])
            residual_plot(y_test, y_pred, ax=axs[1])
            plt.tight_layout()
            plt.show()

        elif choice == 2:
            # Polynomial Regression
            param_grid = {'polynomial_features__degree': [2,3,4,5]}
            model = regression_models.train_polynomial_regression(x_train, y_train, tune=True, param_grid=param_grid)
            print(f"Best HyperParameter(Polynomial) ==> Degree: {regression_models.best_params_poly['polynomial_features__degree']}")
            r2, y_pred = evaluate_model(model, x_test, y_test)
            print(f"R2 Score (Polynomial Regression): {r2}")
            polynomial_plot(x_test["Datum"], y_test, y_pred, "Datum", "pH-Wert", regression_models.best_params_poly['polynomial_features__degree'])

        elif choice == 3:
            # Ridge Regression
            param_grid = {
                'polynomial_features__degree': [2, 3, 4, 5],
                'ridge_regression__alpha': [0.001, 0.01, 0.1, 1]
            }
            model = regression_models.train_ridge(x_train, y_train, tune=True, param_grid=param_grid)
            print("Best HyperParameters(Ridge) ==> alpha:", regression_models.best_params_ridge['ridge_regression__alpha'],
                ", Polynomial Degree:", regression_models.best_params_ridge['polynomial_features__degree'])
            r2, y_pred = evaluate_model(model, x_test, y_test)
            print(f"R2 Score (Ridge Regression): {r2}")
            ridge_plot(regression_models, )

        elif choice == 4:
            # Lasso Regression
            param_grid = {
                'polynomial_features__degree': [2, 3, 4, 5],
                'lasso_regression__alpha': [0.001, 0.01, 0.1, 1]
            }
            model = regression_models.train_lasso(x_train, y_train, tune=True, param_grid=param_grid)
            print("Best HyperParameters(Lasso) ==> alpha:", regression_models.best_params_lasso['lasso_regression__alpha'],
                ", Polynomial Degree:", regression_models.best_params_lasso['polynomial_features__degree'])
            r2, y_pred = evaluate_model(model, x_test, y_test)
            print(f"R2 Score (Lasso Regression): {r2}")
            lasso_plot(regression_models)

        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()