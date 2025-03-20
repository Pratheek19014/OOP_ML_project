"""
Contains regression models alongwith hyperparameter tuning.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.model_selection import GridSearchCV

class RegressionModels:
    def __init__(self):
        self.model = None
        self.best_params = None

    def train_linear_regression(self, x_train, y_train):
        """
        Train a Linear Regression model.
        """
        self.model = LinearRegression()
        self.model.fit(x_train, y_train)
        return self.model
    
    def train_polynomial_regression(self, x_train, y_train, tune=False, param_grid=None, cv=3):
        """
        Train or tune a Polynomial Regression model.
        Parameters:
        - tune: If True, performs GridSearchCV to find the best degree.
        - param_grid: Grid of parameters for tuning (must be provided if tuning).
        - cv: Number of cross-validation folds.
        """
        if tune:
            if param_grid is None:
                raise ValueError(
                    "Parameter grid (param_grid) must be provided for hyperparameter tuning."
                )
            estimator = Pipeline([("polynomial_features", PolynomialFeatures()),("linear_regression", LinearRegression())])
            grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring="r2")
            grid_search.fit(x_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params_poly = grid_search.best_params_
        else:
            raise ValueError(
                "For training Polynomial Regression, provide degree in param_grid."
            )

        return self.model

    def train_ridge(self, x_train, y_train, tune=False, param_grid=None, cv=3):
        """
        Train or tune a Ridge Regression model.
        Parameters:
        - tune: If True, performs GridSearchCV to find the best parameters.
        - param_grid: Parameter grid for tuning (must be provided if tuning).
        - cv: Number of cross-validation folds.
        """
        if tune:
            if param_grid is None:
                raise ValueError(
                    "Parameter grid (param_grid) must be provided for hyperparameter tuning."
                )
                
            ridge_pipeline = Pipeline([
            ("polynomial_features", PolynomialFeatures()),
            ("ridge_regression", Ridge())
            ])
            grid_search = GridSearchCV(estimator=ridge_pipeline, param_grid=param_grid, cv=cv, scoring="r2")
            grid_search.fit(x_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params_ridge = grid_search.best_params_
            self.results_ridge = grid_search.cv_results_
            self.best_degree_ridge = self.best_params_ridge['polynomial_features__degree']
        else:
            raise ValueError(
                "For training Ridge Regression, provide alpha in param_grid."
            )

        return self.model

    def train_lasso(self, x_train, y_train, tune=False, param_grid=None, cv=3):
        """
        Train or tune a Lasso Regression model.
        Parameters:
        - tune: If True, performs GridSearchCV to find the best parameters.
        - param_grid: Parameter grid for tuning (must be provided if tuning).
        - cv: Number of cross-validation folds.
        """
        if tune:
            if param_grid is None:
                raise ValueError(
                    "Parameter grid (param_grid) must be provided for hyperparameter tuning."
                )
            
            lasso_pipeline = Pipeline([
            ("polynomial_features", PolynomialFeatures()),
            ("lasso_regression", Lasso())
            ])    
            
            grid_search = GridSearchCV(estimator=lasso_pipeline, param_grid=param_grid, cv=cv, scoring="r2")
            grid_search.fit(x_train, y_train)
            self.model = grid_search.best_estimator_
            self.best_params_lasso = grid_search.best_params_
            self.results_lasso = grid_search.cv_results_
            self.best_degree_lasso = self.best_params_lasso['polynomial_features__degree']
        else:
            raise ValueError(
                "For training Lasso Regression, provide alpha in param_grid."
            )

        return self.model