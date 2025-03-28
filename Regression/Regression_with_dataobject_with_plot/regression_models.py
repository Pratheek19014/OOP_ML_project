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

    def train_linear_regression(self, dataobj): # x_train, y_train
        """
        Train a Linear Regression model.
        """
        self.model = LinearRegression()
        self.model.fit(dataobj['split_data']['x_train'], dataobj['split_data']['y_train'])
        return self.model
    
    def train_polynomial_regression(self, dataobj, param_grid=None, cv=3): # x_train, y_train
       
        estimator = Pipeline([("polynomial_features", PolynomialFeatures()),("linear_regression", LinearRegression())])
        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring="r2")
        grid_search.fit(dataobj['split_data']['x_train'], dataobj['split_data']['y_train'])
        self.model = grid_search.best_estimator_
        self.best_params_poly = grid_search.best_params_

        return self.model

    def train_ridge(self, dataobj, param_grid=None, cv=3): # x_train, y_train
                
        ridge_pipeline = Pipeline([
        ("polynomial_features", PolynomialFeatures()),
        ("ridge_regression", Ridge())
        ])
        grid_search = GridSearchCV(estimator=ridge_pipeline, param_grid=param_grid, cv=cv, scoring="r2")
        grid_search.fit(dataobj['split_data']['x_train'], dataobj['split_data']['y_train'])
        self.model = grid_search.best_estimator_
        self.best_params_ridge = grid_search.best_params_
        self.results_ridge = grid_search.cv_results_
        self.best_degree_ridge = self.best_params_ridge['polynomial_features__degree']

        return self.model

    def train_lasso(self, dataobj, param_grid=None, cv=3): # x_train, y_train
            
        lasso_pipeline = Pipeline([
        ("polynomial_features", PolynomialFeatures()),
        ("lasso_regression", Lasso())
        ])    
        
        grid_search = GridSearchCV(estimator=lasso_pipeline, param_grid=param_grid, cv=cv, scoring="r2")
        grid_search.fit(dataobj['split_data']['x_train'], dataobj['split_data']['y_train'])
        self.model = grid_search.best_estimator_
        self.best_params_lasso = grid_search.best_params_
        self.results_lasso = grid_search.cv_results_
        self.best_degree_lasso = self.best_params_lasso['polynomial_features__degree']
        #else:
        #    raise ValueError(
        #        "For training Lasso Regression, provide alpha in param_grid."
        #    )

        return self.model