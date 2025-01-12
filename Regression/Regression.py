import pandas as pd
import numpy as np  # remove
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_predict  # remove
from sklearn.metrics import r2_score, mean_squared_error  # remove
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt  # remove


class regression:
    def load_data(self, target_variable):  # confirm with Architecture
        self.df = pd.read_csv("Data/Weather Data.csv")
        self.y = self.df[target_variable]
        self.x = self.df.drop(target_variable, axis=1)

    def split(self, t_size, r_state):  # remove
        x_tr, x_ts, y_tr, y_ts = train_test_split(
            self.x, self.y, test_size=t_size, random_state=r_state
        )
        self.x_train = x_tr
        self.x_test = x_ts
        self.y_train = y_tr
        self.y_test = y_ts

    def linear_regression(self):
        self.model = LinearRegression()
        self.model.fit(self.x_train, self.y_train)
        self.y_predict_linear = self.model.predict(self.x_test)

    def polynomial_regression(self, k_fold, degree):
        #        poly_feat = PolynomialFeatures()
        #        self.x_train_poly = poly_feat.fit_transform(self.x_train)
        #        self.x_test_poly = poly_feat.transform(self.x_test)

        estimator = Pipeline(
            [
                ("polynomial_features", PolynomialFeatures()),
                ("linear_regression", LinearRegression()),
            ]
        )

        params = {"polynomial_features__degree": degree}

        grid = GridSearchCV(estimator, params, cv=k_fold)
        #        grid.fit(self.x_train_poly,self.y_train)
        grid.fit(self.x_train, self.y_train)
        self.best_score_poly = grid.best_score_
        self.best_params_poly = grid.best_params_

        #        self.y_poly_predict = grid.predict(self.x_test_poly)
        self.y_predict_poly = grid.predict(self.x_test)

    def ridge_regression(self, alpha_values, k_fold):
        ridge_model = Ridge()

        param = {"alpha": alpha_values}

        grid = GridSearchCV(estimator=ridge_model, param_grid=param, cv=k_fold)

        grid.fit(self.x_train, self.y_train)

        self.best_alpha = grid.best_params_["alpha"]
        self.best_score_ridge = grid.best_score_
        self.best_ridge_model = grid.best_estimator_

        self.y_predict_ridge = self.best_ridge_model.predict(self.x_test)

    def lasso_regression(self, alpha_values, k_fold):
        lasso_model = Lasso()

        param = {"alpha": alpha_values}

        grid = GridSearchCV(estimator=lasso_model, param_grid=param, cv=k_fold)

        grid.fit(self.x_train, self.y_train)

        self.best_alpha = grid.best_params_["alpha"]
        self.best_score_lasso = grid.best_score_
        self.best_lasso_model = grid.best_estimator_

        self.y_predict_lasso = self.best_lasso_model.predict(self.x_test)

    def error_metric(self, pred_val, actual_val):
        r2score = r2_score(pred_val, actual_val)
        print(r2score)


# class data2gui(regression):
# Below code is only for demonstration purposes.

data = regression()
data.load_data("pH-Wert")
print(data.df)
data.split(0.3, 101)

data.linear_regression()

data.polynomial_regression(k_fold=5, degree=[2, 3, 4, 5])
print(data.best_score_poly, data.best_params_poly)
print(data.y_test.shape)

data.ridge_regression(alpha_values=[0.01, 0.1, 1], k_fold=5)
print(data.best_alpha)
print(data.best_score_ridge)

data.lasso_regression(alpha_values=[0.01, 0.1, 1], k_fold=5)
print(data.best_alpha)
print(data.best_score_lasso)

print("Linear Regression: ", end=" ")
data.error_metric(data.y_predict_linear, data.y_test)
print("poly: ", end=" ")
data.error_metric(data.y_predict_poly, data.y_test)
print("ridge", end=" ")
data.error_metric(data.y_predict_ridge, data.y_test)
print("lasso", end=" ")
data.error_metric(data.y_predict_lasso, data.y_test)
