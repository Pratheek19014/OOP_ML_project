import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def regression_plot(x, y, x_label, y_label, data, ax=None):
    if ax is None:
        ax = plt.gca()
    sns.regplot(x=x, y=y, data=data, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Linear Regression Fit')

def residual_plot(x, y, ax=None):
    if ax is None:
        ax = plt.gca()
    sns.residplot(x=x, y=y, scatter_kws={"s": 80}, ax=ax)
    ax.set_title('Residual Plot')

def polynomial_plot(x_scatter, y_scatter, y_poly, x_label, y_label):
    sns.scatterplot(x=x_scatter, y=y_scatter, color='blue', label='Actual Data')
    sns.lineplot(x=x_scatter, y=y_poly, color='red', label='Polynomial Regression Line')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('Polynomial Regression Fit')
    plt.legend()
    plt.show()

def ridge_plot(data):
    results = data.results_ridge
    best_degree_mask = (results['param_polynomial_features__degree'] == data.best_degree_ridge)
    alphas = results['param_ridge_regression__alpha'][best_degree_mask]
    mean_scores = results['mean_test_score'][best_degree_mask]

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=alphas, y=mean_scores, marker='o', label=f'Best Degree = {data.best_degree_ridge}')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Cross-Validation Score (R2 Score)')
    plt.title('Alpha vs Model Performance (Ridge Regression)')
    plt.legend()
    plt.show()

def lasso_plot(data):
    results = data.results_lasso
    best_degree_mask = (results['param_polynomial_features__degree'] == data.best_degree_lasso)
    alphas = results['param_lasso_regression__alpha'][best_degree_mask]
    mean_scores = results['mean_test_score'][best_degree_mask]

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=alphas, y=mean_scores, marker='o', label=f'Best Degree = {data.best_degree_lasso}')
    plt.xlabel('Alpha (Regularization Strength)')
    plt.ylabel('Cross-Validation Score (R2 Score)')
    plt.title('Alpha vs Model Performance (Lasso Regression)')
    plt.legend()
    plt.show()