"""
Metrics class is used for evaluating model performance.
"""

from sklearn.metrics import r2_score


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return r2_score(y_test, y_pred),y_pred