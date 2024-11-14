#model_training.py
import numpy as np
from sklearn.model_selection import  KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Function to train the XGBoost model
def train_xgboost(X_train, y_train, X_test, y_test):
    xg_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    xg_model.fit(X_train, y_train)
    y_pred = xg_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"XGBoost - Mean Squared Error: {mse}, R² Score: {r2}")
    return xg_model, mse, r2

# Cross-validation with KFold
def cross_validate(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    r2_scores = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        xg_model = XGBRegressor(objective='reg:squarederror', random_state=42)
        xg_model.fit(X_train, y_train)
        y_pred = xg_model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        logger.info(f"Fold {fold} - Mean Squared Error: {mse}, R² Score: {r2}")
        mse_scores.append(mse)
        r2_scores.append(r2)

    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)
    logger.info(f"Average Mean Squared Error (MSE): {avg_mse}")
    logger.info(f"Average R² Score: {avg_r2}")
    return avg_mse, avg_r2
