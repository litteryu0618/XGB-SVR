import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor

# 数据加载与预处理
data = pd.read_csv('F:/XGBoost-SVR/lhs_samples_3.csv')
X = data.drop(columns=['poisson_ratio'])
y = data['poisson_ratio']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 超参数优化
def optimize_mlp(trial):
    params = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(64,), (128, 64)]),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2),
        'max_iter': trial.suggest_int('max_iter', 200, 1000)
    }
    model = MLPRegressor(**params, early_stopping=True)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return mean_squared_error(y_test, pred)

study = optuna.create_study(direction='minimize')
study.optimize(optimize_mlp, n_trials=30)
best_params = study.best_params

# 模型训练与评估
mlp_model = MLPRegressor(**best_params)
mlp_model.fit(X_train, y_train)
y_pred = mlp_model.predict(X_test)

print("\nMLP 模型评估:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred)):.4f}")