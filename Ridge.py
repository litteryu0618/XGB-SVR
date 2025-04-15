from sklearn.linear_model import Ridge

# 超参数优化
def optimize_ridge(trial):
    params = {
        'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True)
    }
    model = Ridge(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return mean_squared_error(y_test, pred)

study = optuna.create_study(direction='minimize')
study.optimize(optimize_ridge, n_trials=30)
best_params = study.best_params

# 模型训练与评估
ridge_model = Ridge(**best_params)
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)

print("\nRidge 模型评估:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred)):.4f}")