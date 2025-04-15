from sklearn.ensemble import RandomForestRegressor

# 超参数优化
def optimize_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return mean_squared_error(y_test, pred)

study = optuna.create_study(direction='minimize')
study.optimize(optimize_rf, n_trials=30)
best_params = study.best_params

# 模型训练与评估
rf_model = RandomForestRegressor(**best_params)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("\nRandom Forest 模型评估:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred)):.4f}")