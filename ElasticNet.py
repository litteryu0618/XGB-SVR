from sklearn.linear_model import ElasticNet

# 超参数优化
def optimize_en(trial):
    params = {
        'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9)
    }
    model = ElasticNet(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return mean_squared_error(y_test, pred)

study = optuna.create_study(direction='minimize')
study.optimize(optimize_en, n_trials=30)
best_params = study.best_params

# 模型训练与评估
en_model = ElasticNet(**best_params)
en_model.fit(X_train, y_train)
y_pred = en_model.predict(X_test)

print("\nElasticNet 模型评估:")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")