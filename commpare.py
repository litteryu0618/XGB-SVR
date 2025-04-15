import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 数据加载与预处理
data = pd.read_csv('F:/XGBoost-SVR/lhs_samples_1.csv')
X = data.drop(columns=['peak_strength'])
y = data['peak_strength']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==================================================================
# 单模型优化函数
# ==================================================================
def optimize_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return mean_squared_error(y_test, pred)

def optimize_svr(trial):
    params = {
        'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
        'epsilon': trial.suggest_float('epsilon', 1e-3, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True)
    }
    model = SVR(kernel='rbf', **params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return mean_squared_error(y_test, pred)

def optimize_mlp(trial):
    params = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(64,), (128,64)]),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2),
        'max_iter': trial.suggest_int('max_iter', 300, 1000)
    }
    model = MLPRegressor(**params, early_stopping=True)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return mean_squared_error(y_test, pred)

def optimize_en(trial):
    params = {
        'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9)
    }
    model = ElasticNet(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return mean_squared_error(y_test, pred)

def optimize_ridge(trial):
    params = {'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True)}
    model = Ridge(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return mean_squared_error(y_test, pred)

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

# ==================================================================
# 模型训练与预测
# ==================================================================
models = {
    'XGBoost': (optimize_xgb, XGBRegressor),
    'SVR': (optimize_svr, SVR),
    'MLP': (optimize_mlp, MLPRegressor),
    'ElasticNet': (optimize_en, ElasticNet),
    'Ridge': (optimize_ridge, Ridge),
    'RandomForest': (optimize_rf, RandomForestRegressor)
}

trained_models = {}
for name, (opt_func, model_class) in models.items():
    study = optuna.create_study(direction='minimize')
    study.optimize(opt_func, n_trials=30)
    best_params = study.best_params
    if name == 'SVR':
        trained_models[name] = model_class(kernel='rbf', **best_params).fit(X_train, y_train)
    else:
        trained_models[name] = model_class(**best_params).fit(X_train, y_train)

# 单模型预测结果
predictions = {name: model.predict(X_test) for name, model in trained_models.items()}

# ==================================================================
# 组合预测（仅XGBoost和SVR参与）
# ==================================================================
def residual_weighting(y_true, pred1, pred2, epsilon=1e-8):
    """改进的残差赋权函数"""
    # 转换为numpy数组并确保浮点类型
    y_true = np.asarray(y_true, dtype=np.float64)
    pred1 = np.asarray(pred1, dtype=np.float64)
    pred2 = np.asarray(pred2, dtype=np.float64)

    errors1 = (y_true - pred1) ** 2 + epsilon
    errors2 = (y_true - pred2) ** 2 + epsilon

    # 直接使用普通除法（已通过epsilon保证分母不为零）
    inv_errors1 = 1.0 / errors1
    inv_errors2 = 1.0 / errors2

    total_error = inv_errors1 + inv_errors2
    w1 = inv_errors1 / total_error
    w2 = inv_errors2 / total_error
    return w1, w2

def adaptive_weighting(y_true, pred1, pred2, window_size=6):
    weights = []
    n_samples = len(y_true)
    # 转换为numpy数组
    y_true = np.asarray(y_true)
    pred1 = np.asarray(pred1)
    pred2 = np.asarray(pred2)

    for i in range(n_samples):
        if i < window_size:
            w1, w2 = 0.5, 0.5
        else:
            start = i - window_size
            end = i
            prev_true = y_true[start:end]
            prev_pred1 = pred1[start:end]
            prev_pred2 = pred2[start:end]

            # 添加平滑项
            mse1 = mean_squared_error(prev_true, prev_pred1) + 1e-8
            mse2 = mean_squared_error(prev_true, prev_pred2) + 1e-8

            total_mse = mse1 + mse2
            w1 = mse2 / total_mse
            w2 = mse1 / total_mse
        weights.append([w1, w2])
    return np.array(weights)[:, 0], np.array(weights)[:, 1]


# 仅使用XGBoost和SVR进行组合
xgb_pred = predictions['XGBoost']
svr_pred = predictions['SVR']

# 等值权重
equal_weight_pred = (xgb_pred + svr_pred) / 2

# 残差权重
w1, w2 = residual_weighting(y_test.values, xgb_pred, svr_pred)
residual_weight_pred = w1 * xgb_pred + w2 * svr_pred

# 自适应权重
w1_adapt, w2_adapt = adaptive_weighting(y_test.values, xgb_pred, svr_pred)
adaptive_weight_pred = w1_adapt * xgb_pred + w2_adapt * svr_pred

# 合并所有预测结果
predictions.update({
    'Equal_Weight': equal_weight_pred,
    'Residual_Weight': residual_weight_pred,
    'Adaptive_Weight': adaptive_weight_pred
})

# ==================================================================
# 结果保存与评估
# ==================================================================
def save_prediction_results(y_true, predictions_dict, filename="F:/XGBoost-SVR/predictions_compare1.csv"):
    results_df = pd.DataFrame({'True_Value': y_true})
    for name, pred in predictions_dict.items():
        results_df[name] = pred
    results_df.to_csv(filename, index=False)
    print(f"预测结果已保存至 {filename}")
    return results_df

def evaluate_model(y_true, y_pred):
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        mape,
        r2_score(y_true, y_pred)
    )

# 保存结果
results_df = save_prediction_results(y_test, predictions)
print("\n前20个样本预测对比：")
print(results_df.head(20).round(4))

# 评估所有模型
print("\n模型评估结果：")
for name, pred in predictions.items():
    rmse, mae, mape, r2 = evaluate_model(y_test, pred)
    print(f"{name:15} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.1f}% | R²: {r2:.4f}")

# 特征重要性（仅XGBoost）
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': trained_models['XGBoost'].feature_importances_
}).sort_values('Importance', ascending=False)
print("\nXGBoost特征重要性：")
print(importance_df)