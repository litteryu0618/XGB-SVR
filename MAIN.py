import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 数据加载与预处理
data = pd.read_csv('F:/XGBoost-SVR/lhs_samples_3.csv')
X = data.drop(columns=['poisson_ratio'])
y = data['poisson_ratio']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# XGBoost模型优化
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


study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(optimize_xgb, n_trials=50)
xgb_model = XGBRegressor(**study_xgb.best_params).fit(X_train, y_train)


# SVR模型优化
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


study_svr = optuna.create_study(direction='minimize')
study_svr.optimize(optimize_svr, n_trials=50)
svr_model = SVR(kernel='rbf', **study_svr.best_params).fit(X_train, y_train)

# 模型预测
xgb_pred = xgb_model.predict(X_test)
svr_pred = svr_model.predict(X_test)

# 组合预测方法
# 等值赋权
equal_weight_pred = (xgb_pred + svr_pred) / 2


# 残差赋权（修复类型错误）
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


# 传入numpy数组而非Series
w1, w2 = residual_weighting(y_test.values, xgb_pred, svr_pred)
residual_weight_pred = w1 * xgb_pred + w2 * svr_pred


# 自适应赋权
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


w1_adapt, w2_adapt = adaptive_weighting(y_test.values, xgb_pred, svr_pred)
adaptive_weight_pred = w1_adapt * xgb_pred + w2_adapt * svr_pred


# 结果保存功能
def save_prediction_results(y_true, predictions_dict, filename="F:/XGBoost-SVR/prediction_results3_notuse.csv"):
    """保存预测结果到CSV文件"""
    # 使用numpy数组避免类型问题
    results_df = pd.DataFrame({
        'True_Value': np.asarray(y_true),
        'Index': y_true.index
    })
    for name, pred in predictions_dict.items():
        results_df[name] = np.asarray(pred)
    results_df = results_df.sort_values('Index').drop(columns='Index')
    results_df.to_csv(filename, index=False)
    print(f"\n预测结果已保存到 {filename}")
    return results_df


predictions = {
    'XGBoost': xgb_pred,
    'SVR': svr_pred,
    'Equal_Weight': equal_weight_pred,
    'Residual_Weight': residual_weight_pred,
    'Adaptive_Weight': adaptive_weight_pred
}

results_df = save_prediction_results(y_test, predictions)

# 打印前10个样本结果
print("\n前10个样本预测对比：")
print(results_df.head(10).round(4))


# 模型评估（保持原样）
def evaluate_model(y_true, y_pred):
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mape = np.where(np.isinf(mape), 0, mape)
    return (
        np.sqrt(mean_squared_error(y_true, y_pred)),
        mean_absolute_error(y_true, y_pred),
        mape,
        r2_score(y_true, y_pred)
    )


models = {
    'XGBoost': xgb_pred,
    'SVR': svr_pred,
    'Equal Weight': equal_weight_pred,
    'Residual Weight': residual_weight_pred,
    'Adaptive Weight': adaptive_weight_pred
}

print("\n模型评估结果：")
for name, pred in models.items():
    rmse, mae, mape, r2 = evaluate_model(y_test, pred)
    print(f"{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²: {r2:.4f}\n")

# 特征重要性分析
feature_importance = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("XGBoost特征重要性排序：")
print(importance_df)