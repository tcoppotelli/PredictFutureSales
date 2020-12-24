import xgboost as xgb

xgb_reg_1 = xgb.XGBRegressor(
    max_depth=10,
    n_estimators=20,
    min_child_weight=0.5,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.1,
    seed=42)
