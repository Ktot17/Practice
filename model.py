import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utils import DataColumns, ResultData


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = df.drop(DataColumns.defect_col_name, axis=1)
    y = df[DataColumns.defect_col_name]

    num_features = [DataColumns.temperature_col_name,
                    DataColumns.humidity_col_name,
                    DataColumns.pressure_col_name]
    cat_features = [DataColumns.brand_col_name,
                    DataColumns.operator_col_name]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(), cat_features),
        ],
        remainder='drop',
    )

    x_transformed = preprocessor.fit_transform(x)

    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
    feature_names = num_features + list(cat_feature_names)

    return pd.DataFrame(x_transformed, columns=feature_names), y


def train_model(x_data: pd.DataFrame, y_data: pd.Series) -> tuple[pd.DataFrame, pd.Series, list[float],
  CatBoostRegressor | RandomForestRegressor, dict, dict]:
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )

    if ((len(np.unique(y_data)) < 10 and x_data.shape[1] > 1000) or
            x_data.shape[1] > 50000):
        model = CatBoostRegressor(verbose=0)
    else:
        model = RandomForestRegressor(
            min_samples_leaf=2, max_features='sqrt', random_state=42
        )

    model.fit(x_train, y_train)

    y_pred, metrics, feature_importance = model_predict(model, x_test, y_test)

    return x_data, y_test, y_pred, model, metrics, feature_importance


def use_model(model: CatBoostRegressor | RandomForestRegressor,
              x_data: pd.DataFrame, y_data: pd.Series) -> (
        tuple[pd.DataFrame, pd.Series, list[float], dict, dict] | None):
    if hasattr(model, 'feature_names_'):
        expected_features = model.feature_names_
    elif hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_
    elif hasattr(model, 'get_feature_names_out'):
        expected_features = model.get_feature_names_out()
    else:
        return None

    x_data_processed = x_data.copy()
    extra_cols = set(x_data.columns) - set(expected_features)
    if extra_cols:
        x_data_processed = x_data_processed.drop(columns=list(extra_cols))

    missing_cols = set(expected_features) - set(x_data_processed.columns)
    if missing_cols:
        for col in missing_cols:
            x_data_processed[col] = np.nan

    x_data_processed = x_data_processed[expected_features]

    y_pred, metrics, feature_importance = model_predict(model, x_data_processed, y_data)

    return x_data_processed, y_data, y_pred, metrics, feature_importance


def model_predict(model: CatBoostRegressor | RandomForestRegressor,
                  x: pd.DataFrame,
                  y: pd.Series) -> tuple[list[float], dict, dict]:
    y_pred = model.predict(x)
    metrics = {
        ResultData.mae: mean_absolute_error(y, y_pred),
        ResultData.r2: r2_score(y, y_pred)
    }

    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        importance = model.get_feature_importance()

    feature_importance = dict(zip(
        [i for i in x.columns.values],
        importance.tolist()
    ))

    return y_pred, metrics, feature_importance
