import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
import random
import seaborn as sns
import math
import warnings
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    make_scorer,
)
from joblib import dump, load
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, hp
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
import lightgbm as lgb
from datetime import datetime
from model import Model

ROOT = Path("/path_to_data")

class LGBM(Model):
    def __init__(self):
        super().__init__()

class XGBoost(Model):
    def __init__(self):
        super().__init__()

def main():
    warnings.filterwarnings("ignore")

    # Weekly data
    set_weekly_data()

    # XGBoost Hyperopt
    granularity = "daily"
    xgboost_hyperopt(granularity)
    model_evaluation(ROOT + f"/xgboost_hyperopt_{granularity}.joblib")

    # XGBoost Bayesian Optimization
    granularity = "weekly"
    xgboost_bayesian_optimization()

    # LightGBM Hyperopt
    granularity = "daily"
    lgbm_hyperopt(granularity)


def set_weekly_data():
    df = pd.read_csv(ROOT)

    df["ds"] = pd.to_datetime(df["ds"])

    df = df[
        (df["ds"] > datetime.strptime("2017-09-15", "%Y-%m-%d"))
        & (df["ds"] <= datetime.strptime("2018-02-15", "%Y-%m-%d"))
        | (df["ds"] > datetime.strptime("2018-09-15", "%Y-%m-%d"))
        & (df["ds"] <= datetime.strptime("2019-02-15", "%Y-%m-%d"))
        | (df["ds"] > datetime.strptime("2019-09-15", "%Y-%m-%d"))
        & (df["ds"] <= datetime.strptime("2020-02-15", "%Y-%m-%d"))
        | (df["ds"] > datetime.strptime("2020-09-15", "%Y-%m-%d"))
        & (df["ds"] <= datetime.strptime("2021-02-15", "%Y-%m-%d"))
        | (df["ds"] > datetime.strptime("2021-09-15", "%Y-%m-%d"))
        & (df["ds"] <= datetime.strptime("2022-02-15", "%Y-%m-%d"))
    ]

    df["date"] = pd.to_datetime(df["ds"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year

    df_copy = df.copy()
    df_copy = df_copy.shift(periods=90)

    df[
        [
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
            "x12",
            "x13",
            "x14",
            "x15",
            "x16",
        ]
    ] = df_copy[
        [
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
            "x12",
            "x13",
            "x14",
            "x15",
            "x16",
        ]
    ]

    del df["Unnamed: 0"]
    df.dropna(subset=["x1"], inplace=True)

    df.reset_index(drop="Index", inplace=True)

    # Group by week and aggregate the data
    df_weekly = df.groupby(pd.Grouper(key="date", freq="W")).agg(
        {
            "ds": "first",
            "date": "first",
            "day_of_week": "first",
            "day_of_year": "first",
            "month": "first",
            "quarter": "first",
            "year": "first",
            "x1": "mean",
            "x2": "mean",
            "x3": "mean",
            "x4": "mean",
            "x5": "mean",
            "x6": "mean",
            "x7": "mean",
            "x8": "mean",
            "x9": "mean",
            "x10": "mean",
            "x11": "mean",
            "x12": "mean",
            "x13": "mean",
            "x14": "mean",
            "x15": "mean",
            "x16": "mean",
            "y": "sum",  # Assuming you want to sum the 'y' values for each week
        }
    )

    cut_date = "2020-07-01"
    training_mask = df_weekly["date"] < cut_date
    training_data = df_weekly.loc[training_mask]
    training_dates = training_data["date"]
    print(training_data.shape)

    testing_mask = df_weekly["date"] >= cut_date
    testing_data = df_weekly.loc[testing_mask]
    print(testing_data.shape)

    # Plotting
    figure, ax = plt.subplots(figsize=(20, 5))
    training_data.plot(ax=ax, label="Training", x="date", y="y")
    testing_data.plot(ax=ax, label="Testing", x="date", y="y")
    plt.show()

    # Dropping unnecessary `date` column
    training_data = training_data.drop(columns=["date"])
    testing_dates = testing_data["date"]
    testing_data = testing_data.drop(columns=["date"])

    X_train = training_data[
        [
            "day_of_week",
            "day_of_year",
            "month",
            "quarter",
            "year",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
            "x12",
            "x13",
            "x14",
            "x15",
            "x16",
        ]
    ]
    y_train = training_data["y"]

    X_test = testing_data[
        [
            "day_of_week",
            "day_of_year",
            "month",
            "quarter",
            "year",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
            "x12",
            "x13",
            "x14",
            "x15",
            "x16",
        ]
    ]
    y_test = testing_data["y"]

    return X_train, X_test, y_train, y_test, training_dates, testing_dates


def set_daily_data():
    df = pd.read_csv(ROOT)

    df["ds"] = pd.to_datetime(df["ds"])

    df = df.sort_values("ds")

    df = df[
        (df["ds"] > datetime.strptime("2017-09-15", "%Y-%m-%d"))
        & (df["ds"] <= datetime.strptime("2018-02-15", "%Y-%m-%d"))
        | (df["ds"] > datetime.strptime("2018-09-15", "%Y-%m-%d"))
        & (df["ds"] <= datetime.strptime("2019-02-15", "%Y-%m-%d"))
        | (df["ds"] > datetime.strptime("2019-09-15", "%Y-%m-%d"))
        & (df["ds"] <= datetime.strptime("2020-02-15", "%Y-%m-%d"))
        | (df["ds"] > datetime.strptime("2020-09-15", "%Y-%m-%d"))
        & (df["ds"] <= datetime.strptime("2021-02-15", "%Y-%m-%d"))
        | (df["ds"] > datetime.strptime("2021-09-15", "%Y-%m-%d"))
        & (df["ds"] <= datetime.strptime("2022-02-15", "%Y-%m-%d"))
    ]

    df["date"] = pd.to_datetime(df["ds"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year

    df_copy = df.copy()
    df_copy = df_copy.shift(periods=90)

    df[
        [
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
            "x12",
            "x13",
            "x14",
            "x15",
            "x16",
        ]
    ] = df_copy[
        [
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
            "x12",
            "x13",
            "x14",
            "x15",
            "x16",
        ]
    ]

    df.dropna(subset=["x1"], inplace=True)

    df.reset_index(drop="Index", inplace=True)

    df.to_csv("lstm_shifted_data.csv")

    cut_date = "2020-07-01"

    training_mask = df["date"] < cut_date
    training_data = df.loc[training_mask]
    training_dates = training_data["date"]
    print(training_data.shape)

    testing_mask = df["date"] >= cut_date
    testing_data = df.loc[testing_mask]
    print(testing_data.shape)

    figure, ax = plt.subplots(figsize=(20, 5))
    training_data.plot(ax=ax, label="Training", x="date", y="y")
    testing_data.plot(ax=ax, label="Testing", x="date", y="y")
    plt.show()

    # Dropping unnecessary `date` column
    training_data = training_data.drop(columns=["date"])
    testing_dates = testing_data["date"]
    testing_data = testing_data.drop(columns=["date"])

    X_train = training_data[
        [
            "day_of_week",
            "day_of_year",
            "month",
            "quarter",
            "year",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
            "x12",
            "x13",
            "x14",
            "x15",
            "x16",
        ]
    ]
    y_train = training_data["y"]

    X_test = testing_data[
        [
            "day_of_week",
            "day_of_year",
            "month",
            "quarter",
            "year",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
            "x12",
            "x13",
            "x14",
            "x15",
            "x16",
        ]
    ]
    y_test = testing_data["y"]

    return X_train, X_test, y_train, y_test, training_dates, testing_dates


def model_evaluation(
    model_path, x_train, X_test, y_train, y_test, training_dates, testing_dates
):
    # Load
    estimator = load(model_path)
    print(estimator)
    # Evaluating GridSearch results
    prediction = estimator.predict(X_test)
    pd.DataFrame(prediction).to_csv("yhat_xgboost_test.csv")
    plot_predictions(testing_dates, y_test, prediction)
    evaluate_model(y_test, prediction)

    print()
    print(
        prediction.sum(),
        y_test.sum(),
        "\nError:",
        (prediction.sum() - y_test.sum()) / y_test.sum(),
    )
    print()
    # Plotting fit
    fitted_values = estimator.predict(X_train)
    pd.DataFrame(fitted_values).to_csv("yhat_xgboost_train.csv")
    plot_predictions(training_dates, y_train, fitted_values)
    evaluate_model(y_train, fitted_values)


def evaluate_model(y_test, prediction):
    print(f"MAE: {mean_absolute_error(y_test, prediction)}")
    print(f"MSE: {mean_squared_error(y_test, prediction)}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, prediction)}")


def plot_predictions(dates, y, prediction):
    df = pd.DataFrame({"date": dates, "actual": y, "prediction": prediction})
    figure, ax = plt.subplots(figsize=(10, 5))
    df.plot(ax=ax, label="Actual", x="date", y="actual")
    df.plot(ax=ax, label="Prediction", x="date", y="prediction")
    plt.legend(["Actual", "Prediction"])
    plt.show()


# FOR DL:
def mean_season_catch_scoring(y_true, y_pred):
    # print(y_true, y_pred)
    # print(type(y_true), type(y_pred))
    # print()

    season_errors = []
    for i in [17, 18, 19, 20]:
        indices = list(
            training_data[
                (training_data["ds"] > f"20{i}-09-14")
                & (training_data["ds"] < f"20{i+1}-02-16")
            ].index
        )
        season_y_true = []
        season_y_pred = []

        print("for")

        # print(i, indices)
        for index in indices:
            print(index)
            print(y_true[index], y_pred[index])
            # print()
            season_y_true.append(y_true[index])
            season_y_pred.append(y_pred[index])

        season_catch_true = sum(season_y_true)

        season_catch_pred = sum(season_y_pred)

        season_errors.append(season_catch_true**2 - season_catch_pred**2)

    return np.log(sum(season_errors) / len(season_errors))


from xgboost import XGBRegressor
from hyperopt import fmin, tpe, hp
from sklearn.model_selection import TimeSeriesSplit
from joblib import dump
from sklearn.model_selection import cross_val_score


def xgboost_hyperopt(granularity="daily"):
    if granularity == "daily":
        set_daily_data()
    elif granularity == "weekly":
        set_weekly_data()

    # XGBoost
    cv_split = TimeSeriesSplit(n_splits=5)

    # Define the objective function to minimize
    def objective(params):
        model = XGBRegressor(
            max_depth=int(params["max_depth"]),
            learning_rate=params["learning_rate"],
            n_estimators=int(params["n_estimators"]),
            colsample_bytree=params["colsample_bytree"],
        )
        score = cross_val_score(
            model, X_train, y_train, cv=cv_split, scoring="neg_root_mean_squared_error"
        ).mean()
        return -score  # Minimize negative RMSE

    # Define the search space for Bayesian Optimization
    space = {
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
        "n_estimators": hp.quniform("n_estimators", 100, 1000, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 0.7),
    }

    # Run Bayesian Optimization
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,  # Adjust the number of evaluations as needed
        verbose=1,
    )

    # Get the best hyperparameters
    best_max_depth = int(best["max_depth"])
    best_learning_rate = best["learning_rate"]
    best_n_estimators = int(best["n_estimators"])
    best_colsample_bytree = best["colsample_bytree"]

    # Create the best estimator
    best_estimator = XGBRegressor(
        max_depth=best_max_depth,
        learning_rate=best_learning_rate,
        n_estimators=best_n_estimators,
        colsample_bytree=best_colsample_bytree,
    )

    # Fit the best estimator on the full training set
    best_estimator.fit(X_train, y_train)

    # Save the best estimator
    dump(
        best_estimator,
        f"/content/drive/MyDrive/Tesis/xgboost_hyperopt_{granularity}.joblib",
    )


def xgboost_bayesian_optimization(granularity="daily"):
    if granularity == "daily":
        set_daily_data()
    elif granularity == "weekly":
        set_weekly_data()

    # XGBoost
    cv_split = TimeSeriesSplit(n_splits=5)

    # Define the search space for Bayesian Optimization
    search_space = {
        "max_depth": (3, 10),
        "learning_rate": (0.01, 0.3),
        "n_estimators": (100, 1000),
        "colsample_bytree": (0.3, 0.7),
    }

    # BayesSearchCV for hyperparameter tuning
    bayes_search = BayesSearchCV(
        XGBRegressor(),
        search_space,
        n_iter=400,  # Adjust the number of iterations as needed
        cv=cv_split,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )

    # Fit the model
    bayes_search.fit(X_train, y_train)

    # Get the best estimator
    best_estimator = bayes_search.best_estimator_

    # Save the best estimator
    dump(
        best_estimator,
        f"/content/drive/MyDrive/Tesis/xgboost_bayesian_{granularity}.joblib",
    )


def lgbm_hyperopt(granularity):
    # LGBM
    cv_split = TimeSeriesSplit(n_splits=4)

    # Define the objective function to minimize
    def objective_lgbm(params):
        model = lgb.LGBMRegressor(
            max_depth=int(params["max_depth"]),
            num_leaves=int(params["num_leaves"]),
            learning_rate=params["learning_rate"],
            n_estimators=int(params["n_estimators"]),
            colsample_bytree=params["colsample_bytree"],
        )
        score = cross_val_score(
            model, X_train, y_train, cv=cv_split, scoring="neg_root_mean_squared_error"
        ).mean()
        return -score  # Minimize negative RMSE

    # Define the search space for Bayesian Optimization
    space_lgbm = {
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "num_leaves": hp.quniform("num_leaves", 10, 120, 1),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
        "n_estimators": hp.quniform("n_estimators", 50, 1000, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 1),
    }

    # Run Bayesian Optimization
    best_lgbm = fmin(
        fn=objective_lgbm, space=space_lgbm, algo=tpe.suggest, max_evals=10
    )  # Adjust the number of evaluations as needed)

    # Get the best hyperparameters
    best_max_depth_lgbm = int(best_lgbm["max_depth"])
    best_num_leaves_lgbm = int(best_lgbm["num_leaves"])
    best_learning_rate_lgbm = best_lgbm["learning_rate"]
    best_n_estimators_lgbm = int(best_lgbm["n_estimators"])
    best_colsample_bytree_lgbm = best_lgbm["colsample_bytree"]

    # Create the best estimator
    best_estimator_lgbm = lgb.LGBMRegressor(
        max_depth=best_max_depth_lgbm,
        num_leaves=best_num_leaves_lgbm,
        learning_rate=best_learning_rate_lgbm,
        n_estimators=best_n_estimators_lgbm,
        colsample_bytree=best_colsample_bytree_lgbm,
    )

    # Fit the best estimator on the full training set
    best_estimator_lgbm.fit(X_train, y_train)

    # Save the best estimator
    dump(
        best_estimator_lgbm,
        f"/content/drive/MyDrive/Tesis/lgbm_hyperopt_only_fishing_seasons_{granularity}.joblib",
    )


if __name__ == "__main__":
    main()
