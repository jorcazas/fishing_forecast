import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from model import Model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.callbacks import EarlyStopping, ModelCheckpoint
from wandb.keras import WandbMetricsLogger
import wandb
import pickle

class LSTM(Model):  
    def __init__(self):
        super().__init__()


def load_and_preprocess_data(root):
    data = pd.read_csv(os.path.join(root, "lstm_xgboost_data.csv"))
    data = data[
        [
            "ds",
            "y",
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
            "xgboost",
        ]
    ]
    data["ds"] = pd.to_datetime(data["ds"])
    data = data.sort_values("ds")

    # Filter data based on date ranges
    date_ranges = [
        ("2017-09-15", "2018-02-15"),
        ("2018-09-15", "2019-02-15"),
        ("2019-09-15", "2020-02-15"),
        ("2020-09-15", "2021-02-15"),
        ("2021-09-15", "2022-02-15"),
    ]

    filtered_data = pd.concat(
        [
            data[
                (data["ds"] > datetime.strptime(start, "%Y-%m-%d"))
                & (data["ds"] <= datetime.strptime(end, "%Y-%m-%d"))
            ]
            for start, end in date_ranges
        ]
    )

    filtered_data.to_csv("lstm_only_fishing_season.csv")
    filtered_data = filtered_data.dropna(subset=["x1"])

    return filtered_data


def create_sequences_and_split_data(features, target, sequence_length):
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = scaler.fit_transform(features)

    # Create sequences for training
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i - sequence_length : i])
        y.append(target[i])

    X, y = np.array(X), np.array(y)

    # Split the data into training and testing sets
    split_ratio = 368 / 628  # cut before 2021 season began
    split_index = int(split_ratio * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return X_train, X_test, y_train, y_test


def define_hyperparameter_search_space():
    experiments = {}
    learning_rates = [0.001, 0.01]
    n_splits_values = [5]
    batch_sizes = [32, 256, 512]

    for first_layer in range(200, 1000, 500):
        for second_layer in range(800, 1000, 500):
            for n_splits in n_splits_values:
                for learning_rate in learning_rates:
                    for batch_size in batch_sizes:
                        key = f"{first_layer}_{second_layer}_{n_splits}_{learning_rate}_{batch_size}"
                        experiments[key] = {
                            "first_layer": first_layer,
                            "second_layer": second_layer,
                            "n_splits": n_splits,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                        }

    return experiments


def train_and_evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    experiments,
    tested_models,
    sequence_length,
    n_features,
):
    models = {}
    midpoint = len(experiments) // 2

    for key, experiment in list(experiments.items())[:midpoint]:
        if key not in tested_models:
            tf.random.set_seed(1807)
            run = wandb.init(
                project="cobi-forecast-only-fishing-seasons", name=f"lstm-{key}"
            )
            config = wandb.config
            config.learning_rate = experiment["learning_rate"]
            config.epochs = 300
            n_splits = experiment["n_splits"]
            tscv = TimeSeriesSplit(n_splits=n_splits)

            for train_index, test_index in tscv.split(X_train):
                X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

                model = Sequential()
                model.add(
                    LSTM(
                        experiment["first_layer"],
                        activation="relu",
                        return_sequences=True,
                        input_shape=(sequence_length, n_features),
                    )
                )
                model.add(Dropout(0.3))
                model.add(
                    LSTM(
                        experiment["second_layer"],
                        activation="relu",
                        input_shape=(sequence_length, n_features),
                    )
                )
                model.add(Dropout(0.3))
                model.add(Dense(1, activation="relu"))

                model.compile(optimizer="adam", loss="mean_squared_logarithmic_error")
                early_stopping = EarlyStopping(
                    monitor="val_loss", mode="min", patience=100, verbose=1
                )
                model_name = key.replace(".0", "")
                model_checkpoint = ModelCheckpoint(
                    f"/content/drive/MyDrive/Tesis/lstm/lstm_xboost_model_{model_name}_fold.h5",
                    save_best_only=True,
                    monitor="val_loss",
                    mode="min",
                    verbose=1,
                )

                history = model.fit(
                    X_train_fold,
                    y_train_fold,
                    epochs=config.epochs,
                    batch_size=experiment["batch_size"],
                    validation_data=(X_val_fold, y_val_fold),
                    callbacks=[
                        early_stopping,
                        model_checkpoint,
                        WandbMetricsLogger(model),
                    ],
                )

            model_name = key.replace(".0", "")
            model.save(
                f"/content/drive/MyDrive/Tesis/lstm/lstm_xboost_model_{model_name}.h5"
            )

            X_test_ = X_test.reshape((X_test.shape[0], sequence_length, n_features))
            predictions = model.predict(X_test_)

            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)

            models[key] = [mae, mse, rmse]
            with open("/content/drive/MyDrive/Tesis/models_2_half.pkl", "wb") as fp:
                pickle.dump(models, fp)


def evaluate_final_model(model_name, X_test, y_test, sequence_length, n_features):
    model = load_model(f"/content/drive/MyDrive/Tesis/lstm/{model_name}.h5")
    X_test_ = X_test.reshape((X_test.shape[0], sequence_length, n_features))
    predictions = model.predict(X_test_)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual", color="blue")
    plt.plot(predictions, label="Predicted", color="red")
    plt.title("Actual vs. Predicted")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def main():
    root = "path_to_data"
    data = load_and_preprocess_data(root)

    data.set_index("ds", inplace=True)
    features = data.drop(columns=["y"])
    target = data["y"]

    sequence_length = 10
    n_features = features.shape[1]

    X_train, X_test, y_train, y_test = create_sequences_and_split_data(
        features, target, sequence_length
    )

    experiments = define_hyperparameter_search_space()

    tested_models = pd.read_pickle("/content/drive/MyDrive/Tesis/models_2_half.pkl")

    train_and_evaluate_models(
        X_train,
        y_train,
        X_test,
        y_test,
        experiments,
        tested_models,
        sequence_length,
        n_features,
    )

    # Evaluate final model
    evaluate_final_model(
        "lstm_xboost_model_2700_800_5_001_512",
        X_test,
        y_test,
        sequence_length,
        n_features,
    )


if __name__ == "__main__":
    main()
