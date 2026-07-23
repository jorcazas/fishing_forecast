import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import Model

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from mango import scheduler, Tuner

root = "path_to_data/"


class ARIMA(Model):
    def __init__(self):
        super().__init__()



def main():
    # Load your dataset
    data = pd.read_csv(root + "lstm_data.csv")

    # Filter by 'especie' and 'unidad_economica'
    arima_data = data[["ds", "y"]]

    # Order by Date
    arima_data.sort_values(by=["ds"], inplace=True)

    # Take the sum of the values for each date
    arima_data = arima_data.groupby("ds").sum()

    # Convert 'ds' to datetime and set it as the index
    arima_data["ds"] = pd.to_datetime(arima_data["ds"])
    arima_data.set_index("ds", inplace=True)

    # Perform any necessary data transformations here, if needed
    # For example, you can take the log of 'Y' if the data is not stationary
    # data['Y'] = np.log(data['Y'])

    # Split the data into training and testing sets (SLICED at 2021-01-01)
    cut_date = "2020-07-01"
    train = arima_data[arima_data.index < cut_date]
    test = arima_data[arima_data.index >= cut_date]

    def arima_objective_function(args_list):
        global data_values

        params_evaluated = []
        results = []

        for params in args_list:
            try:
                p, d, q = params["p"], params["d"], params["q"]
                trend = params["trend"]

                model = ARIMA(data_values, order=(p, d, q), trend=trend)
                predictions = model.fit()
                mse = mean_squared_error(data_values, predictions.fittedvalues)
                params_evaluated.append(params)
                results.append(mse)
            except:
                params_evaluated.append(params)
                results.append(1e5)

        return params_evaluated, results

    param_space = dict(
        p=range(0, 50), d=range(0, 50), q=range(0, 50), trend=["n", "c", "t", "ct"]
    )

    conf_dict = {"num_iteration": 10}
    data_values = list(arima_data[arima_data.index < cut_date]["y"])
    tuner = Tuner(param_space, arima_objective_function, conf_dict)
    results = tuner.minimize()

    print("Best parameters:", results["best_params"])
    print("Best loss:", results["best_objective"])

    def plot_arima(data_values, order=(1, 1, 1), trend="c"):
        print("Final model:", order, trend)
        model = ARIMA(data_values, order=order, trend=trend)
        results = model.fit()

        error = mean_squared_error(data_values, results.fittedvalues)
        print("MSE error is:", error)

        plt.figure(figsize=(15, 6))
        plt.plot(data_values, label="Original Series", linewidth=4)
        plt.plot(
            results.fittedvalues,
            color="red",
            label="Predictions",
            linestyle="dashed",
            linewidth=3,
        )
        plt.legend(fontsize=25)
        plt.xlabel("Months", fontsize=25)
        plt.ylabel("Count", fontsize=25)
        plt.show()

    best_order = (
        results["best_params"]["p"],
        results["best_params"]["d"],
        results["best_params"]["q"],
    )
    plot_arima(data_values, order=best_order, trend=results["best_params"]["trend"])

    # Fit an ARIMA model
    order = (1, 0, 2)
    model = ARIMA(train, order=order)
    model_fit = model.fit()

    # Make predictions
    predictions = model_fit.forecast(steps=len(test))

    # Calculate RMSE to evaluate the model
    rmse = np.sqrt(mean_squared_error(test, predictions))
    print(f"RMSE: {rmse}")

    # Plot the original data and predictions
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test["y"], label="Actual")
    plt.xticks(np.arange(0, len(test), 14))
    plt.plot(test.index, predictions, label="Predicted", color="red")
    plt.legend()
    plt.title("ARIMA Forecast")
    plt.show()


if __name__ == "__main__":
    main()
