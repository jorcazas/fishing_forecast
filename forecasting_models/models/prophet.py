import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model.model import Model

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mango.tuner import Tuner
from mango.domain.distribution import loguniform

root = "path_to_data"

class Prophet(Model):
    def __init__(self):
        super().__init__()



def main():
    param_dict = {
        "changepoint_prior_scale": loguniform(-3, 4),
        "seasonality_prior_scale": loguniform(1, 2),
    }

    # Load your dataset
    data = pd.read_csv(root + "/Arribos2017-2021.csv")

    # Filter by 'especie' and 'unidad_economica'
    filtered_data = data[
        (data["especie"].str.contains("LANGOSTA"))
        & (data["unidad_economica"] == "LITORAL DE BAJA CALIFORNIA S DE PR DE RL")
    ]

    # Adapt to ARIMA model
    arima_data = filtered_data[["periodo_fin", "peso_desembarcado"]]

    # Rename columns to Date and Y
    arima_data.columns = ["ds", "y"]

    # Order by Date
    arima_data.sort_values(by=["ds"], inplace=True)

    # Take the sum of the values for each date
    arima_data = arima_data.groupby("ds").sum()

    # Convert 'ds' to datetime and set it as the index
    arima_data["ds"] = pd.to_datetime(arima_data["ds"])
    arima_data.set_index("ds", inplace=True)

    # Split the data into training and testing sets (SLICED at 2021-01-01)
    train = arima_data[arima_data.index <= "2021-01-01"]
    test = arima_data[arima_data.index >= "2021-01-01"]

    prophet_train = train.reset_index()
    prophet_test = test.reset_index()

    model = Prophet()

    count_called = 1

    def objective_prophet(args_list):
        global train, count_called

        print("count_called:", count_called)
        count_called += 1

        hyper_evaluated = []
        results = []
        for hyper_par in args_list:
            m = Prophet(**hyper_par)
            m.fit(prophet_train)
            y_pred = m.predict(prophet_test)
            mse = mean_squared_error(prophet_test["y"], y_pred["trend"])
            mse = mse / 10e5
            result = (-1.0) * mse
            results.append(result)
            hyper_evaluated.append(hyper_par)

        return hyper_evaluated, results

    conf_dict = {"batch_size": 5, "num_iteration": 50, "initial_random": 10}
    tuner_user = Tuner(param_dict, objective_prophet, conf_dict)
    results = tuner_user.maximize()

    best_prophet = Prophet(
        changepoint_prior_scale=0.01615965247350855,
        seasonality_prior_scale=750.1081180235057,
    )
    best_prophet = best_prophet.fit(prophet_train)

    future = best_prophet.make_future_dataframe(periods=len(prophet_test))
    forecast = best_prophet.predict(future)

    fig1 = best_prophet.plot(forecast)

    forecast_test = forecast[forecast["ds"] >= "2021-01-01"]

    print(prophet_test.sum())
    print(forecast_test.sum())

    se = np.square(forecast_test.loc[:, "yhat"] - prophet_test["y"])
    mse = np.mean(se)
    rmse = np.sqrt(mse)

    print(mse, rmse)
    print(mean_absolute_error(forecast_test.loc[:, "yhat"], prophet_test["y"]))
    print(mean_squared_error(forecast_test.loc[:, "yhat"], prophet_test["y"]))

    m = Prophet()
    m.fit(prophet_train)

    future = m.make_future_dataframe(periods=365)
    future.tail()

    forecast = m.predict(future)
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()

    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)

    from prophet.plot import plot_plotly, plot_components_plotly

    plot_plotly(m, forecast)
    plot_components_plotly(m, forecast)


if __name__ == "__main__":
    main()
