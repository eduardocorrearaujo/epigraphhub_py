#!/usr/bin/env python3

"""
This module contains functions to apply neural networks with multiple outputs. The focus of the function
is forecast time series using recurrent neural networks as lstm ans bi-lstm. 
"""

import pickle
from datetime import datetime, timedelta
from time import time

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from matplotlib import pyplot as plt
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.utils import plot_model

from epigraphhub.analysis.clustering import compute_clusters
from epigraphhub.data.get_data import (
    get_cluster_data,
    get_georegion_data,
    get_updated_data_swiss,
)
from epigraphhub.data.preprocessing import lstm_split_data as split_data
from epigraphhub.data.preprocessing import normalize_data


def transform_data(
    target_curve_name,
    df,
    look_back=21,
    predict_n=14,
    split=0.75,
):

    """
    This function normalize a given dataset and split then in a format accept by a neural
    network.
    :param target_curve_name: string. Name of the target column.
    :param df: DataFrame. Dataframe with features and targets.
    :param look_back: int. Number of past observations that will be used in the prediction.
    :param predict_n: int. Size of the forecast horizon.
    :param split: float. Porcentage of the data that will be used for train the model.

    :returns:
             Array. Array of size = (n, loook_back, len(df.columns))
             Array. Array of size = (n, predict_n)
             Array. Array of size = (n, loook_back, len(df.columns))
             Array. Array of size = (n, predict_n)
             float.
             list of dates.
             Array. Array of size = (1, loook_back, len(df.columns))
    """
    indice = list(df.index)
    indice = [i.date() for i in indice]

    target_col = list(df.columns).index(f"{target_curve_name}")

    norm_data, max_features = normalize_data(df)
    factor = max_features[target_col]

    X_forecast = np.empty((1, look_back, norm_data.shape[1]))

    X_forecast[:, :, :] = norm_data[-look_back:]

    X_train, Y_train, X_test, Y_test = split_data(
        norm_data, look_back, split, predict_n, Y_column=target_col
    )

    return X_train, Y_train, X_test, Y_test, factor, indice, X_forecast


def get_data_model(
    target_curve_name,
    cluster,
    predictors,
    ini_date=None,
    look_back=21,
    predict_n=14,
    split=0.75,
    vaccine=True,
    smooth=True,
    updated_data=False,
):

    """
    Function to get and the transform the switzerland data in the requested format.
    :params target_curve_name: string. Name of the target curve
    :params cluster: List of strings. Name of the cantons to get the data
    :params predictors: List of strings. Name of the tables to get the data from
    :params ini_date: string|None. Filter the dataset from a specific date.
    :param look_back: int. Number of past information that the network uses to learn about the forecasted values
    :param predict_n: int. Forecast horizon.
    :params split: float. Percentage of data used to train the model.
    :params vaccine: Boolean. If True the vaccine data for switzerland is used.
    :params smooth: Boolean. If True a rolling average of 7 days is applied in the data.
    :params updated_data. Boolean. Only valid for canton = 'GE'

    :returns:
             Array. Array of size = (n, loook_back, len(df.columns))
             Array. Array of size = (n, predict_n)
             Array. Array of size = (n, loook_back, len(df.columns))
             Array. Array of size = (n, predict_n)
             float.
             list of dates.
             Array. Array of size = (1, loook_back, len(df.columns))
    """

    df = get_cluster_data(
        "switzerland", predictors, list(cluster), vaccine=vaccine, smooth=smooth
    )

    df = df.fillna(0)

    # removing the last three days of data to avoid delay in the reporting.
    df = df.iloc[:-3]

    if ini_date is not None:
        df = df.loc[ini_date:]

    if f"{target_curve_name}" == "hosp_GE":
        if updated_data:

            df_new = get_updated_data_swiss(smooth)

            df.loc[df_new.index[0] : df_new.index[-1], "hosp_GE"] = df_new.hosp_GE

            df = df.loc[: df_new.index[-1]]

    X_train, Y_train, X_test, Y_test, factor, indice, X_forecast = transform_data(
        f"{target_curve_name}",
        df,
        look_back=look_back,
        predict_n=predict_n,
        split=split,
    )

    return X_train, Y_train, X_test, Y_test, factor, indice, X_forecast


def build_model(hidden, features, predict_n, look_back=10, batch_size=1):
    """
    Builds and returns the LSTM model with the parameters given
    :param hidden: number of hidden nodes
    :param features: number of variables in the example table
    :param look_back: Number of time-steps to look back before predicting
    :param batch_size: batch size for batch training
    :return: sequential model.
    """

    inp = keras.Input(
        shape=(look_back, features),
        # batch_shape=(batch_size, look_back, features)
    )
    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful=False,
        kernel_initializer="he_uniform",
        batch_input_shape=(batch_size, look_back, features),
        return_sequences=True,
        activation="relu",
        dropout=0.1,
        recurrent_dropout=0.1,
        implementation=2,
        name="lstm",
        unit_forget_bias=True,
    )(inp, training=True)
    x = Dropout(0.2, name="dropout")(x, training=True)
    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        kernel_initializer="he_uniform",
        stateful=False,
        batch_input_shape=(batch_size, look_back, features),
        # return_sequences=True,
        activation="relu",
        dropout=0.1,
        recurrent_dropout=0.1,
        implementation=2,
        unit_forget_bias=True,
        name="lstm_2",
    )(x, training=True)
    x = Dropout(0.2, name="dropout_1")(x, training=True)
    out = Dense(
        predict_n,
        activation="relu",
        kernel_initializer="he_uniform",
        bias_initializer="zeros",
        name="dense",
    )(x)
    model = keras.Model(inp, out)

    start = time()
    model.compile(loss="msle", optimizer="adam", metrics=["accuracy", "mape", "mse"])
    print("Compilation Time : ", time() - start)
    plot_model(model, to_file="LSTM_model.png")
    print(model.summary())
    return model


def train(
    model,
    X_train,
    Y_train,
    batch_size=1,
    epochs=10,
    path=None,
    label_history="history",
    label_model="trained_model",
    save=False,
):
    """
    Function to train a LSTM model and save the history of the model
    :param model: model to be trained
    :param X_train: arrays. Features to train the model
    :param Y_train: arrays. Targets of the model
    :param batch_size: int. batch_size used to compute the model
    :param epochs: int. epochs of the model
    :params path: string. Where the model will be saved
    :params label_history: string. Name of the file with the history model
    :params label_model: string. Name of the file with the trained model
    :params save: Boolean. Decide if the history will be saved or not
    :return: model fitted
    """

    TB_callback = TensorBoard(
        log_dir="./tensorboard",
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        # embeddings_freq=10
    )

    hist = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.15,
        shuffle=False,
        verbose=1,
        callbacks=[TB_callback, EarlyStopping(patience=15)],
    )

    if save:
        if path is None:
            with open(f"{label_history}_{epochs}.pkl", "wb") as f:
                pickle.dump(hist.history, f)

            model.save(f"{label_model}_{epochs}.h5")

        else:
            with open("f'{path}/{label_history}_{epochs}.pkl", "wb") as f:
                pickle.dump(hist.history, f)

            model.save(f"{path}/{label_model}_{epochs}.h5")

    return hist


def plot_training_history(
    hist, series, title="Loss series", save=False, path=None, label="training_history"
):
    """
    Plot the Loss series selected in the params series.
    :param hist: Training history object returned by "model.fit()"
    :param series: List of strings. Loss series that will be plotted. The possible options
                    are: ['loss', 'accuracy', 'mape', 'mse', 'val_loss', 'val_accuracy',
                          'val_mape', 'val_mse']

    :param title: string. Title of the plot.
    :param save: boolean. If True the plot is saved
    :param path: string. Path to save the file
    :param label: string. Name to save the plot
    :returns: None
    """
    fig, ax = plt.subplots(dpi=300)
    for i in series:
        df_ = pd.DataFrame(hist.history[i], columns=[i])
        df_.plot(logy=True, ax=ax)
    plt.grid()
    plt.title(title)

    if save:
        if path:
            plt.savefig(f"{path}/{label}.png", bbox_inches="tight")

        else:
            plt.savefig(f"{label}.png", bbox_inches="tight")

    plt.show()

    return None


def plot_predicted_vs_data(
    predicted,
    Ydata,
    indice,
    title,
    label,
    factor,
    xlabel="time",
    ylabel="incidence",
    split_point=None,
    uncertainty=False,
    save=False,
    path=None,
):

    """
    Plot the model's predictions against data
    :params predicted:array. model predictions
    :params Ydata:array. observed data
    :params indice: array|Series|list. dates of the observed dates
    :params title: string. Title of the plot
    :params label: string. name to save the plot
    :params factor: float. Normalizing factor for the target variable
    :params xlabel: string. Name of the x axis in the plot.
    :params ylabel: string. Name of the y axis in the plot.
    :params split_point: float. Separation between the train and test datasets. It's a value in the range (0,1)
    :params uncertainty: boolean. If is possible to compute the confidence interval of the predictions from
                            the predict param
    :params save: If True the plot is saved.
    :params path: string. Name of the path that the model will be saved.
    :returns: None
    """

    plt.clf()
    if len(predicted.shape) == 2:
        df_predicted = pd.DataFrame(predicted)
    else:
        df_predicted = pd.DataFrame(np.percentile(predicted, 50, axis=2))
        df_predicted25 = pd.DataFrame(np.percentile(predicted, 2.5, axis=2))
        df_predicted975 = pd.DataFrame(np.percentile(predicted, 97.5, axis=2))
        uncertainty = True

    if split_point is not None:
        if uncertainty:
            ymax = max(
                max(df_predicted.iloc[:, -1]) * factor,
                max(df_predicted25.iloc[:, -1]) * factor,
                max(df_predicted975.iloc[:, -1]) * factor,
                Ydata.max() * factor,
            )
            plt.vlines(
                indice[split_point], 0, ymax, "g", "dashdot", lw=2, label="Train/Test"
            )

        else:
            ymax = max(predicted.max() * factor, Ydata.max() * factor)
            plt.vlines(
                indice[split_point], 0, ymax, "g", "dashdot", lw=2, label="Train/Test"
            )

    # plot only the last (furthest) prediction point
    plt.plot(
        indice[len(indice) - Ydata.shape[0] :],
        Ydata[:, -1] * factor,
        "k-",
        alpha=0.7,
        label="data",
    )
    plt.plot(
        indice[len(indice) - Ydata.shape[0] :],
        df_predicted.iloc[:, -1] * factor,
        "r-",
        alpha=0.5,
        label="median",
    )
    if uncertainty:
        plt.fill_between(
            indice[len(indice) - Ydata.shape[0] :],
            df_predicted25[df_predicted25.columns[-1]] * factor,
            df_predicted975[df_predicted975.columns[-1]] * factor,
            color="b",
            alpha=0.3,
            label="95%",
        )

    tag = "_unc" if uncertainty else ""
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30)
    plt.legend()
    if save:
        if path is None:
            plt.savefig(f"{label}.png", bbox_inches="tight", dpi=300)

        else:
            plt.savefig(f"{path}/{label}.png", bbox_inches="tight", dpi=300)
    plt.show()


def predict(model, Xdata, uncertainty=False):
    """
    Function to make predictions of a trained model
    :params model: Pre trained model.
    :params Xdata: array. Array of features to make the predictions.
    :params uncertainty: Boolean. If true the model return 100 predictions and not just one.

    :returns: Array. Array with the predictions.
    """

    if uncertainty:
        predicted = np.stack([model.predict(Xdata, batch_size=1, verbose=1) for i in range(100)], axis=2)  # type: ignore
    else:
        predicted = model.predict(Xdata, batch_size=1, verbose=1)

    return predicted


def calculate_metrics(pred, ytrue, factor, uncertainty=True):
    """
    Function to compute some errors metrics of the predictions and the real data
    :params pred: array 2D. Array of 2D size with the predictions
    :params ytrue: array 2D. Array of 2D size with the real data
    :params factor: float. Correction factor to transform the pred array in the same scale of ytrue
    :params uncertainty. Boolean. If true the metrics will be calcultated according to the median.
    :returns: DataFrame
    """

    if uncertainty:
        pred = np.percentile(pred, 50, axis=2)

    metrics = pd.DataFrame(
        index=(
            "mean_absolute_error",
            "explained_variance_score",
            "mean_squared_error",
            "mean_squared_log_error",
            "median_absolute_error",
            "r2_score",
        )
    )
    for col in range(pred.shape[1]):
        y = ytrue[:, col] * factor
        p = pred[:, col] * factor
        l = [
            mean_absolute_error(y, p),
            explained_variance_score(y, p),
            mean_squared_error(y, p),
            mean_squared_log_error(y, p),
            median_absolute_error(y, p),
            r2_score(y, p),
        ]
        metrics[col] = l

    return metrics


def train_eval(
    model,
    X_train,
    Y_train,
    X_test,
    Y_test,
    factor,
    indice,
    batch=1,
    epochs=100,
    path=None,
    label_history="history_region",
    label_model="trained_model_region",
    uncertainty=True,
    save=False,
):

    """
    Function to train and evaluate a model
    :params model: model that will be trained.
    :params X_train: array. Array of features to train the model.
    :params Y_train: array. Array with the targets to train the model.
    :params X_test: array. Array with the features to teste the model.
    :params Y_test: array. Array with the targets to test the model.
    :params factor: float. It used to change the scale of the predictions of the model.
    :params indice: list of dates. List with the dates associated with the target values.
    :params uncertainty: Boolean. If true the model return 100 predictions and not just one.
    :params batch: int. batch_size used to compute the model.
    :params epochs: int. Number of times that the model will be trained.
    :params path: string. String to save the model trained.
    :params label_history: string. File name where the history of the model will be saved.
    :params label_model: string. File name where the model will be saved.
    :params uncertainty: boolean. If true a confidence interval for the predictions are returned.
    :params save: boolean. If true the trained model is saved.

    :returns: Dataframe. The dataframe has columns for the target values an the predictions of the model.
    """

    history = train(
        model,
        X_train,
        Y_train,
        batch_size=batch,
        epochs=epochs,
        label_history=label_history,
        label_model=label_model,
        path=path,
        save=save,
    )

    Y_data = np.concatenate((Y_train, Y_test), axis=0)  # type: ignore

    predicted_out = predict(model, X_test, uncertainty)

    predicted_in = predict(model, X_train, uncertainty)

    predicted = np.concatenate((predicted_in, predicted_out), axis=0)  # type: ignore

    df_pred = pd.DataFrame()

    if uncertainty:
        df_predicted = pd.DataFrame(np.percentile(predicted, 50, axis=2))
        df_predicted25 = pd.DataFrame(np.percentile(predicted, 2.5, axis=2))
        df_predicted975 = pd.DataFrame(np.percentile(predicted, 97.5, axis=2))

        df_pred["date"] = indice[X_train.shape[1] + Y_train.shape[1] :]

        df_pred["target"] = Y_data[1:, -1] * factor

        df_pred["lower"] = df_predicted25[df_predicted25.columns[-1]] * factor

        df_pred["median"] = df_predicted[df_predicted.columns[-1]] * factor

        df_pred["upper"] = df_predicted975[df_predicted975.columns[-1]] * factor

        df_pred["train_size"] = [
            len(X_train) - (X_train.shape[1] + Y_train.shape[1])
        ] * len(df_pred)

    else:
        if len(predicted.shape) == 2:
            df_predicted = pd.DataFrame(predicted)

        df_pred["date"] = indice[Y_train.shape[1] + X_train.shape[1] :]

        df_pred["target"] = Y_data[1:, -1] * factor

        df_pred["predict"] = df_predicted[df_predicted.columns[-1]] * factor

        df_pred["train_size"] = [
            len(X_train) - (Y_train.shape[1] + X_train.shape[1])
        ] * len(df_pred)

    return df_pred


def forecast(
    X_for,
    factor,
    indice,
    epochs,
    path=None,
    label_model="trained_model_region",
    uncertainty=True,
):

    """
    Function to forecast a trained and saved model
    :params X_for: array. Array of features to apply the forecast.
    :params factor: float. It used to change the scale of the predictions of the model.
    :params indice: list of dates. List with the dates associated with the target values.
    :params epochs: int. Number of epochs used to train the model.
    :params path: string. Indicates where the model was saved.
    :params label_model: string. The file name of the saved model.
    :params uncertainty: boolean. If true a confidence interval for the predictions are returned.

    :returns: Dataframe. The dataframe has columns with the forecasted values
    """

    if path is None:
        model = keras.models.load_model(f"{label_model}_{epochs}.h5")
    else:
        model = keras.models.load_model(f"{path}/{label_model}_{epochs}.h5")

    if uncertainty:
        predicted = np.stack([model.predict(X_for, batch_size=1, verbose=1) for i in range(100)], axis=2)  # type: ignore
    else:
        predicted = model.predict(X_for, batch_size=1, verbose=1)

    forecast_dates = []

    last_day = datetime.strftime(indice[-1], "%Y-%m-%d")

    a = datetime.strptime(last_day, "%Y-%m-%d")

    for i in np.arange(1, len(predicted[0]) + 1):

        d_i = datetime.strftime(a + timedelta(days=int(i)), "%Y-%m-%d")

        forecast_dates.append(datetime.strptime(d_i, "%Y-%m-%d"))

    df_for = pd.DataFrame()

    if uncertainty:
        df_predicted = pd.DataFrame(np.percentile(predicted, 50, axis=2)).T
        df_predicted25 = pd.DataFrame(np.percentile(predicted, 2.5, axis=2)).T
        df_predicted975 = pd.DataFrame(np.percentile(predicted, 97.5, axis=2)).T

        df_for["date"] = forecast_dates

        df_for["lower"] = df_predicted25[df_predicted25.columns[-1]] * factor

        df_for["median"] = df_predicted[df_predicted.columns[-1]] * factor

        df_for["upper"] = df_predicted975[df_predicted975.columns[-1]] * factor

    else:
        if len(predicted.shape) == 2:
            df_predicted = pd.DataFrame(predicted).T

        df_for["date"] = forecast_dates

        df_for["predict"] = df_predicted[df_predicted.columns[-1]] * factor

    return df_for


def train_eval_single_canton(
    target_curve_name,
    canton,
    predictors,
    split=0.75,
    vaccine=True,
    smooth=True,
    ini_date=None,
    updated_data=False,
    uncertainty=True,
    path=None,
    save=False,
    hidden=4,
    epochs=100,
    look_back=21,
    predict_n=14,
    label_model="trained_eval_model",
    label_history="history_eval",
):

    """
    Function to train and evaluate the model for one georegion
    :params target_curve_name: string. Name of the target column.
    :params canton: string. Name of the canton.
    :params predictors: Name of the tables that will be used to create the features of the model.
    :params split: float. Percentage of data used to train the model.
    :params vaccine: It determines if the vaccine data from owid will be used or not.
    :params smooth: It determines if data will be smoothed or not.
    :params ini_date: Determines the beggining of the train dataset.
    :params updated_data: boolean. If true the HUG data is used.
    :params uncertainty: boolean. If true a confidence interval for the predictions are returned.
    :params path: string. Indicates where the model will be saved.
    :params save: boolean. If true the model is saved.
    :params hidden: int. Number of the hidden layers used in the prediction.
    :params look_back: int. Number of past informations used in the predictions.
    :params predict_n. int. Forecast horizon.
    :params label_model: string. Filename to save the model.
    :params label_history. string. Filename to save the model history.

    :returns: DataFrame.
    """

    X_train, Y_train, X_test, Y_test, factor, indice, X_forecast = get_data_model(
        f"{target_curve_name}_{canton}",
        [canton],
        predictors,
        ini_date=ini_date,
        look_back=look_back,
        predict_n=predict_n,
        split=split,
        vaccine=vaccine,
        smooth=smooth,
        updated_data=updated_data,
    )

    model = build_model(
        hidden, X_train.shape[2], predict_n=predict_n, look_back=look_back
    )

    # get predictions
    df = train_eval(
        model,
        X_train,
        Y_train,
        X_test,
        Y_test,
        factor,
        indice,
        batch=1,
        epochs=epochs,
        label_model=label_model,
        label_history=label_history,
        uncertainty=uncertainty,
        save=save,
        path=path,
    )

    df["canton"] = [canton] * len(df)

    return df


def train_eval_all_cantons(
    target_curve_name,
    predictors,
    split=0.75,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    uncertainty=True,
    save=False,
    path=None,
    hidden=12,
    epochs=100,
    look_back=21,
    predict_n=14,
    label_history="history_trained_model",
    label_model="trained_model",
):

    """
    Function to train and evaluate all the cantons in switzerland
    :params target_curve_name: string. Name of the target column.
    :params predictors: Name of the tables that will be used to create the features of the model.
    :params split: float. Percentage of data used to train the model.
    :params vaccine: It determines if the vaccine data from owid will be used or not.
    :params smooth: It determines if data will be smoothed or not.
    :params ini_date: Determines the beggining of the train dataset.
    :params updated_data: boolean. If true the HUG data is used.
    :params uncertainty: boolean. If true a confidence interval for the predictions are returned.
    :params path: string. Indicates where the model will be saved.
    :params save: boolean. If true the model is saved.
    :params hidden: int. Number of the hidden layers used in the prediction.
    :params look_back: int. Number of past informations used in the predictions.
    :params predict_n. int. Forecast horizon.
    :params label_model: string. Filename to save the model.
    :params label_history. string. Filename to save the model history.

    :returns: DataFrame.
    """

    df_all = pd.DataFrame()

    cantons = get_georegion_data("switzerland", "foph_cases", "All", ['"geoRegion"'])
    cantons = cantons.geoRegion.unique()
    cantons = list(cantons)
    cantons.remove("CH")
    cantons.remove("CHFL")
    cantons.remove("FL")

    for canton in cantons:
        df_pred = train_eval_single_canton(
            target_curve_name,
            canton,
            predictors,
            split=split,
            vaccine=vaccine,
            smooth=smooth,
            ini_date=ini_date,
            updated_data=False,
            uncertainty=uncertainty,
            epochs=epochs,
            path=path,
            save=save,
            hidden=hidden,
            look_back=look_back,
            predict_n=predict_n,
            label_history=f"{label_history}_{canton}",
            label_model=f"{label_model}_{canton}",
        )

        df_all = pd.concat([df_all, df_pred])

    return df_all


def train_single_canton(
    target_curve_name,
    canton,
    predictors,
    vaccine=True,
    smooth=True,
    ini_date=None,
    updated_data=True,
    save=False,
    path=None,
    hidden=4,
    epochs=100,
    batch=1,
    label_model="trained_model",
    label_history="history_trained_model",
    look_back=21,
    predict_n=14,
):

    """
    Function to train a model for one canton (given all the data available)
    :params target_curve_name: string. Name of the target column.
    :params canton: string. Name of the canton.
    :params predictors: Name of the tables that will be used to create the features of the model.
    :params vaccine: It determines if the vaccine data from owid will be used or not.
    :params smooth: It determines if data will be smoothed or not.
    :params ini_date: Determines the beggining of the train dataset.
    :params updated_data: boolean. If true the HUG data is used.
    :params uncertainty: boolean. If true a confidence interval for the predictions are returned.
    :params path: string. Indicates where the model will be saved.
    :params save: boolean. If true the model is saved.
    :params hidden: int. Number of the hidden layers used in the prediction.
    :params look_back: int. Number of past informations used in the predictions.
    :params predict_n. int. Forecast horizon.
    :params label_model: string. Filename to save the model.
    :params label_history. string. Filename to save the model history.

    :returns: DataFrame.
    """

    X_train, Y_train, X_test, Y_test, factor, indice, X_forecast = get_data_model(
        f"{target_curve_name}_{canton}",
        [canton],
        predictors,
        ini_date=ini_date,
        look_back=look_back,
        predict_n=predict_n,
        split=1,
        vaccine=vaccine,
        smooth=smooth,
        updated_data=updated_data,
    )

    model = build_model(
        hidden, X_train.shape[2], predict_n=predict_n, look_back=look_back
    )

    history = train(
        model,
        X_train,
        Y_train,
        batch_size=batch,
        epochs=epochs,
        label_history=label_history,
        label_model=label_model,
        save=save,
        path=path,
    )

    return


def train_all_cantons(
    target_curve_name,
    predictors,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    updated_data=True,
    save=False,
    path=None,
    hidden=12,
    epochs=100,
    look_back=21,
    predict_n=14,
    label_model="train_model",
    label_history="history_model",
):
    """
    Function to train a model for each canton in switerland (given all the data available)
    :params target_curve_name: string. Name of the target column.
    :params predictors: Name of the tables that will be used to create the features of the model.
    :params vaccine: It determines if the vaccine data from owid will be used or not.
    :params smooth: It determines if data will be smoothed or not.
    :params ini_date: Determines the beggining of the train dataset.
    :params updated_data: boolean. If true the HUG data is used.
    :params uncertainty: boolean. If true a confidence interval for the predictions are returned.
    :params path: string. Indicates where the model will be saved.
    :params save: boolean. If true the model is saved.
    :params hidden: int. Number of the hidden layers used in the prediction.
    :params look_back: int. Number of past informations used in the predictions.
    :params predict_n. int. Forecast horizon.
    :params label_model: string. Filename to save the model.
    :params label_history. string. Filename to save the model history.

    :returns: DataFrame.
    """
    cantons = get_georegion_data("switzerland", "foph_cases", "All", ['"geoRegion"'])
    cantons = cantons.geoRegion.unique()
    cantons = list(cantons)
    cantons.remove("CH")
    cantons.remove("CHFL")
    cantons.remove("FL")

    for canton in cantons:

        train_single_canton(
            target_curve_name,
            canton,
            predictors,
            vaccine=vaccine,
            smooth=smooth,
            ini_date=ini_date,
            updated_data=updated_data,
            save=save,
            path=path,
            hidden=hidden,
            epochs=epochs,
            label_model=f"{label_model}_{canton}",
            label_history=f"{label_history}_{canton}",
            look_back=look_back,
            predict_n=predict_n,
        )

    return


def forecast_single_canton(
    epochs,
    target_curve_name,
    canton,
    predictors,
    vaccine=True,
    smooth=True,
    ini_date=None,
    updated_data=True,
    uncertainty=True,
    path=None,
    label_model="trained_model",
    look_back=21,
    predict_n=14,
):
    """
    Function to forecast one canton given a pre saved model
    :params target_curve_name: string. Name of the target column.
    :params canton: string. Name of the canton.
    :params predictors: Name of the tables that will be used to create the features of the model.
    :params vaccine: It determines if the vaccine data from owid will be used or not.
    :params smooth: It determines if data will be smoothed or not.
    :params ini_date: Determines the beggining of the train dataset.
    :params updated_data: boolean. If true the HUG data is used.
    :params uncertainty: boolean. If true a confidence interval for the predictions are returned.
    :params path: string. Indicates where the model will be saved.
    :params look_back: int. Number of past informations used in the predictions.
    :params predict_n. int. Forecast horizon.
    :params label_model: string. Filename to save the model.

    :returns: DataFrame.
    """

    X_train, Y_train, X_test, Y_test, factor, indice, X_forecast = get_data_model(
        f"{target_curve_name}_{canton}",
        [canton],
        predictors,
        ini_date=ini_date,
        look_back=look_back,
        predict_n=predict_n,
        split=1,
        vaccine=vaccine,
        smooth=smooth,
        updated_data=updated_data,
    )

    df_for = forecast(
        X_forecast,
        factor,
        indice,
        epochs,
        path=path,
        label_model=label_model,
        uncertainty=uncertainty,
    )

    df_for["canton"] = [canton] * len(df_for)

    return df_for


def forecast_all_cantons(
    target_curve_name,
    predictors,
    epochs=100,
    vaccine=True,
    smooth=True,
    ini_date="2020-03-01",
    updated_data=True,
    uncertainty=True,
    path=None,
    label_model="trained_model",
    look_back=21,
    predict_n=14,
):
    """
    Function to forecast all cantons in switzerland given a pre saved model
    :params target_curve_name: string. Name of the target column.
    :params canton: string. Name of the canton.
    :params predictors: Name of the tables that will be used to create the features of the model.
    :params vaccine: It determines if the vaccine data from owid will be used or not.
    :params smooth: It determines if data will be smoothed or not.
    :params ini_date: Determines the beggining of the train dataset.
    :params updated_data: boolean. If true the HUG data is used.
    :params uncertainty: boolean. If true a confidence interval for the predictions are returned.
    :params path: string. Indicates where the model will be saved.
    :params look_back: int. Number of past informations used in the predictions.
    :params predict_n. int. Forecast horizon.
    :params label_model: string. Filename to save the model.

    :returns: DataFrame.
    """
    df_all = pd.DataFrame()

    cantons = get_georegion_data("switzerland", "foph_cases", "All", ['"geoRegion"'])
    cantons = cantons.geoRegion.unique()
    cantons = list(cantons)
    cantons.remove("CH")
    cantons.remove("CHFL")
    cantons.remove("FL")

    for canton in cantons:

        df_for = forecast_single_canton(
            epochs,
            target_curve_name,
            canton,
            predictors,
            vaccine=vaccine,
            smooth=smooth,
            ini_date=ini_date,
            updated_data=updated_data,
            uncertainty=uncertainty,
            path=path,
            label_model=f"{label_model}_{canton}",
            look_back=look_back,
            predict_n=predict_n,
        )

        df_all = pd.concat([df_all, df_for])

    return df_all
