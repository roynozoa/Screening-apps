# import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
import os

CWD = os.getcwd()

HORIZON = 1 # predict next day
WINDOW_SIZE = 4 # use worth of data


def update_data():
    data = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/country_data/Indonesia.csv')
    data.drop(columns=['source_url'], inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data.total_vaccinations[-1]
    latest_date = data.index[-1]
    d = dict.fromkeys(data.select_dtypes(np.int64).columns, np.float32)
    data = data.astype(d)
    return data

def prepare_data():
    data = update_data()
    vacc_one = data.people_vaccinated.to_numpy()
    vacc_full = data.people_fully_vaccinated.to_numpy()
    timesteps = data.index.to_numpy()
    return vacc_one, vacc_full, timesteps

def create_next_timestep():
    data = update_data()
    last_timestep = data.index[-1]
    next_time_steps = get_future_dates(start_date=last_timestep, horizon=20)
    return next_time_steps




# Create function to view NumPy arrays as windows 
def make_windows(x, window_size=7, horizon=1):
    """
    Turns a 1D array into a 2D array of sequential windows of window_size.
    """
    # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
    window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
    # print(f"Window step:\n {window_step}")

    # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
    window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T # create 2D array of windows of size window_size
    # print(f"Window indexes:\n {window_indexes[:3], window_indexes[-3:], window_indexes.shape}")

    # 3. Index on the target array (time series) with 2D array of multiple window steps
    windowed_array = x[window_indexes]

    # 4. Get the labelled windows
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

    return windows, labels

# Make the train/test splits
def make_train_test_splits(windows, labels, test_split=0.2):
    """
    Splits matching pairs of windows and labels into train and test splits.
    """
    split_size = int(len(windows) * (1-test_split)) # this will default to 80% train/20% test
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels

# Create function to label windowed data
def get_labelled_windows(x, horizon=1):
    """
    Creates labels for windowed dataset.

    E.g. if horizon=1 (default)
    Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
    """
    return x[:, :-horizon], x[:, -horizon:]


def get_future_dates(start_date, horizon=1, offset=1):
    """
    Returns array of datetime values from ranging from start_date to start_date+horizon.

    start_date: date to start range (np.datetime64)
    horizon: number of day to add onto start date for range (int)
    offset: if offset=1 (default), original date is not included, if offset=0, original date is included
    """
    return np.arange(start_date + np.timedelta64(offset, "D"), start_date + np.timedelta64(horizon+1, "D"), dtype="datetime64[D]")

def predict_one():
    new_model = tf.keras.models.load_model(CWD + "/notebooks/model_experiments/model_conv1D")

    vacc_one, _, timesteps = prepare_data()
    full_windows, full_labels = make_windows(vacc_one, window_size=WINDOW_SIZE, horizon=HORIZON)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
    # Make predictions on the future

    # List for new preds
    future_forecast = []
    last_window = vacc_one[-WINDOW_SIZE:] # get the last window of the training data
    into_future = 20 # how far to predict into the future

    for i in range(into_future):
        # Make a pred for the last window, then append the prediction, append it again, append it again
        pred = new_model.predict(tf.expand_dims(last_window, axis=0))
        print(f"Predicting on: \n {last_window} -> Prediction: {tf.squeeze(pred).numpy()}\n")
        future_forecast.append(tf.squeeze(pred).numpy())
        # Update last window with new pred and get WINDOW_SIZE most recent preds (model was trained on WINDOW_SIZE windows)
        last_window = np.append(last_window, pred)[-WINDOW_SIZE:]
    
    return future_forecast

def predict_full():
    new_model = tf.keras.models.load_model(CWD + "/notebooks/model_experiments/model_conv1Dfull")
    _, vacc_full, timesteps = prepare_data()
    full_windows, full_labels = make_windows(vacc_full, window_size=WINDOW_SIZE, horizon=HORIZON)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

    # List for new preds
    future_forecast = []
    last_window = vacc_full[-WINDOW_SIZE:] # get the last window of the training data
    into_future = 20 # how far to predict into the future

    for i in range(into_future):
        # Make a pred for the last window, then append the prediction, append it again, append it again
        pred = new_model.predict(tf.expand_dims(last_window, axis=0))
        print(f"Predicting on: \n {last_window} -> Prediction: {tf.squeeze(pred).numpy()}\n")
        future_forecast.append(tf.squeeze(pred).numpy())
        # Update last window with new pred and get WINDOW_SIZE most recent preds (model was trained on WINDOW_SIZE windows)
        last_window = np.append(last_window, pred)[-WINDOW_SIZE:]
    
    return future_forecast

if __name__ == '__main__':
    one_future = predict_one()
    full_future = predict_full()
    next_time_steps = create_next_timestep()

    print(one_future, full_future, next_time_steps)
