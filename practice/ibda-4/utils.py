import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def create_sample(timeseries:np.ndarray, begin:int=0, end:int=None, input_lag:int=5, duration:int=1) -> (np.ndarray, np.ndarray):
    x = []
    y = []
    begin = begin + input_lag
    if end is None:
        end = len(timeseries) - duration

    for i in range(begin, end):
        x_indices = range(i-input_lag, i)
        y_indices = range(i, i+duration)
        x.append(timeseries[x_indices].reshape(input_lag, 1))
        y.append(timeseries[y_indices])
    return np.array(x), np.array(y)


def _create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data:list, delta:int=0, title:str='Sample Example') -> plt.figure:
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = _create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt


def plot_train_history(history:tf.keras.callbacks.History, title:str='Training and validation loss') -> None:
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()
    
    
def multi_step_plot(x, y_true, y_pred) -> plt.plot:
    plt.figure(figsize=(12, 6))
    num_in = _create_time_steps(len(x))
    num_out = np.arange(len(y_true))

    plt.plot(num_in, np.array(x), label='History')
    plt.plot(num_out, np.array(y_true), 'bo', label='True Future')
    if y_pred.any():
        plt.plot(num_out, np.array(y_pred), 'ro', label='Predicted Future')
    plt.legend(loc='upper left')
    return plt
