import numpy as np
import datetime
import pandas as pd
import random


def _get_year(amplitude:float=1., bias:float=0.):
    return amplitude * np.sin(np.linspace(-np.pi/2, 3/2*np.pi, 48*365)) + bias


def _get_day():
    return np.sin(np.linspace(-np.pi/2, 3/2*np.pi, 48))


def _get_days(n_days:int=365):
    day = _get_day()
    days = np.empty((0,))
    for d in range(365):
        days = np.append(days, day)
    return days


def _get_period(n_ticks:int=48*7):
    return np.sin(np.linspace(0, 2*np.pi, n_ticks))


def _get_time(year:int=2049):
    timeline = []
    dt = datetime.datetime(year, 1, 1, 0, 0, 0)
    td = datetime.timedelta(minutes=30)
    for i in range(48*365):
        timeline.append(dt)
        dt = dt + td
    return timeline


def _add_noise(weather_data:pd.DataFrame, deviation:float=0.02, outliers:float=0.1, nans:float=0.1, round:int=2):
    result = weather_data.copy()
    #add deviation (except datetime)
    for key in result.columns[1:]:
        result.loc[key] = np.round(result[key] * random.uniform(1 - deviation, 1 + deviation), round)
    #add outliers and nans
    if outliers or nans:
        for key in result.columns[1:]:
            for i in range(48*365):
                if random.random() < nans: 
                    result.loc[i, key] = None
                if random.random() < outliers:
                    result.loc[i, key] = np.round(result.loc[i, key] * 1_000_000, round)
    return result
