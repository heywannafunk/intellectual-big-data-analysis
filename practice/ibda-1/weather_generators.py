import numpy as np
import random
import utils
import pandas as pd
import matplotlib.pyplot as plt

def generate_temperature(winter_avg:float, summer_avg:float, round:int=2):
    '''Генерирует синтетические данные о температуре за год.'''
    
    amplitude = (np.abs(winter_avg) + np.abs(summer_avg)) / 2
    
    # year carrier
    year = utils._get_year(amplitude, amplitude + winter_avg)
    
     # day carrier
    whole_year_daily = utils._get_days()
    
    # modulation
    year_temperature = year + whole_year_daily
    
    year_temperature = np.round(year_temperature, round)  
    return year_temperature


def generate_lighting(round:int=2):
    '''Генерирует синтетические данные об освещённости за год.'''
    
    # modulator
    modulator = -np.sin(np.linspace(-np.pi/2, 3/2*np.pi, 48))**2 + 1
    whole_year_modulator = np.empty((0,))
    for day in range(365):
        whole_year_modulator = np.append(whole_year_modulator, modulator)
        
    # day carrier
    whole_year_daily = utils._get_days()
        
    # year carrier
    year = utils._get_year()
    
    # year lighting 
    year_lighting = whole_year_daily + (year * whole_year_modulator)/2
    
    year_lighting = np.round(year_lighting, round)  
    return year_lighting


def generate_humidity(temperature_data:np.ndarray, round:int=2):
    '''Генерирует синтетические данные о влажности за год.'''
    
    # Аппроксимация зависимости максимальной абсолютной влажности в г/м^3 от температуры:
    #     y = np.exp(1.3095+0.0565*x)
    #     (набор значений, по которым проводилась аппроксимация: https://ru.wikipedia.org/wiki/Влажность) 
    #
    # Однако, для простоты функция генерирует относительную влажность, исходя из предположения, 
    # что её величина имеет нормальное распределение. 
    
    year_humidity = np.empty((0,), dtype=float)
    
    for t in temperature_data:
        year_humidity = np.append(year_humidity, random.gauss(mu=0.75, sigma=0.04))
    
    year_humidity = np.round(year_humidity, round)  
    return year_humidity


def generate_pressure(temperature_data:np.ndarray, humidity_data:np.ndarray, round:int=2):
    '''Генерирует синтетические данные об атмосферном давлении за год.'''
    
    # normalized temperature
    normalized_temperature = (temperature_data - temperature_data.min()) / (temperature_data.max() - temperature_data.min())
    normalized_humidity = (humidity_data - humidity_data.min()) / (humidity_data.max() - humidity_data.min())
    
    # Отмечены колебания атмосферного давления на уровне моря в пределах 641 — 816 мм рт. ст.
    # https://ru.wikipedia.org/wiki/Атмосферное_давление
    year_pressure = np.empty((48*365,), dtype=float)
    
    for i, t in enumerate(year_pressure):
        multiplier = normalized_temperature[i] - 0.2 * normalized_humidity[i]
        # 800 - условный максимум давления
        # 700 - условный минимум
        year_pressure[i] = ((780 - 700) * (1-multiplier)) + 700 
    
    year_pressure = np.round(year_pressure, round)
    return year_pressure


def generate_weather(winter_avg:float, summer_avg:float, year:int=2049, save:bool=True, filename:str='weather_data.csv', verbose:bool=False, noise:bool=True, deviation:float=0.25, outliers:float=0.01, nans:float=0.01):
    '''Генерирует данные о погоде за год.'''
    
    t = generate_temperature(winter_avg, summer_avg)
    l = generate_lighting()
    h = generate_humidity(t)
    p = generate_pressure(t, h)
    time = utils._get_time(year)
    
    weather_data = {'datetime': time,
                    'temperature': t,
                    'lighting': l,
                    'humidity': h,
                    'pressure': p}
    
    df = pd.DataFrame(weather_data)
    
    if noise:
        df = utils._add_noise(df, deviation=deviation, outliers=outliers, nans=nans)
    
    if verbose:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.plot(t)
        ax2.plot(l)
        ax3.plot(h)
        ax4.plot(p)
        plt.show()
        
    if save:
        assert(filename.endswith('.csv'))
        df.to_csv(filename, index=False)
    
    return df
