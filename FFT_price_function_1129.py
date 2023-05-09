# %% [markdown]
# # Import packages
# 

# %%
import numpy as np
import pylab as pl
from numpy import fft
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from dateutil.relativedelta import relativedelta


# %%
def fourier_transfer_function(
    n_harm, stock_name, date_data_start, date_transfer_start, date_transfer_end):

    # get data_stock's infomation
    date_data_end = date_transfer_start
    data_stock = yf.Ticker(stock_name).history(
        start=date_data_start, end=date_data_end)['Close']
    array_data = np.array(data_stock)
    n_data = array_data.size
    time_data = np.arange(0, n_data)

    # detrend data
    # find linear trend in data
    Polynomial = np.polyfit(time_data, array_data, 1)
    data_notrend = array_data - Polynomial[0] * time_data    # detrended x

    # fft process
    data_freqdom = fft.fft(data_notrend, n=n_data)
    frequence = fft.fftfreq(n_data)
    f_positive = frequence[np.where(frequence > 0)]
    data_freqdom_positive = data_freqdom[np.where(frequence > 0)]

    # sort indexes
    indexes = list(range(f_positive.size))      # frequencies
    # sort method 1
    # indexes.sort(key = lambda i: np.absolute(frequence[i]))     # sort indexes by frequency, lower -> higher
    # sort method 2 :
    # sort indexes by amplitudes, lower -> higher
    indexes.sort(key=lambda i: np.absolute(data_freqdom[i]))
    indexes.reverse()       # sort indexes by amplitudes, higher -> lower

    # get data_all_time'size
    data_all_time = yf.Ticker(stock_name).history(
        start=date_data_start, end=date_transfer_end)['Close']
    time_transfer = np.arange(0, data_all_time.size)
    mixed_harmonic = np.zeros(data_all_time.size)

    # mix harmonics
    for i in indexes[:n_harm]:
        ampli = np.absolute(data_freqdom_positive[i]) / n_data     # amplitude
        phase = np.angle(data_freqdom_positive[i])      # phase
        harmonic = ampli * \
            np.cos(2 * np.pi * f_positive[i] * time_transfer + phase)
        mixed_harmonic += harmonic

    processed_signal = pd.DataFrame(
        {'Close': mixed_harmonic}, index=data_all_time.index)
    return processed_signal


# %%
# fourier_transfer_function(20, "^GSPC", '2021-01-01', '2022-01-01', '2022-02-01')


# %%
def find_pv_function(pv_range, data):
    pd.options.mode.chained_assignment = None
    pv = data['Close']
    data['peaks'] = pd.Series(dtype='float64')
    data['valleys'] = pd.Series(dtype='float64')
    peaks = data['peaks']
    valleys = data['valleys']
    for idx in range(0, len(pv)):
        if pv[idx] == pv.iloc[idx-pv_range:idx+pv_range].max():
            peaks.iloc[idx] = pv[idx]
        if pv[idx] == pv.iloc[idx-pv_range:idx+pv_range].min():
            valleys.iloc[idx] = pv[idx]
    return peaks, valleys

# %%
def find_pv_delay_function(data, processed_signal):
    if (data.index[0] == processed_signal.index[0] and data.index[-1] == processed_signal.index[-1]):
        p_data = pd.DataFrame(
            {'peaks': data['peaks'], 'count': range(len(data))})
        p_data = p_data.drop(p_data[p_data['peaks'].isna()].index)
        p_data_count = list(p_data['count'])
        p_signal = pd.DataFrame(
            {'peaks': processed_signal['peaks'], 'count': range(len(processed_signal))})
        p_signal = p_signal.drop(
            p_signal[p_signal['peaks'].isna()].index)
        p_signal_list = list(p_signal['count'])
        p_delay = []
        for i in range(0, len(p_signal_list)):
            temp = []
            temp_abs = []
            temp_2 = []
            for j in range(0, len(p_data_count)):
                temp.append((p_data_count[j] - p_signal_list[i]))
                temp_abs.append(abs(p_data_count[j] - p_signal_list[i]))
            for k in range(0, len(temp_abs)):
                if temp_abs[k] == min(temp_abs):
                    temp_2 = temp[k]
            p_delay.append(temp_2)
        p_signal['delay'] = p_delay

        v_data = pd.DataFrame(
            {'valleys': data['valleys'], 'count': range(len(data))})
        v_data = v_data.drop(v_data[v_data['valleys'].isna()].index)
        v_data_count = list(v_data['count'])
        v_signal = pd.DataFrame(
            {'valleys': processed_signal['valleys'], 'count': range(len(processed_signal))})
        v_signal = v_signal.drop(
            v_signal[v_signal['valleys'].isna()].index)
        v_signal_list = list(v_signal['count'])
        v_delay = []
        for i in range(0, len(v_signal_list)):
            temp = []
            temp_abs = []
            temp_2 = []
            for j in range(0, len(v_data_count)):
                temp.append((v_data_count[j] - v_signal_list[i]))
                temp_abs.append(abs(v_data_count[j] - v_signal_list[i]))
            for k in range(0, len(temp_abs)):
                if temp_abs[k] == min(temp_abs):
                    temp_2 = temp[k]
            v_delay.append(temp_2)
        v_signal['delay'] = v_delay
        return p_signal['delay'], v_signal['delay']
    else:
        print('error : data = ', data.index,
              'processed_signal = ', processed_signal.index)

# %%
def draw_plot(data, processed_signal):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    axes[0].plot(data.index, data['Close'],
                 c='gray', label='data', linewidth=3)
    axes[1].plot(processed_signal.index, processed_signal['Close'],
                 c='gray', label='Predict', linewidth=3)
    try:
        axes[0].plot(data.index, data['peaks'], '^',
                     c='royalblue', label='peaks')
        axes[0].plot(data.index, data['valleys'], 'v',
                     c='orangered', label='valleys')
        axes[1].plot(processed_signal.index, processed_signal['peaks'],
                     '^', c='royalblue', label='peaks')
        axes[1].plot(processed_signal.index, processed_signal['valleys'], 'v',
                     c='orangered', label='valleys')
    except:
        pass
    try:
        for i, label in enumerate(processed_signal['peaks_delay']):
            axes[1].annotate(label, (processed_signal['peaks'].index[i],
                             processed_signal['peaks'][i]), fontsize=14)
        for i, label in enumerate(processed_signal['valleys_delay']):
            axes[1].annotate(label, (processed_signal['valleys'].index[i],
                             processed_signal['valleys'][i]), fontsize=14)
    except:
        pass
    axes[0].set_ylabel("price", fontsize=14)
    axes[0].grid(True)
    axes[1].grid(True)
    axes[1].set_ylabel("amplitude", fontsize=14)
    axes[0].legend()
    axes[1].legend()
    plt.show()
    return

# %%
def get_fit_error_function(processed_signal, fit_method):
    signal_p_dropna = processed_signal.drop(
        processed_signal[processed_signal['peaks_delay'].isna()].index)
    signal_v_dropna = processed_signal.drop(
        processed_signal[processed_signal['valleys_delay'].isna()].index)
    if fit_method == 'mean':
        error_p = signal_p_dropna['peaks_delay'].mean()
        error_v = signal_v_dropna['valleys_delay'].mean()
    elif fit_method == 'abs':
        error_p = abs(processed_signal['peaks_delay']).mean()
        error_v = abs(processed_signal['valleys_delay']).mean()
    else:
        return 'wrong fit_method'
    error = (error_p + error_v)/2
    return error

# %%
def get_first_delay_function(processed_signal):
    temp = pd.DataFrame()
    temp['peaks_delay'] = processed_signal['peaks_delay']
    temp['valleys_delay'] = processed_signal['valleys_delay']
    temp = temp.dropna(how='all')
    if np.isnan(temp['peaks_delay'].iloc[0]) == False:
        Date = temp['peaks_delay'].index[0]
        delay = temp['peaks_delay'].iloc[0]
        pv = 'peaks'
    else:
        Date = temp['valleys_delay'].index[0]
        delay = temp['valleys_delay'].iloc[0]
        pv = 'valleys'
    return Date, delay, pv

# %%
def single_task(
    stock_name, date_data_start, date_predict_start, 
    date_predict_end, n_harm, pv_range, fit_method):

    data = yf.Ticker(stock_name).history(start=date_data_start, end=date_predict_end)
    processed_signal = fourier_transfer_function(n_harm, stock_name, date_data_start, date_predict_start, date_predict_end)
    data['peaks'] = find_pv_function(pv_range, data)[0]
    data['valleys'] = find_pv_function(pv_range, data)[1]
    processed_signal['peaks'] = find_pv_function(pv_range, processed_signal)[0]
    processed_signal['valleys'] = find_pv_function(pv_range, processed_signal)[1]
    processed_signal['peaks_delay'] = find_pv_delay_function(
        data[data.index <= date_predict_start], processed_signal[processed_signal.index <= date_predict_start])[0]
    processed_signal['valleys_delay'] = find_pv_delay_function(
        data[data.index <= date_predict_start], processed_signal[processed_signal.index <= date_predict_start])[1]
    error = get_fit_error_function(processed_signal, fit_method)
    processed_signal['peaks_delay'] = find_pv_delay_function(data, processed_signal)[0]
    processed_signal['valleys_delay'] = find_pv_delay_function(data, processed_signal)[1]
    return processed_signal, error

# %%
def fit_error_task(
    stock_name, date_data_start, date_predict_start, date_predict_end, 
    n_harm_lower_limit, n_harm_upper_limit, pv_range, fit_method):

    errors = pd.Series(dtype='float64')
    for i in range(n_harm_lower_limit, n_harm_upper_limit+1):
        temp = single_task(stock_name, date_data_start, date_predict_start,
                           date_predict_end, i, pv_range, fit_method)
        errors = pd.concat([errors, pd.Series(temp[1])])
    errors = errors.reset_index(drop=True)
    errors = errors.abs()
    best_fit = errors.idxmin() + n_harm_lower_limit
    processed_signal, errors = single_task(stock_name, date_data_start, date_predict_start,
                                    date_predict_end, best_fit, pv_range, fit_method)
    return processed_signal, errors, best_fit

# %%
def main_function(
    stock_name, date_predict_start, data_range, slide_range, n_slide, pv_range, 
    n_harm_lower_limit, n_harm_upper_limit, fit_method):
    
    date_predict_start = datetime.datetime.strptime(date_predict_start, '%Y-%m-%d') # ex.'2021-01-01'
    result_table = pd.DataFrame(
        columns=['Start_Date', 'Target_Date', 'delay', 'pv', 'error', 'best_fit'])
    for i in range(n_slide):
        date_data_start = date_predict_start - relativedelta(months=+data_range) # ex.'2020-07-01'
        date_predict_end = date_predict_start + relativedelta(months=+data_range) # ex.'2021-07-01'
        processed_signal, error, best_fit = fit_error_task(
            stock_name, date_data_start, date_predict_start, date_predict_end, 
            n_harm_lower_limit, n_harm_upper_limit, pv_range, fit_method)
        processed_signal = processed_signal.drop(
            processed_signal[processed_signal.index < date_predict_start].index)
        result_table.loc[i, 'error'] = round(error, 2)
        result_table.loc[i, 'best_fit'] = best_fit
        result_table.loc[i, 'Start_Date'] = date_predict_start # ex.'2021-01-01'
        result_table.loc[i, 'Target_Date'], result_table.loc[i,'delay'], \
            result_table.loc[i, 'pv'] = get_first_delay_function(processed_signal)
        date_data_start = date_data_start + relativedelta(weeks=+slide_range) # ex.'2020-07-15'
        date_predict_start = date_predict_start + relativedelta(weeks=+slide_range) # ex.'2021-01-15'
    final_error = round(
        sum([abs(ele) for ele in result_table['delay']]) / len(result_table['delay']), 2)
    return final_error, result_table

# %%
def draw_plot_result_table(data, final_error, result_table):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    axes[0].plot(data.index, data['Close'], 'gray', label='data', linewidth=3)
    axes[0].plot(data.index, data['peaks'], '^', c='royalblue', label='peaks')
    axes[0].plot(data.index, data['valleys'], 'v',c='orangered', label='valleys')
    for i, label in enumerate(result_table['delay']):
        if result_table['pv'][i] == 'peaks':
            axes[1].plot(result_table['Target_Date'][i],result_table['delay'][i], '*',
                         c='royalblue', label='peaks')
            axes[1].annotate(label, (result_table['Target_Date'][i],
                                     result_table['delay'][i]), fontsize=14)
        else:
            axes[1].plot(result_table['Target_Date'][i], result_table['delay'][i], '*', 
                         c='orangered', label='valleys')
            axes[1].annotate(label, (result_table['Target_Date'][i],
                             result_table['delay'][i]), fontsize=14)
    axes[0].set_ylabel("Stock price", fontsize=14)
    axes[0].grid(True)
    axes[1].grid(True)
    axes[1].set_ylabel("delay", fontsize=14)
    axes[0].set_xlim(data.index[0], data.index[-1])
    axes[1].set_xlim(data.index[0], data.index[-1])
    plt.show()
    return
