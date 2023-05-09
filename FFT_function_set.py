import sys
import numpy as np
import pylab as pl
from numpy import fft
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from dateutil.relativedelta import relativedelta

def fourier_transfer(n_harm, stock, date_0, date_1, date_2):
    data = yf.Ticker(stock).history(start=date_0, end=date_1)
    price = data['Close']
    x = np.array(price)
    data2 = yf.Ticker(stock).history(start=date_0, end=date_2)
    price2 = data2['Close']
    n_x = x.size
    t = np.arange(0, n_x)
    p = np.polyfit(t, x, 1)     # find linear trend in x
    x_notrend = x - p[0] * t    # detrended x
    x_freqdom = fft.fft(x_notrend, n=n_x)
    f = fft.fftfreq(n_x)
    indexes = list(range(n_x))      # frequencies
    # sort method 1
    # indexes.sort(key = lambda i: np.absolute(f[i]))     # sort indexes by frequency, lower -> higher
    # sort method 2
    # sort indexes by amplitudes, lower -> higher
    indexes.sort(key=lambda i: np.absolute(x_freqdom[i]))
    indexes.reverse()       # sort indexes by amplitudes, higher -> lower
    t = np.arange(0, price2.size)
    restored_sig = np.zeros(t.size)
    count = 0
    for i in indexes[1:1 + n_harm * 2]:
        if (count <= 100 or count >= (n_harm * 2 - 2)):
            if (count % 2 == 0):
                ampli = np.absolute(x_freqdom[i]) / n_x     # amplitude
                phase = np.angle(x_freqdom[i])      # phase
                signal = ampli * np.cos(2 * np.pi * f[i] * t + phase)
                restored_sig += signal
            count += 1
    signal = restored_sig
    predict = pd.DataFrame()
    predict.index = price2.index
    predict['Close'] = signal
    # predict = predict.drop(predict[predict.index < date_1].index)
    return predict

def peak_valleys(pv_range, data):
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

def peak_valleys_delay(pv_1, pv_2):
    import datetime
    predict_p_1 = pd.DataFrame()
    predict_p_1['peaks_1'] = pv_1['peaks']
    predict_p_1 = predict_p_1.dropna(how='all')
    li_peak_1 = list(predict_p_1['peaks_1'].index)
    predict_p_2 = pd.DataFrame()
    predict_p_2['peaks_2'] = pv_2['peaks']
    predict_p_2 = predict_p_2.dropna(how='all')
    li_peak_2 = list(predict_p_2['peaks_2'].index)
    li_peak = []
    for i in range(0, len(li_peak_2)):
        temp = []
        temp_abs = []
        temp2 = []
        for j in range(0, len(li_peak_1)):
            temp.append((li_peak_1[j] - li_peak_2[i]).days)
            temp_abs.append(abs(li_peak_1[j] - li_peak_2[i]).days)
        for k in range(0, len(temp_abs)):
            if temp_abs[k] == min(temp_abs):
                temp2 = round(temp[k])
        li_peak.append(temp2)
    predict_p_2['delay'] = li_peak
    predict_v_1 = pd.DataFrame()
    predict_v_1['valleys_1'] = pv_1['valleys']
    predict_v_1 = predict_v_1.dropna(how='all')
    li_valley_1 = list(predict_v_1['valleys_1'].index)
    predict_v_2 = pd.DataFrame()
    predict_v_2['valleys_2'] = pv_2['valleys']
    predict_v_2 = predict_v_2.dropna(how='all')
    li_valley_2 = list(predict_v_2['valleys_2'].index)
    li_valley = []
    for i in range(0, len(li_valley_2)):
        temp = []
        temp_abs = []
        temp2 = []
        for j in range(0, len(li_valley_1)):
            temp.append((li_valley_1[j] - li_valley_2[i]).days)
            temp_abs.append(abs(li_valley_1[j] - li_valley_2[i]).days)
        for k in range(0, len(temp_abs)):
            if temp_abs[k] == min(temp_abs):
                temp2 = round(temp[k])
        li_valley.append(temp2)
    predict_v_2['delay'] = li_valley
    return predict_p_2['delay'], predict_v_2['delay']

def draw_plot_1(data, predict):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    axes[0].plot(data.index, data['Close'],
                 c='gray', label='data', linewidth=3)
    axes[1].plot(predict.index, predict['Close'], c='gray', label='Predict', linewidth=3)
    try:
        axes[0].plot(data.index, data['peaks'], '^',
                     c='royalblue', label='peaks')
        axes[0].plot(data.index, data['valleys'], 'v',
                     c='orangered', label='valleys')
        axes[1].plot(predict.index, predict['peaks'], '^', c='royalblue', label='peaks')
        axes[1].plot(predict.index, predict['valleys'], 'v',
                     c='orangered', label='valleys')
    except:
        pass
    try:
        for i, label in enumerate(predict['peaks_delay']):
            axes[1].annotate(label, (predict['peaks'].index[i],
                             predict['peaks'][i]), fontsize=14)
        for i, label in enumerate(predict['valleys_delay']):
            axes[1].annotate(label, (predict['valleys'].index[i],
                             predict['valleys'][i]), fontsize=14)
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

def fit_error(predict, type):
    if type == 'mean':
        predict_temp = predict.drop(predict[predict['peaks_delay'].isna()].index)
        error_p = predict_temp['peaks_delay'].mean()
        predict_temp = predict.drop(predict[predict['valleys_delay'].isna()].index)
        error_v = predict_temp['valleys_delay'].mean()
    elif type == 'abs':
        predict_temp = predict.drop(predict[predict['peaks_delay'].isna()].index)
        error_p = abs(predict['peaks_delay']).mean()
        predict_temp = predict.drop(predict[predict['valleys_delay'].isna()].index)
        error_v = abs(predict['valleys_delay']).mean()
    else:
        return 'wrong type'
    return error_p, error_v

def get_first_delay(predict):
    predict2 = pd.DataFrame()
    predict2['peaks_delay'] = predict['peaks_delay']
    predict2['valleys_delay'] = predict['valleys_delay']
    predict2 = predict2.dropna(how='all')
    if np.isnan(predict2['peaks_delay'].iloc[0]) == False:
        Date = predict2['peaks_delay'].index[0]
        delay = predict2['peaks_delay'].iloc[0]
        pv = 'peaks'
    else:
        Date = predict2['valleys_delay'].index[0]
        delay = predict2['valleys_delay'].iloc[0]
        pv = 'valleys'
    # Date = datetime.datetime.strftime(Date,'%Y-%m-%d')
    return Date, delay, pv

def main_function_1(stock, date_0, date_1, date_2, n_harm, pv_range, type):
    data = yf.Ticker(stock).history(start=date_0, end=date_2)
    data['peaks'] = peak_valleys(pv_range, data)[0]
    data['valleys'] = peak_valleys(pv_range, data)[1]
    predict = fourier_transfer(n_harm, stock, date_0, date_1, date_2)
    predict['peaks'] = peak_valleys(pv_range, predict)[0]
    predict['valleys'] = peak_valleys(pv_range, predict)[1]
    predict['peaks_delay'] = peak_valleys_delay(
        data[data.index <= date_1], predict[predict.index <= date_1])[0]
    predict['valleys_delay'] = peak_valleys_delay(
        data[data.index <= date_1], predict[predict.index <= date_1])[1]
    error = fit_error(predict, type)
    error = (error[0] + error[1])/2
    predict['peaks_delay'] = peak_valleys_delay(data, predict)[0]
    predict['valleys_delay'] = peak_valleys_delay(data, predict)[1]
    return data, predict, error

def main_function_slide(stock, date_start, data_range, predict_range, n_harm, pv_range, n_slide):
    date_start = datetime.datetime.strptime(date_start, '%Y-%m-%d')
    date_0 = date_start + relativedelta(months=-data_range)
    predict = pd.DataFrame()
    data = pd.DataFrame()
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    for i in range(n_slide):
        date_1 = date_0 + relativedelta(months=+data_range)
        date_2 = date_1 + relativedelta(months=+predict_range)
        temp = fourier_transfer(n_harm, stock, date_0, date_1, date_2)
        temp = temp.drop(temp[temp.index < date_1].index)
        axes[1].plot(temp.index, temp['Close'], label='Predict', linewidth=3)
        temp['peaks'] = peak_valleys(pv_range, temp)[0]
        temp['valleys'] = peak_valleys(pv_range, temp)[1]
        data_temp = yf.Ticker(stock).history(start=date_1, end=date_2)
        axes[0].plot(data_temp.index, data_temp['Close'],
                     linewidth=3, label='Test_data')
        data_temp['peaks'] = peak_valleys(pv_range, data_temp)[0]
        data_temp['valleys'] = peak_valleys(pv_range, data_temp)[1]
        temp['peaks_delay'] = peak_valleys_delay(data_temp, temp)[0]
        temp['valleys_delay'] = peak_valleys_delay(data_temp, temp)[1]
        axes[0].plot(data_temp.index, data_temp['peaks'],
                     '^', c='royalblue', label='peaks')
        axes[0].plot(data_temp.index, data_temp['valleys'],
                     'v', c='orangered', label='valleys')
        axes[1].plot(temp.index, temp['peaks'], '^',
                     c='royalblue', label='peaks')
        axes[1].plot(temp.index, temp['valleys'], 'v',
                     c='orangered', label='valleys')
        predict = pd.concat([predict, temp])
        data = pd.concat([data, data_temp])
        date_0 = date_0 + relativedelta(months=+1)
    for i, label in enumerate(predict['peaks_delay']):
        axes[1].annotate(label, (predict['peaks'].index[i],
                         predict['peaks'][i]), fontsize=14)
    for i, label in enumerate(predict['valleys_delay']):
        axes[1].annotate(label, (predict['valleys'].index[i],
                         predict['valleys'][i]), fontsize=14)
    axes[0].set_ylabel("price", fontsize=14)
    axes[0].grid(True)
    axes[1].grid(True)
    axes[1].set_ylabel("amplitude", fontsize=14)
    plt.show()
    return data, predict

def main_function_1_fit_error(stock, date_0, date_1, date_2, n_harm_0, n_harm_1, pv_range, type):
    error = pd.Series(dtype='float64')
    predict = pd.Series(dtype='float64')
    for i in range(n_harm_0, n_harm_1+1):
        temp = main_function_1(stock, date_0, date_1,
                               date_2, i, pv_range, type)
        error = pd.concat([error, pd.Series(temp[2])])
    # print(predict)
    error = error.reset_index(drop=True)
    error = error.abs()
    best_fit = error.idxmin() + n_harm_0
    temp_2 = main_function_1(stock, date_0, date_1,
                             date_2, best_fit, pv_range, type)
    data, predict, error = temp_2
    return data, predict, error, best_fit

def main_function_slide_fit_error(stock, date_start, data_range, predict_range, n_harm_0, n_harm_1, pv_range, n_slide, type):
    date_start = datetime.datetime.strptime(date_start, '%Y-%m-%d')
    date_0 = date_start + relativedelta(months=-data_range)
    predict = pd.DataFrame()
    data = pd.DataFrame()
    fit_error = []
    best_fit = []
    slide_error = []
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    for i in range(n_slide):
        date_1 = date_0 + relativedelta(months=+data_range)
        date_2 = date_1 + relativedelta(months=+predict_range)
        data_drop, temp, error_temp, best_fit_temp = main_function_1_fit_error(
            stock, date_0, date_1, date_2, n_harm_0, n_harm_1, 2, type)
        fit_error.append(round(error_temp, 2))
        best_fit.append(best_fit_temp)
        temp = temp.drop(temp[temp.index < date_1].index)
        axes[1].plot(temp.index, temp['Close'], label='Predict', linewidth=3)
        temp['peaks'] = peak_valleys(pv_range, temp)[0]
        temp['valleys'] = peak_valleys(pv_range, temp)[1]
        data_temp = yf.Ticker(stock).history(start=date_1, end=date_2)
        axes[0].plot(data_temp.index, data_temp['Close'],
                     linewidth=3, label='Test_data')
        data_temp['peaks'] = peak_valleys(pv_range, data_temp)[0]
        data_temp['valleys'] = peak_valleys(pv_range, data_temp)[1]
        temp['peaks_delay'] = peak_valleys_delay(data_temp, temp)[0]
        temp['valleys_delay'] = peak_valleys_delay(data_temp, temp)[1]
        predict_temp_error = pd.concat([temp['peaks_delay'], temp['valleys_delay']])
        try:
            slide_error.append(predict_temp_error.dropna()[0])
        except:
            pass
        axes[0].plot(data_temp.index, data_temp['peaks'],
                     '^', c='royalblue', label='peaks')
        axes[0].plot(data_temp.index, data_temp['valleys'],
                     'v', c='orangered', label='valleys')
        axes[1].plot(temp.index, temp['peaks'], '^',
                     c='royalblue', label='peaks')
        axes[1].plot(temp.index, temp['valleys'], 'v',
                     c='orangered', label='valleys')
        predict = pd.concat([predict, temp])
        data = pd.concat([data, data_temp])
        date_0 = date_0 + relativedelta(months=+predict_range)
    for i, label in enumerate(predict['peaks_delay']):
        axes[1].annotate(label, (predict['peaks'].index[i],
                         predict['peaks'][i]), fontsize=14)
    for i, label in enumerate(predict['valleys_delay']):
        axes[1].annotate(label, (predict['valleys'].index[i],
                         predict['valleys'][i]), fontsize=14)
    axes[0].set_ylabel("price", fontsize=14)
    axes[0].grid(True)
    axes[1].grid(True)
    axes[1].set_ylabel("amplitude", fontsize=14)
    final_error = round(sum([abs(ele)
                        for ele in slide_error]) / len(slide_error), 2)
    plt.show()
    return data, predict, fit_error, best_fit, final_error

def main_function_slide_fit_error_slide_range(stock, date_start, data_range, slide_range, n_slide, pv_range, n_harm_0, n_harm_1, type):
    predict_range = data_range
    date_start = datetime.datetime.strptime(date_start, '%Y-%m-%d')
    date_0 = date_start + relativedelta(months=-data_range)
    predict = pd.DataFrame()
    data = pd.DataFrame()
    fit_error = []
    best_fit = []
    predict_delay = pd.DataFrame(columns=['Date', 'delay', 'pv'])
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    for i in range(n_slide):
        date_1 = date_0 + relativedelta(months=+data_range)
        date_2 = date_1 + relativedelta(months=+predict_range)
        data_drop, temp, error_temp, best_fit_temp = main_function_1_fit_error(
            stock, date_0, date_1, date_2, n_harm_0, n_harm_1, pv_range, type)
        fit_error.append(round(error_temp, 2))
        best_fit.append(best_fit_temp)
        temp = temp.drop(temp[temp.index < date_1].index)
        temp['peaks'] = peak_valleys(pv_range, temp)[0]
        temp['valleys'] = peak_valleys(pv_range, temp)[1]
        data_temp = yf.Ticker(stock).history(start=date_1, end=date_2)
        data_temp['peaks'] = peak_valleys(pv_range, data_temp)[0]
        data_temp['valleys'] = peak_valleys(pv_range, data_temp)[1]
        temp['peaks_delay'] = peak_valleys_delay(data_temp, temp)[0]
        temp['valleys_delay'] = peak_valleys_delay(data_temp, temp)[1]
        temp = temp.drop(columns='Close')
        predict_delay.loc[i, 'Date'], predict_delay.loc[i,
                                              'delay'], predict_delay.loc[i, 'pv'] = get_first_delay(temp)
        predict = pd.concat([predict[predict.index < temp.index[0]], temp])
        date_0 = date_0 + relativedelta(weeks=+slide_range)
    data = yf.Ticker(stock).history(start=date_start, end=date_2)
    axes[0].plot(data.index, data['Close'], 'gray',
                 linewidth=3, label='Test_data')
    data['peaks'] = peak_valleys(pv_range, data)[0]
    data['valleys'] = peak_valleys(pv_range, data)[1]
    axes[0].plot(data.index, data['peaks'], '^', c='royalblue', label='peaks')
    axes[0].plot(data.index, data['valleys'], 'v',
                 c='orangered', label='valleys')
    predict_delay = predict_delay.set_index(predict_delay['Date'])
    predict_delay = predict_delay.drop(columns='Date')
    axes[0].set_ylabel("price", fontsize=14)
    axes[0].grid(True)
    axes[1].grid(True)
    axes[1].set_ylabel("delay", fontsize=14)
    final_error = round(
        sum([abs(ele) for ele in predict_delay['delay']]) / len(predict_delay['delay']), 2)
    a = pd.DataFrame(index=predict.index, columns=['delay', 'pv'])
    a.loc[predict_delay.index] = predict_delay
    axes[0].set_xlim(data.index[0], data.index[-1])
    axes[1].set_xlim(data.index[0], data.index[-1])
    for i, label in enumerate(a['delay']):
        if a['pv'][i] == 'peaks':
            axes[1].plot(a['delay'].index[i], a['delay'][i],
                         '*', c='royalblue', label='peaks')
        else:
            axes[1].plot(a['delay'].index[i], a['delay'][i],
                         '*', c='orangered', label='valleys')
        axes[1].annotate(label, (a['delay'].index[i],
                         a['delay'][i]), fontsize=14)
    plt.show()
    return data, predict, fit_error, best_fit, final_error, predict_delay

slide_abs_test = main_function_slide_fit_error_slide_range(
    stock="^GSPC", date_start='2021-01-01', data_range=6,
    slide_range=2, n_slide=1, pv_range=2, n_harm_0=10, n_harm_1=10, type='abs'
)
print(slide_abs_test[2])
print(slide_abs_test[3])
print(slide_abs_test[4])
print(slide_abs_test[5])
