"""
Functions for analysing csv with processed data. (Processing tif to get csv seperately)
"""

######################################################
# Imports

import os
import sys
import glob
import imageio
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy import stats
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.signal import hilbert
from scipy.signal import gaussian
from scipy.signal import correlate
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt



######################################################
# Utility

def load_df(path, fs, t_cutoff):
    df = pd.read_csv(path)
    return df, fs, t_cutoff


def load_data(i, df):
    name = df[df['id'] == i]['name'].unique()[0]
    arr = df[df['id'] == i][['t', 'FI_corr']].values.T
    sig = arr[1][~np.isnan(arr[1])]  # filter nan
    t = arr[0][~np.isnan(arr[1])]
    return name, t, sig


def load_area(i, df):
    name = df[df['id'] == i]['name'].unique()[0]
    switch = df[df['id'] == i]['switch'].unique()[0]
    arr = df[df['id'] == i][['t', 'area', 'area2']].values.T
    area = arr[1][~np.isnan(arr[1])]  # filter nan
    area2 = arr[2][~np.isnan(arr[2])]  # filter nan
    t = arr[0][~np.isnan(arr[1])]
    t2 = arr[0][~np.isnan(arr[2])]
    return name, t, t2, area, area2, switch


def sig_interp(sig, non_uniform_sampling, t_total, fs):
    """Interpolates (cubic spline) signal sampled at non uniform timepoints at uniform timepoints with given sampling parameters. Normalized. """
    i_cutoff = (np.abs(non_uniform_sampling - t_total)).argmin()
    interp_func = interp1d(non_uniform_sampling, sig,
                           fill_value='extrapolate', kind='cubic')
    sampling = np.linspace(0, t_total, fs*t_total)  # new, uniform sampling
    sig_interp = interp_func(sampling)
    sig_interp = (sig_interp - np.min(sig_interp)) / \
        (np.max(sig_interp) - np.min(sig_interp))
    sig_interp = sig_interp - np.mean(sig_interp)
    return sampling, sig_interp, i_cutoff

def sig_interp_nonorm(sig, non_uniform_sampling, t_total, fs):
    """Interpolates (cubic spline) signal sampled at non uniform timepoints at uniform timepoints with given sampling parameters. Normalized. """
    i_cutoff = (np.abs(non_uniform_sampling - t_total)).argmin()
    interp_func = interp1d(non_uniform_sampling, sig,
                           fill_value='extrapolate', kind='cubic')
    sampling = np.linspace(0, t_total, fs*t_total)  # new, uniform sampling
    sig_interp = interp_func(sampling)
    return sampling, sig_interp, i_cutoff

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter_noshift(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def bootstrap(data, calc, draws):
    out = []
    # main loop
    i = 0
    while i < draws:
        resampled_data = np.random.choice(data, len(data), replace=True)
        out.append(calc(resampled_data))
        i += 1
    return np.array(out)

def bootstrap2(arr, calc, draws):
    indices = np.arange(0, arr.shape[0], 1).astype(int)
    out = []
    # main loop
    i = 0
    while i < draws:
        resampled_indices = np.random.choice(indices, len(indices), replace=True)
        resampled_data = arr[resampled_indices]
        out.append(calc(resampled_data))
        i += 1
    return np.array(out)

######################################################
# Periods

def peak_locations(sig, fs):
    peaks, _ = find_peaks(sig, prominence=0.15, distance=20)
#     peaks = np.zeros_like(peaks_est) # parabolic interpolation
#     for i, p in enumerate(peaks_est):
#         px, py = parabolic(sig, p)
#         peaks[i] = px
    T = peaks[1:] - peaks[:-1]
    return peaks/fs, T/fs

def autocorr(sig):
    #return np.correlate(sig, sig, mode='same')
    result = np.correlate(sig, sig, mode='full')
    return result[int(result.size/2):]

def crosscorr(sig1, sig2):
    result = np.correlate(sig1, sig2, mode='full')
    return result[int(result.size/2):]

### adapted from: https://gist.github.com/endolith/255291
def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def peak_fit(f, x, window=5):
    # initial estimate
    
    
    def fitfunc(x, a, b, c):
        return (a*(x - b)**2 + c)
    p0 = [-1, x, f[x]]
    fit_interval = np.arange(x - window//2, x+window//2+1, 1).astype(int)
    popt, pcov = curve_fit(fitfunc, fit_interval, f[fit_interval], p0=p0)
    
    return (popt[1], np.sqrt(np.diag(pcov))[1])


def freq_from_autocorr(sig, fs):
    """
    Estimate frequency using autocorrelation
    """
    # Calculate autocorrelation and throw away the negative lags
    corr = correlate(sig, sig, mode='full')
    corr = corr[len(corr)//2:]

    # Find the first low point
    d = np.diff(corr)
    start = np.nonzero(d > 0)[0][0]
    
    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = np.argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px

def autocorr_first_peak(autocorr):
    start = np.nonzero(np.diff(autocorr)>0)[0][0]
    peak_i = np.argmax(autocorr[start:]) + start
    return peak_i

######################################################
# Decay

def func(x, a, b, c, d):
    return a*np.exp(-b*(x-c)) + d


def peak_decay(peaks, T, time, sig, i):
    # fit exponential decay to peaks
    # parameters: relevant time interval, peak distance
    
    peaks = np.append(peaks, time[-1]) # add experiment end time to calculate time difference with last detected peak 
    dec = []
    rsq = []
    intervals = []
    for j, p in enumerate(peaks[:-1]): # loop over real peaks
        # check time difference to next peak
        if peaks[j+1]-p > 6:
            interval = [p+1,p+4.5]
            cond = (interval[0]<time) & (time<interval[1])
            xdata = time[cond]
            ydata = sig[cond]
            p0 = [(ydata[0]-ydata[-1]), 1, xdata[0], ydata[-1]]
            try:
                popt, pcov = curve_fit(func, xdata, ydata, p0)
                dec.append(popt[1])
                rsq.append(rsquared(ydata, func(xdata, *popt)))
                intervals.append(interval)
            except:
                print(f'No fit for id {i} peak {j}')
    return np.array(dec), np.array(rsq), np.array(intervals)


def sumSquaresTot(y):
    ymean = np.mean(y)
    s = 0
    for i in range(len(y)):
        s += np.power(y[i]-ymean, 2)
    return s


def sumResiduals(y, f):
    s = 0
    for i in range(len(y)):
        s += np.power(y[i]-f[i], 2)
    return s


def rsquared(y, f):
    return (1-sumResiduals(y, f)/sumSquaresTot(y))

######################################################
# Growth

def div_events(time, area):
    t = np.array(time)
    a = np.array(area)
    a_diff = np.diff(a)

    perc_change = a_diff/a[:-1]
    # detect
    condition = (perc_change < -0.3) & (perc_change > -0.8)
    # option: limit to single spikes
    div = t[:-1][condition]
    return div


def growth_rate(time, div, fs):
    num_div = np.zeros_like(time)
    for t in div:
        index = np.argwhere(time == t)[0][0]
        num_div[index:] += 1
    num_div_smooth = savgol_filter(num_div, 17, 1)

    g_rate = savgol_filter(np.diff(num_div_smooth), 11,
                           1)*fs  # doublings per hour
    return g_rate


def filter_irregular_growth(time, area, division_events, growth_rate):
    return regularity

######################################################
# Plot

def subplot_labels_oo(axs):
    for n, ax in enumerate(axs.flat):
        ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform=ax.transAxes, 
                size=12, weight='bold')

def hist_kde(data, title):
    kernel = gaussian_kde(data)
    Trange = np.linspace(np.min(data) - 0.1*np.mean(data), np.max(data) + 0.1*np.mean(data), 1000)
    kde_peak = Trange[kernel(Trange).argmax()]

    #plt.figure(figsize=(5, 3))
    n, b, p = plt.hist(data,  density=True, color='g', alpha=0.6)
    plt.plot(Trange, kernel(Trange), color='g')
    l = "KDE peak: %.2f" % kde_peak
    plt.vlines(kde_peak, 0, 1.1*np.max(n), label=l, linestyles='dashed')
    plt.legend()
    #plt.ylabel('Probability')
    plt.title(title)
    
def hist_kde_oo(ax, data, title):
    kernel = gaussian_kde(data)
    Trange = np.linspace(np.min(data) - 0.1*np.mean(data), np.max(data) + 0.1*np.mean(data), 1000)
    kde_peak = Trange[kernel(Trange).argmax()]

    #plt.figure(figsize=(5, 3))
    n, b, p = ax.hist(data,  density=True, color='g', alpha=0.6)
    ax.plot(Trange, kernel(Trange), color='g')
    l = "KDE peak: %.2f" % kde_peak
    ax.vlines(kde_peak, 0, 1.1*np.max(n), label=l, linestyles='dashed')
    ax.legend()
    #plt.ylabel('Probability')
    ax.set_title(title)

def plot_g_rate(div, g_rate, g_rate2, i):
    plt.figure(figsize=(20,5))
    plt.subplot(122)
    plt.vlines(div, ymin = 0, ymax = 2, alpha = 0.5, label=len(div))
    plt.plot(t[:-1], g_rate, label='growth rate 1')
    plt.plot(t2[:-1], g_rate2, label='growth rate 2')
    plt.legend()

    plt.subplot(121)
    name, ts, sig = load_data(i, df)
    plt.plot(ts, sig)

def plot_freq_hilbert(signal):
    s = signal - np.mean(signal)
    
    plt.figure(figsize=(8,8))

    plt.subplot(411)
    plt.title("Signal")
    plt.plot(s, label='s')
    plt.plot(np.abs(s + j*hilbert(s)), label='abs(s + i*s_H)')
    plt.legend()

    plt.subplot(412)
    plt.title("Phase")
    phase = np.angle(hilbert(s))
    plt.plot(phase)

    plt.subplot(413)
    plt.title("Phase unwrapped")
    plt.plot(np.unwrap(phase))

    plt.subplot(414)
    plt.title("Frequency")
    freq = np.diff(np.unwrap(phase))
    omega = np.mean(freq)
    period = (2*np.pi)/omega

    plt.plot(freq, label=f'T = {period/6:.3g} h')
    plt.legend()
    plt.hlines(np.mean(freq), 0, freq.shape[0])

    plt.tight_layout()
    #plt.show()
    plt.savefig('pics/2_hilbert_21.png', dpi=400)


def plot_hilbert_phase_space(s_smooth, s_analytical, i):
    plt.figure(figsize=(5,5))
    plt.plot(s_smooth, np.imag(s_analytical))
    plt.xlabel('sig')
    plt.ylabel('sig_H')
    plt.plot(0, 0, '.', ms=20)
    plt.title(i)

def plot_dot_circle(angles, uangles):
    plt.figure(figsize=(8,8))
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    plt.plot(0, 0, 'kx')
    av_angle = np.mean(uangles)
    plt.plot(np.cos(av_angle), np.sin(av_angle), 'ro', ms=15)
    for i in range(angles.shape[0]):
        plt.plot(np.cos(angles[i]), np.sin(angles[i]), 'x', ms=10)
    
def plot_circle_movie(TODO):
    for i in range(phases0.shape[1]):
        plot_dot_circle(phases0[:,i], uphases0[:,i])
        path = '../200114_proc/5_data/animation/200114_circle_' + f'{i:03}' + '.png'
        plt.title(f'{t[i]:.2g} h')
        plt.savefig(path, dpi=100)
        plt.close('all')

    path = '../200114_proc/5_data/animation/'
    filenames = [i for i in glob.glob(os.path.join(path, '*.png'))]

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    gif_path = path + 'movie.gif'
    imageio.mimsave(gif_path, images)