# Import Library.
import numpy as np
import pandas as pd
import eeglib
import mne
from statsmodels.stats.weightstats import ztest
from tabulate import tabulate

# Sample Rate
SR = 200
# Number of Channels
NCH = 3
# Number of Classes
NCL = 5
# Upper bound frequency of the normal EEG signal, used in band pass filter
UpperFreq = 80
# Lower bound frequency of the normal EEG signal, used in band pass filter
LowerFreq = 0.5


def load_data(path_vy, path_vn, path_ty, path_tn, path_ne):
    # Reading data from files.
    vy = np.loadtxt(path_vy, dtype=(NCH - 1) * 'f, ' + 'f', comments='%', delimiter=',',
                    usecols=(tuple(range(1, NCH + 1))))
    vn = np.loadtxt(path_vn, dtype=(NCH - 1) * 'f, ' + 'f', comments='%', delimiter=',',
                    usecols=(tuple(range(1, NCH + 1))))
    ty = np.loadtxt(path_ty, dtype=(NCH - 1) * 'f, ' + 'f', comments='%', delimiter=',',
                    usecols=(tuple(range(1, NCH + 1))))
    tn = np.loadtxt(path_tn, dtype=(NCH - 1) * 'f, ' + 'f', comments='%', delimiter=',',
                    usecols=(tuple(range(1, NCH + 1))))
    ne = np.loadtxt(path_ne, dtype=(NCH - 1) * 'f, ' + 'f', comments='%', delimiter=',',
                    usecols=(tuple(range(1, NCH + 1))))

    # Converting array of tuples to 2-d array.
    t = []
    for i in [*vy[:]]:
        t.append([*i])
    vy = np.array(t).astype('f')

    t = []
    for i in [*vn[:]]:
        t.append([*i])
    vn = np.array(t).astype('f')

    t = []
    for i in [*ty[:]]:
        t.append([*i])
    ty = np.array(t).astype('f')

    t = []
    for i in [*tn[:]]:
        t.append([*i])
    tn = np.array(t).astype('f')

    t = []
    for i in [*ne[:]]:
        t.append([*i])
    ne = np.array(t).astype('f')

    return vy, vn, ty, tn, ne


def mov_avg(arr):
    """
    Moving Average filter to Reduce Noise.
    """
    # Window Size
    w = 7
    avg = np.array([])
    
    # Compute Average and fill list
    for i in range(NCH):
        avg = np.append(avg, pd.Series(arr[:, i]).rolling(window=w).mean())
    avg = avg.reshape(arr.shape)
    
    #   As Moving Average filter reduces the size of the list, in the elements 
    #   that do not have a value, we consider the average of the neighbors.
    for i in range(avg.shape[0]):
        for j in range(avg.shape[1]):
            if np.isnan(avg[i, j]):
                if i == 0:
                    avg[i, j] = round((arr[i, j] + arr[i + 1, j]) / 2, 2)
                else:
                    avg[i, j] = round((arr[i - 1, j] + arr[i, j] + arr[i + 1, j]) / 3, 2)
    return avg


def plot_psd(arr):
    # Loading Data to MNE Library
    raw = mne.io.RawArray(arr,
                          mne.create_info(['Vocalized Yes', 'Vocalized No', 
                                           'Thinking Yes', 'Thinking No', 'Neutral'],
                                          SR, ch_types='misc'))
    # City electricity frequency in US
    freqs = (60, 80)
    
    # Applying Notch filter to remove city electricity frequency
    raw_notch_fit = raw.copy().notch_filter(
        freqs=freqs, picks='all', method='spectrum_fit', filter_length='8s')
    
    # Applying Band Pass Filter
    raw_notch_fit.filter(l_freq=LowerFreq, h_freq=UpperFreq, picks='misc')

    # Plotting Data in The Frequency Domain
    fig = raw_notch_fit.compute_psd(fmax=80, picks='misc').plot(amplitude=False, show=False, picks='misc')
    
    # Styling figure
    x = fig.get_axes()[0]
    l = x.get_lines()[0]
    l.remove()
    l = x.get_lines()[0]
    l.remove()
    color = ['red', 'darkgreen', 'navy', 'blueviolet', 'gold']
    for l, c in zip(x.get_lines(), color):
        l.set_linewidth(1)
        l.set_color(c)
    fig.legend(['Vocalized Yes', 'Vocalized No', 'Thinking Yes', 'Thinking No', 'Neutral'], loc='upper right')
    fig.show()

    # Plotting Data in The Time Domain
    raw_notch_fit.plot(duration=100, clipping=0.5, scalings=dict(misc=4000000e-6))
    
    # Return Data in the Time and Frequency Domain
    return raw_notch_fit.get_data(picks='misc'), raw_notch_fit.compute_psd(fmax=80, picks='misc').get_data(picks='misc')

# paths to the data of different classes (please change it accordingly)
path_vocalized_yes = 'DATA\OpenBCI-RAW-2023-04-01_16-30-03-Vocalized-Yes.txt'
path_vocalized_no = 'DATA\OpenBCI-RAW-2023-04-01_16-32-04-Vocalized-No.txt'
path_thinking_yes = 'DATA\OpenBCI-RAW-2023-04-01_16-31-03-Thinking-Yes.txt'
path_thinking_no = 'DATA\OpenBCI-RAW-2023-04-01_16-33-19-Thinking-No.txt'
path_neutral = 'DATA\OpenBCI-RAW-2023-04-01_16-29-03-Neutral.txt'
vocalized_yes, vocalized_no, thinking_yes, thinking_no, neutral = load_data(path_vocalized_yes, path_vocalized_no,
                                                                            path_thinking_yes, path_thinking_no,
                                                                            path_neutral)

# Applying moving average filter
vocalized_yes = mov_avg(vocalized_yes).reshape(NCH, vocalized_yes.shape[0])
vocalized_no = mov_avg(vocalized_no).reshape(NCH, vocalized_no.shape[0])
thinking_yes = mov_avg(thinking_yes).reshape(NCH, thinking_yes.shape[0])
thinking_no = mov_avg(thinking_no).reshape(NCH, thinking_no.shape[0])
neutral = mov_avg(neutral).reshape(NCH, neutral.shape[0])

# Concatinating all classes together 
# Because the data had different sizes, we considered the minimum values 
# of the data sizes (that for the current data was 5980).
min_data_row = min(vocalized_yes.shape[1], vocalized_no.shape[1], thinking_yes.shape[1], thinking_no.shape[1], neutral.shape[1])
data = np.zeros((NCL, NCH * min_data_row))
data[0] = np.concatenate(([vocalized_yes[i, :min_data_row] for i in range(NCH)]))
data[1] = np.concatenate(([vocalized_no[i, :min_data_row] for i in range(NCH)]))
data[2] = np.concatenate(([thinking_yes[i, :min_data_row] for i in range(NCH)]))
data[3] = np.concatenate(([thinking_no[i, :min_data_row] for i in range(NCH)]))
data[4] = np.concatenate(([neutral[i, :min_data_row] for i in range(NCH)]))

# Loading data to eeglib library to normalize and removing artifacts (like ECG)
hd = eeglib.helpers.Helper(data, sampleRate=SR, lowpass=UpperFreq, highpass=LowerFreq, normalize=True, ICA=True)
d = hd.data
td, fd = plot_psd(d)

# Computing p-values (of z-test) for different classes in both time and frequency domains
res = []
for i in range(5):
    for j in range(5):
        res.append(
            [str(i) + ' ' + str(j), f'{ztest(td[i, :], td[j, :])[1]:.3f}', f'{ztest(fd[i, :], fd[j, :])[1]:.3f}'])
print(tabulate(res, headers=['Between', 'Time Domain', 'Freq Domain']))
