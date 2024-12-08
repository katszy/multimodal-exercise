import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# code source: https://github.com/WiIIson/EMGFlow-Python-Package

def CalcIEMG(Signal, col, sr):
    if col not in Signal.columns:
        raise Exception("Column " + col + " not in Signal")
    if sr <= 0:
        raise Exception("Sampling rate cannot be 0 or negative")
    IEMG = np.sum(np.abs(Signal[col]) * (1 / sr))
    return IEMG


def CalcMAV(Signal, col):
    if col not in Signal.columns:
        raise Exception("Column " + col + " not in Signal")
    N = len(Signal[col])
    MAV = np.sum(np.abs(Signal[col])) / N
    return MAV


def CalcMMAV(Signal, col):
    if col not in Signal.columns:
        raise Exception("Column " + col + " not in Signal")
    N = len(Signal[col])
    vals = np.abs(Signal[col]).values
    total = 0
    for n in range(N):
        if 0.25 * N <= n <= 0.75 * N:
            total += vals[n]
        else:
            total += 0.5 * vals[n]
    MMAV = total / N
    return MMAV


def CalcSSI(Signal, col, sr):
    if col not in Signal.columns:
        raise Exception("Column " + col + " not in Signal")
    if sr <= 0:
        raise Exception("Sampling rate cannot be 0 or negative")
    SSI = np.sum((np.abs(Signal[col]) * (1 / sr)) ** 2)
    return SSI


def CalcVAR(Signal, col):
    if col not in Signal.columns:
        raise Exception("Column " + col + " not in Signal")
    N = len(Signal[col])
    VAR = np.var(Signal[col])
    return VAR


def CalcVOrder(Signal, col):
    if col not in Signal.columns:
        raise Exception("Column " + col + " not in Signal")
    vOrder = np.sqrt(CalcVAR(Signal, col))
    return vOrder


def CalcRMS(Signal, col):
    if col not in Signal.columns:
        raise Exception("Column " + col + " not in Signal")
    RMS = np.sqrt(np.mean(Signal[col] ** 2))
    return RMS


def CalcWL(Signal, col):
    if col not in Signal.columns:
        raise Exception("Column " + col + " not in Signal")
    vals = Signal[col].values
    WL = np.sum(np.abs(np.diff(vals)))
    return WL


def CalcLOG(Signal, col):
    if col not in Signal.columns:
        raise Exception("Column " + col + " not in Signal")
    N = len(Signal[col])
    LOG = np.exp((1 / N) * np.sum(np.log(np.abs(Signal[col]) + 1e-10)))
    return LOG


def CalcMFL(Signal, col):
    if col not in Signal.columns:
        raise Exception("Column " + col + " not in Signal")
    vals = Signal[col].values
    diff = np.diff(vals)
    MFL = np.log(np.sqrt(np.sum(diff ** 2)))
    return MFL


def CalcAP(Signal, col):
    if col not in Signal.columns:
        raise Exception("Column " + col + " not in Signal")
    AP = np.mean(Signal[col] ** 2)
    return AP


def CalcTwitchRatio(psd, freq=60):
    """
    Calculate the Twitch Ratio of a PSD.

    Parameters
    ----------
    psd : DataFrame
        A Pandas DataFrame containing a 'Frequency' and 'Power' column.
    freq : float, optional
        Frequency threshold of the Twitch Ratio separating fast-twitching (high-frequency)
        muscles from slow-twitching (low-frequency) muscles.

    Raises
    ------
    Exception
        An exception is raised if freq is less or equal to 0.
    Exception
        An exception is raised if psd does not only have columns 'Frequency' and 'Power'

    Returns
    -------
    twitch_ratio : float
        Twitch Ratio of the PSD.

    """

    if freq <= 0:
        raise Exception("freq cannot be less or equal to 0")

    if set(psd.columns.values) != {'Frequency', 'Power'}:
        raise Exception("psd must be a Power Spectrum Density dataframe with only a 'Frequency' and 'Power' column")

    fast_twitch = psd[psd['Frequency'] > freq]
    slow_twitch = psd[psd['Frequency'] < freq]

    twitch_ratio = np.sum(fast_twitch['Power']) / np.sum(slow_twitch['Power'])

    return twitch_ratio


#
# =============================================================================
#

def CalcTwitchIndex(psd, freq=60):
    """
    Calculate the Twitch Index of a PSD.

    Parameters
    ----------
    psd : DataFrame
        A Pandas DataFrame containing a 'Frequency' and 'Power' column.
    freq : float, optional
        Frequency threshold of the Twitch Index separating fast-twitching (high-frequency)
        muscles from slow-twitching (low-frequency) muscles.

    Raises
    ------
    Exception
        An exception is raised if freq is less or equal to 0.
    Exception
        An exception is raised if psd does not only have columns 'Frequency' and 'Power'

    Returns
    -------
    twitch_index : float
        Twitch Index of the PSD.

    """

    if freq <= 0:
        raise Exception("freq cannot be less or equal to 0")

    if set(psd.columns.values) != {'Frequency', 'Power'}:
        raise Exception("psd must be a Power Spectrum Density dataframe with only a 'Frequency' and 'Power' column")

    fast_twitch = psd[psd['Frequency'] > freq]
    slow_twitch = psd[psd['Frequency'] < freq]

    twitch_index = np.max(fast_twitch['Power']) / np.max(slow_twitch['Power'])

    return twitch_index


#
# =============================================================================
#

def CalcTwitchSlope(psd, freq=60):
    """
    Calculate the Twitch Slope of a PSD.

    Parameters
    ----------
    psd : DataFrame
        A Pandas DataFrame containing a 'Frequency' and 'Power' column.
    freq : float, optional
        Frequency threshold of the Twitch Slope separating fast-twitching (high-frequency)
        muscles from slow-twitching (low-frequency) muscles.

    Raises
    ------
    Exception
        An exception is raised if freq is less or equal to 0.
    Exception
        An exception is raised if psd does not only have columns 'Frequency' and 'Power'

    Returns
    -------
    fast_slope : float
        Twitch Slope of the fast-twitching muscles.
    slow_slope : float
        Twitch Slope of the slow-twitching muscles.

    """

    if freq <= 0:
        raise Exception("freq cannot be less or equal to 0")

    if set(psd.columns.values) != {'Frequency', 'Power'}:
        raise Exception("psd must be a Power Spectrum Density dataframe with only a 'Frequency' and 'Power' column")

    fast_twitch = psd[psd['Frequency'] > freq]
    slow_twitch = psd[psd['Frequency'] < freq]

    x_fast = fast_twitch['Frequency']
    y_fast = fast_twitch['Power']
    A_fast = np.vstack([x_fast, np.ones(len(x_fast))]).T

    x_slow = slow_twitch['Frequency']
    y_slow = slow_twitch['Power']
    A_slow = np.vstack([x_slow, np.ones(len(x_slow))]).T

    fast_alpha = np.linalg.lstsq(A_fast, y_fast, rcond=None)[0]
    slow_alpha = np.linalg.lstsq(A_slow, y_slow, rcond=None)[0]

    fast_slope = fast_alpha[0]
    slow_slope = slow_alpha[0]

    return fast_slope, slow_slope


#
# =============================================================================
#

def CalcSC(psd):
    """
    Calculate the Spectral Centroid (SC) of a PSD.

    Parameters
    ----------
    psd : DataFrame
        A Pandas DataFrame containing a 'Frequency' and 'Power' column.

    Raises
    ------
    Exception
        An exception is raised if psd does not only have columns 'Frequency' and 'Power'

    Returns
    -------
    SC : float
        SC of the PSD.

    """

    if set(psd.columns.values) != {'Frequency', 'Power'}:
        raise Exception("psd must be a Power Spectrum Density dataframe with only a 'Frequency' and 'Power' column")

    SC = np.sum(psd['Power'] * psd['Frequency']) / np.sum(psd['Power'])
    return SC


#
# =============================================================================
#

def CalcSF(psd):
    """
    Calculate the Spectral Flatness (SF) of a PSD.

    Parameters
    ----------
    psd : DataFrame
        A Pandas DataFrame containing a 'Frequency' and 'Power' column.

    Raises
    ------
    Exception
        An exception is raised if psd does not only have columns 'Frequency' and 'Power'

    Returns
    -------
    SF : float
        SF of the PSD.

    """

    if set(psd.columns.values) != {'Frequency', 'Power'}:
        raise Exception("psd must be a Power Spectrum Density dataframe with only a 'Frequency' and 'Power' column")

    N = psd.shape[0]
    SF = np.prod(psd['Power'] ** (1 / N)) / ((1 / N) * np.sum(psd['Power']))
    return SF


#
# =============================================================================
#

def CalcSS(psd):
    """
    Calculate the Spectral Spread (SS) of a PSD.

    Parameters
    ----------
    psd : DataFrame
        A Pandas DataFrame containing a 'Frequency' and 'Power' column.

    Raises
    ------
    Exception
        An exception is raised if psd does not only have columns 'Frequency' and 'Power'

    Returns
    -------
    SS : float
        SS of the PSD.

    """

    if set(psd.columns.values) != {'Frequency', 'Power'}:
        raise Exception("psd must be a Power Spectrum Density dataframe with only a 'Frequency' and 'Power' column")

    SC = CalcSC(psd)
    SS = np.sum(((psd['Frequency'] - SC) ** 2) * psd['Power']) / np.sum(psd['Power'])
    return SS


#
# =============================================================================
#

def CalcSDec(psd):
    """
    Calculate the Spectral Decrease (SDec) of a PSD.

    Parameters
    ----------
    psd : DataFrame
        A Pandas DataFrame containing a 'Frequency' and 'Power' column.

    Raises
    ------
    Exception
        An exception is raised if psd does not only have columns 'Frequency' and 'Power'

    Returns
    -------
    SDec : float
        SDec of the PSD.

    """

    if set(psd.columns.values) != {'Frequency', 'Power'}:
        raise Exception("psd must be a Power Spectrum Density dataframe with only a 'Frequency' and 'Power' column")

    N = psd.shape[0]
    vals = np.array(psd['Power'])
    SDec = np.sum((vals[1:] - vals[0]) / N) / np.sum(vals[1:])
    return SDec


#
# =============================================================================
#

def CalcSEntropy(psd):
    """
    Calculate the Spectral Entropy of a PSD.

    Parameters
    ----------
    psd : DataFrame
        A Pandas DataFrame containing a 'Frequency' and 'Power' column.

    Raises
    ------
    Exception
        An exception is raised if psd does not only have columns 'Frequency' and 'Power'

    Returns
    -------
    SEntropy : float
        Spectral Entropy of the PSD.

    """

    if set(psd.columns.values) != {'Frequency', 'Power'}:
        raise Exception("psd must be a Power Spectrum Density dataframe with only a 'Frequency' and 'Power' column")

    prob = psd['Power'] / np.sum(psd['Power'])
    SEntropy = -np.sum(prob * np.log(prob))
    return SEntropy


#
# =============================================================================
#

def CalcSRoll(psd, percent=0.85):
    """
    Calculate the Spectral Rolloff of a PSD.

    Parameters
    ----------
    psd : DataFrame
        A Pandas DataFrame containing a 'Frequency' and 'Power' column.
    percent : float, optional
        The percentage of power to look for the Spectral Rolloff after. The default is 0.85.

    Raises
    ------
    Exception
        An exception is raised if psd does not only have columns 'Frequency' and 'Power'
    Exception
        An exception is raised if percent is not between 0 and 1

    Returns
    -------
    float
        Spectral Rolloff of the PSD.

    """

    if set(psd.columns.values) != {'Frequency', 'Power'}:
        raise Exception("psd must be a Power Spectrum Density dataframe with only a 'Frequency' and 'Power' column")

    if percent <= 0 or percent >= 1:
        raise Exception("percent must be between 0 and 1")

    total_prob = 0
    total_power = np.sum(psd['Power'])
    # Make copy and reset rows to iterate over them
    psdCalc = psd.copy()
    psdCalc = psdCalc.reset_index()
    for i in range(len(psdCalc)):
        prob = psdCalc.loc[i, 'Power'] / total_power
        total_prob += prob
        if total_power >= percent:
            return psdCalc.loc[i, 'Frequency']


#
# =============================================================================
#

def CalcSBW(psd, p=2):
    """
    Calculate the Spectral Bandwidth (SBW) of a PSD.

    Parameters
    ----------
    psd : DataFrame
        A Pandas DataFrame containing a 'Frequency' and 'Power' column.
    p : int, optional
        Order of the SBW. The default is 2, which gives the standard deviation around the centroid.

    Raises
    ------
    Exception
        An exception is raised if psd does not only have columns 'Frequency' and 'Power'
    Exception
        An exception is raised if p is not greater than 0

    Returns
    -------
    SBW : float
        The SBW of the PSD.

    """

    if set(psd.columns.values) != {'Frequency', 'Power'}:
        raise Exception("psd must be a Power Spectrum Density dataframe with only a 'Frequency' and 'Power' column")

    if p <= 0:
        raise Exception("p must be greater than 0")

    cent = CalcSC(psd)
    SBW = (np.sum(psd['Power'] * (psd['Frequency'] - cent) ** p)) ** (1 / p)
    return SBW


# Main function to calculate both time-domain and spectral features
def EMG2PSD(Sig_vals, sr=2000, normalize=True):
    """
    Creates a PSD graph of a Signal. Uses the Welch method, meaning it can be
    used as a Long Term Average Spectrum (LTAS).

    Parameters
    ----------
    Sig_vals : float list
        A list of float values. A column of a Signal.
    sr : float
        Sampling rate of the Signal.
    normalize : bool, optional
        If True, will normalize the result. If False, will not. The default is True.

        Raises
    ------
    Exception
        An exception is raised if the sampling rate is less or equal to 0

    Returns
    -------
    psd : DataFrame
        A DataFrame containing a 'Frequency' and 'Power' column. The Power column
        indicates the intensity of each frequency in the Signal provided. Results
        will be normalized if 'normalize' is set to True.
    """

    if sr <= 0:
        raise Exception("Sampling rate must be greater or equal to 0")

    # Initial parameters
    Sig_vals = Sig_vals - np.mean(Sig_vals)
    N = len(Sig_vals)

    # Calculate minimum frequency given sampling rate
    min_frequency = (2 * sr) / (N / 2)

    # Calculate window size givern sampling rate
    nperseg = int((2 / min_frequency) * sr)
    nfft = nperseg * 2

    # Apply welch method with hanning window
    frequency, power = scipy.signal.welch(
        Sig_vals,
        fs=sr,
        scaling='density',
        detrend=False,
        nfft=nfft,
        average='mean',
        nperseg=nperseg,
        window='hann'
    )

    # Normalize if set to true
    if normalize is True:
        power /= np.max(power)

    # Create dataframe of results
    psd = pd.DataFrame({'Frequency': frequency, 'Power': power})
    # Filter given
    psd = psd.loc[np.logical_and(psd['Frequency'] >= min_frequency,
                                 psd['Frequency'] <= np.inf)]

    return psd