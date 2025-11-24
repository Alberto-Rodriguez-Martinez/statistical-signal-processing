# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 08:28:38 2025

@author: alrom


Apps for statistical signal processing course
"""

# Imports used across the notebook
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import dartslab as dl 


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch


# Datasets for the ML section
try:
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# np.random.seed(7)  # reproducibility for the live session

def plot_psd(ax, x, fs, f0, use_welch=True, skip_dc=True,
             db_floor=-200.0, y_percentiles=(5, 99.5),
             nperseg_factor=8):
    """
    Plot PSD robustly, avoiding DC scale issues.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to draw on.
    x : ndarray
        Signal samples.
    fs : float
        Sampling frequency (Hz).
    f0 : float
        Carrier (used to set a reasonable x-limit).
    use_welch : bool
        If True, use Welch's method; otherwise, simple periodogram.
    skip_dc : bool
        If True, drop the DC bin (f=0) to avoid log-scale issues.
    db_floor : float
        dB floor to avoid -inf (i.e., 10*log10(Pxx + eps)).
    y_percentiles : tuple(float, float)
        Percentile-based y-limits in dB to keep the plot readable.
    nperseg_factor : int
        Welch segment length ~ len(x)/nperseg_factor (rounded to power of 2).
    """
    N = len(x)

    if use_welch:
        # reasonable segment length: power of two near N/nperseg_factor
        raw = max(256, N // nperseg_factor)
        # round to nearest power of two:
        nperseg = 1 << int(np.round(np.log2(raw)))
        nperseg = min(nperseg, N)
        f, Pxx = welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg//2,
                       detrend="constant", scaling="density")
    else:
        f, Pxx = periodogram(x, fs=fs, window="hann", detrend="constant", scaling="density")

    # Optionally drop DC bin
    if skip_dc and f.size > 1:
        f = f[1:]
        Pxx = Pxx[1:]

    # Convert to dB with numeric floor
    eps = 10**(db_floor/10.0)  # e.g., -200 dB floor
    Pxx_db = 10.0 * np.log10(Pxx + eps)

    # Plot in dB
    ax.plot(f, Pxx_db, lw=1)
    ax.set_xlim(0, min(5*f0, fs/2))
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dB/Hz]")
    ax.set_title("Power Spectral Density")

    # Robust y-limits using percentiles (ignore extreme tails)
    if Pxx_db.size > 10:
        lo = np.percentile(Pxx_db, y_percentiles[0])
        hi = np.percentile(Pxx_db, y_percentiles[1])
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.set_ylim(lo, hi)


def autocorr(x, max_lag=None, unbiased=True, normalize=True, detrend_mean=True, use_fft=True):
    """
    Compute autocorrelation r_xx[k] for lags k = -(K..K) (centered), returns (lags, rxx).

    Parameters
    ----------
    x : array_like
        Input signal.
    max_lag : int or None
        If None, uses len(x)-1. If int, limits the maximum |lag|.
    unbiased : bool
        If True, divide by (N - |k|); else divide by N (biased).
    normalize : bool
        If True, divide by r_xx[0] so r_xx[0] = 1.
    detrend_mean : bool
        If True, remove the mean before computing autocorrelation.
    use_fft : bool
        If True, use FFT method (O(N log N)); else use np.correlate (O(N^2)).

    Returns
    -------
    lags : ndarray of ints (length 2*K+1)
    rxx  : ndarray of floats (same length)
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    if detrend_mean:
        x = x - np.mean(x)

    K = (N - 1) if (max_lag is None) else int(min(max_lag, N - 1))

    if use_fft:
        # Compute autocorrelation for nonnegative lags via FFT
        nfft = 1 << int(np.ceil(np.log2(2*N - 1)))
        X = np.fft.rfft(x, n=nfft)
        S = X * np.conjugate(X)
        r = np.fft.irfft(S, n=nfft)[:N]  # r[0..N-1], lags 0..N-1
        # Build full sequence for negative lags by symmetry
        r_pos = r[:K+1]
        r_neg = r[1:K+1][::-1]  # r(-k) = r(k) for real signals
        r_full = np.concatenate([r_neg, r_pos])
    else:
        r_full = np.correlate(x, x, mode='full')
        mid = N - 1
        r_full = r_full[mid-K:mid+K+1]

    # Normalization: unbiased or biased
    if unbiased:
        denom = np.concatenate([np.arange(N-K, N), np.arange(N, N-K-1, -1)])
    else:
        denom = np.full(2*K+1, N, dtype=float)
    r_full = r_full / denom

    # Normalize to 1 at zero lag
    if normalize and r_full[K] != 0:
        r_full = r_full / r_full[K]

    lags = np.arange(-K, K+1)
    return lags, r_full

# Utility: empirical CDF
def empirical_cdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(x)+1) / len(x)
    return xs, ys

def sinusoid_with_noise(A=1.0, f0=10.0, mean=0.0, sigma=0.5, fs=2000.0, T=1.0,
                       acf_max_lag_periods=5, acf_unbiased=True, acf_normalize=True):
    """
    Generates a sinusoidal signal corrupted by Gaussian white noise and visualizes its properties.

    This function creates a sinusoidal signal with specified amplitude and frequency,
    and then adds Gaussian white noise to it. It provides a comprehensive visualization
    of the resulting signal, including:
    1. The time-domain waveform of the signal plus noise.
    2. A histogram of the added noise, compared against its theoretical Gaussian
       Probability Density Function (PDF).
    3. The autocorrelation function (ACF) of the combined signal.
    4. The Power Spectral Density (PSD) of the added noise.

    Parameters
    ----------
    A : float, optional
        Amplitude of the sinusoidal component. Defaults to 1.0.
    f0 : float, optional
        Oscillation frequency of the sinusoidal component in Hz. Defaults to 10.0.
    mean : float, optional
        Mean (average) of the Gaussian white noise. Defaults to 0.0.
    sigma : float, optional
        Standard deviation (spread) of the Gaussian white noise. Defaults to 0.5.
    fs : float, optional
        Sampling frequency in Hz. The number of samples per second. Defaults to 2000.0.
    T : float, optional
        Total duration of the sequence in seconds. Defaults to 1.0.
    acf_max_lag_periods : int or float, optional
        Maximum lag for the autocorrelation function, expressed as a number of
        periods of the sinusoid (T0). Defaults to 5.
    acf_unbiased : bool, optional
        If True, the autocorrelation estimate will be unbiased. Defaults to True.
    acf_normalize : bool, optional
        If True, the autocorrelation function will be normalized such that
        rxx(0) = 1. Defaults to True.

    Returns
    -------
    x : numpy.ndarray
        The generated signal, which is the sinusoid plus the white noise.

    Notes
    -----
    The function assumes the existence of `dsp_lab` module with `autocorr` and
    `plot_psd` functions.
    """
    # Calculate the period of the sinusoidal component.
    T0 = 1.0 / f0

    # Ensure sampling frequency is sufficient (at least 10 times the sinusoid frequency).
    # If not, adjust it and print a message.
    if fs < 10 * f0:
        fs = 10 * f0
        print(f'Sampling frequency set to 10*f0 = {fs:.2f} Hz')
    
    # Generate the time vector for the signal.
    # np.linspace creates evenly spaced numbers over the interval [0, T).
    # int(fs*T) determines the total number of samples.
    # endpoint=False excludes the end value T, ensuring correct sampling points.
    t = np.linspace(0, T, int(fs * T), endpoint=False)
    
    # Generate the sinusoidal signal component.
    # A * sin(2 * pi * f0 * t) is the standard formula for a sine wave.
    signal = A * np.sin(2 * np.pi * f0 * t)
    
    # Generate Gaussian white noise.
    # np.random.normal draws samples from a normal distribution with specified
    # mean, standard deviation (sigma), and size.
    noise = np.random.normal(mean, sigma, size=t.size)
    
    # Combine the sinusoidal signal and the noise to form the final output signal 'x'.
    x = signal + noise

    # Create a figure and a 2x2 grid of subplots for visualization.
    # figsize=(8, 8) sets the total dimensions of the figure in inches.
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # --- Subplot [0,0]: Signal in Time Domain ---
    # Plot the combined signal 'x' against time 't'.
    axes[0,0].plot(t, x, lw=1) # lw=1 sets the line width to 1.
    axes[0,0].set_xlabel("Time [s]") # Label for the x-axis.
    axes[0,0].set_ylabel("Amplitude") # Label for the y-axis.
    # Set the title with formatted parameter values for clarity.
    axes[0,0].set_title(f"Signal + Noise (A={A:.2f}, f0={f0:.2f}Hz, σ={sigma:.2f})")

    # --- Subplot [0,1]: Noise Histogram + Theoretical Gaussian PDF ---
    # Generate a histogram of the noise data.
    # n, bins, _: These capture the histogram values, bin edges, and patch objects.
    axes[0,1].hist(noise, bins=40, density=True, alpha=0.6, 
                   color='lightgray', edgecolor='gray')    
    # Create x-values for plotting the theoretical PDF.
    x_pdf = np.linspace(noise.min(), noise.max(), 300)
    # Calculate the theoretical Gaussian PDF values.
    # norm.pdf calculates the probability density function for a normal distribution.
    # Here, mean is 0 and standard deviation is sigma for the noise.
    pdf = norm.pdf(x_pdf, 0, sigma)
    # Plot the theoretical PDF on top of the histogram.
    axes[0,1].plot(x_pdf, pdf, 'r', lw=2, label='Theoretical PDF') # 'r' for red color, lw=2 for line width.
    axes[0,1].set_xlabel("Noise value") # Label for the x-axis.
    axes[0,1].set_ylabel("Density") # Label for the y-axis.
    axes[0,1].set_title("Histogram + Gaussian PDF") # Title of the subplot.
    axes[0,1].legend() # Display the legend for the theoretical PDF line.

    # --- Subplot [1,0]: Autocorrelation Function (ACF) ---
    # Calculate the maximum lag in samples for the ACF.
    # It's derived from the desired number of periods, the sinusoid period (T0), and sampling frequency.
    K = int(np.round(acf_max_lag_periods * T0 * fs))
    # Compute the autocorrelation using a utility function from dsp_lab.
    # lags_samples: Lag values in terms of number of samples.
    # rxx: The autocorrelation values.
    # max_lag=K: Sets the maximum lag for the computation.
    # unbiased=acf_unbiased: Applies an unbiased estimation if True.
    # normalize=acf_normalize: Normalizes the ACF if True (rxx(0)=1).
    # detrend_mean=True: Removes the mean before computing ACF, which is good practice.
    lags_samples, rxx = autocorr(x, max_lag=K, unbiased=acf_unbiased, 
                                    normalize=acf_normalize, detrend_mean=True)
    # Convert lag samples to lag in seconds.
    lags_sec = lags_samples / fs
    
    # Plot the autocorrelation function.
    axes[1,0].plot(lags_sec, rxx, lw=1)
    axes[1,0].set_xlabel("Lag [s]") # Label for the x-axis.
    axes[1,0].set_ylabel("Autocorrelation") # Label for the y-axis.
    # Set the title with the formatted number of periods.
    axes[1,0].set_title(f"Autocorrelation (±{acf_max_lag_periods:.2f} periods)")
    
    # Mark zero lag and ±T0 (sinusoid period) on the plot for reference.
    axes[1,0].axvline(0.0, color='k', lw=0.8, linestyle=':') # Vertical line at lag=0.
    axes[1,0].axvline(T0,  color='k', lw=0.6, linestyle='--') # Vertical line at lag=T0.
    axes[1,0].axvline(-T0, color='k', lw=0.6, linestyle='--') # Vertical line at lag=-T0.
    
    # --- Subplot [1,1]: Power Spectral Density (PSD) of the Noise ---
    # Uses a utility function from the dsp_lab library to plot the PSD of the noise.
    # axes[1,1]: The specific subplot to draw on.
    # noise: The data whose PSD is to be calculated and plotted.
    # fs: Sampling frequency of the noise.
    # f0=fs/2: Reference frequency, often set to the Nyquist frequency for PSDs.
    # use_welch=True: Employ Welch's method for smoother PSD estimation.
    # skip_dc=True: Excludes the DC (0 Hz) component.
    # db_floor=-200.0: Sets a lower limit for the dB scale.
    # y_percentiles=(5, 99.5): Defines percentile range for y-axis limits.
    # nperseg_factor=8: Parameter for Welch's method.
    plot_psd(axes[1,1], noise, fs=fs, f0=fs/2,
         use_welch=True,
         skip_dc=True,
         db_floor=-200.0,
         y_percentiles=(5, 99.5),
         nperseg_factor=8)

    # Adjust subplot parameters for a tight layout, preventing labels from overlapping.
    plt.tight_layout()
    plt.show() # Display the generated figure.
    
    return x # Return the generated signal with noise.
    
def constant_in_white_noise(C=1.0, sigma=0.5, noise_mean=0.0, fs=1000.0, T=1.0, 
                            acf_max_lag_periods=5, acf_unbiased=True, acf_normalize=True):
    """
    Generates a constant signal embedded in Gaussian white noise and visualizes its properties.

    This function creates a signal consisting of a constant value 'C'
    superimposed with Gaussian white noise characterized by a specified mean and
    standard deviation. It then generates four plots:
    1. The combined signal (constant + noise) over time.
    2. A histogram of the noise distribution compared against its theoretical
       Gaussian Probability Density Function (PDF).
    3. The autocorrelation function (ACF) of the combined signal 'x'.
    4. The Power Spectral Density (PSD) of the generated noise.
      

    Parameters
    ----------
    C : float, optional
        The constant value of the signal. Defaults to 1.0.
    sigma : float, optional
        The standard deviation (spread) of the Gaussian white noise. Defaults to 0.5.
    noise_mean : float, optional
        The mean (average) of the Gaussian white noise. Defaults to 0.0.
    fs : float, optional
        The sampling frequency in Hz. This determines the number of samples
        generated over the duration 'T'. Defaults to 1000.0.
    T : float, optional
        The total duration of the signal in seconds. Defaults to 1.0.
    acf_max_lag_periods : int or float, optional
        Maximum lag for the autocorrelation function. For a constant signal + white noise,
        this parameter's interpretation of "periods" is less direct than for a sinusoid.
        It defines the maximum lag in units of a "reference period" (T0), which is
        internally set to 1.0 / fs as white noise has no inherent period. Defaults to 5.
    acf_unbiased : bool, optional
        If True, the autocorrelation estimate will be unbiased. Defaults to True.
    acf_normalize : bool, optional
        If True, the autocorrelation function will be normalized such that
        rxx(0) = 1. Defaults to True.

    Returns
    -------
    none
    
    Notes
    -----
    The function assumes the existence of `dsp_lab` module with `autocorr` and
    `plot_psd` functions.
    """

    # Calculate time vector based on sampling frequency and duration
    # endpoint=False ensures that the last point is T - (1/fs), preventing
    # duplicate points if T is a multiple of (1/fs) and a new cycle would start.
    t = np.linspace(0, T, int(fs * T), endpoint=False)

    # Generate the constant signal component
    # np.full_like creates an array of the same shape and type as 't', filled with 'C'.
    signal = np.full_like(t, C, dtype=float)

    # Generate Gaussian white noise
    # np.random.normal draws samples from a normal (Gaussian) distribution
    # with specified mean, standard deviation (sigma), and size.
    noise = np.random.normal(noise_mean, sigma, size=t.size)

    # Combine the constant signal and the noise to form the final signal 'x'
    x = signal + noise

    # Create a figure and a 2x2 grid of subplots for visualization.
    # We now have 4 plots, so a 2x2 layout is appropriate.
    fig, axes = plt.subplots(2, 2, figsize=(10, 8)) # Adjusted figsize for better layout

    # --- Subplot [0,0]: Signal in Time Domain ---
    # Plot the generated signal 'x' against time 't'.
    axes[0,0].plot(t, x, lw=1) # lw=1 sets the line width to 1.
    axes[0,0].set_xlabel("Time [s]") # Label for the x-axis
    axes[0,0].set_ylabel("Amplitude") # Label for the y-axis
    # Set the title with formatted parameter values for clarity.
    axes[0,0].set_title(f"Constant + White Noise (C={C:.2f}, μ={noise_mean:.2f}, σ={sigma:.2f})")

    # --- Subplot [0,1]: Noise Histogram + Theoretical Gaussian PDF ---
    # Generate a histogram of the noise data.
    # n: The values of the histogram bins.
    # bins: The edges of the histogram bins.
    # _: Placeholder for the `Patches` object returned by `hist`.
    axes[0,1].hist(noise, bins=40, density=True, alpha=0.6,
                 color='lightgray', edgecolor='gray')
    
    # Create x-values for plotting the theoretical PDF
    # np.linspace generates evenly spaced numbers over a specified interval.
    x_pdf = np.linspace(noise.min(), noise.max(), 300)
    
    # Calculate the theoretical Gaussian PDF values
    # norm.pdf calculates the probability density function for a normal distribution.
    # loc: Mean of the distribution.
    # scale: Standard deviation of the distribution.
    pdf = norm.pdf(x_pdf, loc=noise_mean, scale=sigma)
    
    # Plot the theoretical PDF on top of the histogram
    axes[0,1].plot(x_pdf, pdf, 'r', lw=2, label='Theoretical PDF') # 'r' for red color
    axes[0,1].set_xlabel("Noise value") # Label for the x-axis
    axes[0,1].set_ylabel("Density") # Label for the y-axis
    axes[0,1].set_title("Histogram + Gaussian PDF") # Title of the subplot
    axes[0,1].legend() # Display the legend for the theoretical PDF line.

    # --- Subplot [1,0]: Autocorrelation Function (ACF) ---
    # This section calculates and plots the Autocorrelation Function (ACF) of the combined signal 'x'.
    # For a constant signal with added white noise, the ACF is expected to show a strong peak
    # at lag 0 (due to the signal's variance and noise variance) and then quickly drop to a constant value
    # (the square of the constant signal's mean if detrend_mean=False, or near zero if detrend_mean=True).
    # Since `detrend_mean=True` is used, the mean of 'x' (which is C + noise_mean) is removed
    # before calculating the ACF, so the ACF will primarily show the correlation of the noise
    # and any remaining structure after mean removal. For white noise, this means a peak at 0 and
    # then near zero for other lags.

    # Define a 'reference period' (T0) for max_lag calculation.
    # For a constant signal + white noise, there's no natural 'period' like in a sinusoid.
    # A common approach for time-series is to use a small fraction of the sampling period
    # or just 1.0 for simplicity if 'acf_max_lag_periods' is interpreted as 'max_lag_factor'.
    # Here, setting T0 to 1.0 / fs means 'acf_max_lag_periods' directly scales the lag in samples.
    # If the intent was for 'acf_max_lag_periods' to define a maximum lag in seconds,
    # then K should be int(np.round(acf_max_lag_periods * fs)).
    # Keeping T0 = 1.0 / fs to align with the original interpretation if it was intended to scale sample lags.
    # Consider renaming 'acf_max_lag_periods' to 'acf_max_lag_factor_samples' for clarity
    # if it's meant to be a multiplier for the sampling period.
    T0_ref = 5.0 / fs # Reference "period" is the sampling period for this context.

    # Calculate the maximum lag in samples for the ACF.
    # K represents the number of samples corresponding to acf_max_lag_periods * T0_ref.
    K = int(np.round(acf_max_lag_periods * T0_ref * fs))
    
    # Compute the autocorrelation using the dsp_lab.autocorr utility function.
    # 'x': The input signal (constant + noise) for which to compute the ACF.
    # 'max_lag=K': Specifies the maximum lag (in samples) up to which the ACF is computed.
    # 'unbiased=acf_unbiased': If True, applies an unbiased estimate, which often
    #                          gives more accurate results for finite data lengths.
    # 'normalize=acf_normalize': If True, scales the ACF so that the value at zero lag (rxx[0]) is 1.
    # 'detrend_mean=True': Removes the mean of the signal 'x' before computing the ACF.
    #                      This is crucial for signals with a non-zero mean (like C + noise_mean)
    #                      to prevent the mean from dominating the ACF and obscuring correlation patterns.
    lags_samples, rxx = autocorr(x, max_lag=K, unbiased=acf_unbiased, 
                                    normalize=acf_normalize, detrend_mean=True)
    
    # Convert lag values from samples to seconds for plotting on the x-axis.
    lags_sec = lags_samples / fs
    
    # Plot the autocorrelation function on the [1,0] subplot.
    axes[1,0].plot(lags_sec, rxx, lw=1)
    axes[1,0].set_xlabel("Lag [s]") # Label for the x-axis: Lag in seconds.
    axes[1,0].set_ylabel("Autocorrelation") # Label for the y-axis.
    # Set the title to indicate the ACF and the maximum lag in "periods" (as defined by the parameter).
    axes[1,0].set_title(f"Autocorrelation (max lag ±{acf_max_lag_periods:.2f} ref periods)")
    
    # Mark zero lag on the plot with a dotted black line for reference.
    axes[1,0].axvline(0.0, color='k', lw=0.8, linestyle=':')
    # For a constant + white noise signal, marking T0 or -T0 (sinusoid period) isn't directly relevant.
    # Here, T0_ref (1/fs) represents the minimum time increment.
    # If a specific 'period' for the constant signal was intended (e.g., for cyclic patterns if C wasn't constant),
    # it would need to be defined. For white noise, it's essentially delta-like at lag 0.
    # I've kept the lines but be aware of their interpretation in this context.
    axes[1,0].axvline(T0_ref,  color='k', lw=0.6, linestyle='--') # Mark the reference period (1/fs)
    axes[1,0].axvline(-T0_ref, color='k', lw=0.6, linestyle='--') # Mark the negative reference period (1/fs)
    
    
    # --- Subplot [1,1]: Power Spectral Density (PSD) of the Noise ---
    # Uses a utility function from the dsp_lab library to plot the PSD.
    # axes[1,1]: The specific subplot to draw on.
    # noise: The data whose PSD is to be calculated and plotted.
    # fs: Sampling frequency of the noise.
    # f0: A reference frequency (often fs/2 for Nyquist).
    # use_welch=True: Employ Welch's method for PSD estimation, which often provides
    #                 smoother and more reliable estimates by averaging multiple FFTs.
    # skip_dc=True: Excludes the DC (0 Hz) component from the plot, which can
    #               sometimes dominate and obscure other features.
    # db_floor=-200.0: Sets a lower limit for the dB scale to prevent log(0) issues
    #                  and make the plot more readable.
    # y_percentiles=(5, 99.5): Defines the percentile range for the y-axis limits,
    #                         helping to auto-scale the plot effectively.
    # nperseg_factor=8: A parameter for Welch's method, related to segment length.
    #                   A higher factor might mean more frequency resolution but
    #                   less statistical smoothing.
    plot_psd(axes[1,1], noise, fs=fs, f0=fs/2,
         use_welch=True,
         skip_dc=True,
         db_floor=-200.0,
         y_percentiles=(5, 99.5),
         nperseg_factor=8)

    # Adjust subplot parameters for a tight layout, preventing labels from overlapping.
    plt.tight_layout()
    plt.show() # Display the generated figure.
    
    #return x # Return the generated signal with noise.

def cw_burst_with_delay(A=1.0, f0=200.0, fs=10_000.0,
                        periods_total=20, periods_burst=5, delay_periods=3,
                        noise_mean=0.0, noise_sigma=0.1, annotate=True, show_psd=True,
                        acf_max_lag_periods=5, acf_unbiased=True, acf_normalize=True):
    """
    Generate a CW sine burst with configurable delay, noise, and PSD display.

    Parameters
    ----------
    A : float
        Amplitude of the sine wave.
    f0 : float
        Carrier frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    periods_total : int
        Total duration of the signal in number of periods.
    periods_burst : int
        Number of active sine periods (burst duration).
    delay_periods : float
        Start delay (in carrier periods) before the burst begins.
    noise_mean : float
        Mean of the Gaussian white noise.
    noise_sigma : float
        Standard deviation (σ) of the Gaussian white noise.
    annotate : bool
        Whether to mark burst start/end in time plots.
    show_psd : bool
        If True, plot the Power Spectral Density (periodogram).

    Returns
    -------
    none
    """
    # --- Time base ---
    T0 = 1.0 / f0
    N_total = int(np.round(periods_total * T0 * fs))
    t = np.arange(N_total) / fs

    # Burst start/end times
    tau_start = delay_periods * T0
    tau_end   = tau_start + periods_burst * T0
    T_total   = periods_total * T0

    if tau_end > T_total:
        raise ValueError("Burst does not fit inside the total duration. Adjust parameters.")

    # --- Signal generation ---
    mask = (t >= tau_start) & (t < tau_end)
    s = np.zeros_like(t)
    s[mask] = A * np.sin(2 * np.pi * f0 * (t[mask] - tau_start))

    # Add Gaussian white noise
    noise = np.random.normal(noise_mean, noise_sigma, size=t.size)
    x = s + noise

    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 4))

    # (1) Full time-domain view
    axes[0,0].plot(t, x, lw=1)
    axes[0,0].set_xlabel("Time [s]")
    axes[0,0].set_ylabel("Amplitude")
    axes[0,0].set_title(rf"Full signal (f0={f0:.2f} Hz, A={A})")
    if annotate:
        axes[0,0].axvline(tau_start, color='r', linestyle='--')
        axes[0,0].axvline(tau_end,   color='r', linestyle='--')

    # (2) Zoom around burst
    t_lo = max(0.0, tau_start - 2*T0)
    t_hi = min(t[-1], tau_end + 2*T0)
    axes[0,1].plot(t, x, lw=1)
    axes[0,1].set_xlim(t_lo, t_hi)
    axes[0,1].set_xlabel("Time [s]")
    axes[0,1].set_ylabel("Amplitude")
    axes[0,1].set_title("Zoom around burst")
    if annotate:
        axes[0,].axvline(tau_start, color='r', linestyle='--')
        axes[0,1].axvline(tau_end,   color='r', linestyle='--')

    # --- Subplot [1,0]: Autocorrelation Function (ACF) ---
    # This section calculates and plots the Autocorrelation Function (ACF) of the combined signal 'x'.
    # For a constant signal with added white noise, the ACF is expected to show a strong peak
    # at lag 0 (due to the signal's variance and noise variance) and then quickly drop to a constant value
    # (the square of the constant signal's mean if detrend_mean=False, or near zero if detrend_mean=True).
    # Since `detrend_mean=True` is used, the mean of 'x' (which is C + noise_mean) is removed
    # before calculating the ACF, so the ACF will primarily show the correlation of the noise
    # and any remaining structure after mean removal. For white noise, this means a peak at 0 and
    # then near zero for other lags.

    # Define a 'reference period' (T0) for max_lag calculation.
    # For a constant signal + white noise, there's no natural 'period' like in a sinusoid.
    # A common approach for time-series is to use a small fraction of the sampling period
    # or just 1.0 for simplicity if 'acf_max_lag_periods' is interpreted as 'max_lag_factor'.
    # Here, setting T0 to 1.0 / fs means 'acf_max_lag_periods' directly scales the lag in samples.
    # If the intent was for 'acf_max_lag_periods' to define a maximum lag in seconds,
    # then K should be int(np.round(acf_max_lag_periods * fs)).
    # Keeping T0 = 1.0 / fs to align with the original interpretation if it was intended to scale sample lags.
    # Consider renaming 'acf_max_lag_periods' to 'acf_max_lag_factor_samples' for clarity
    # if it's meant to be a multiplier for the sampling period.
    T0_ref = 5.0 / fs # Reference "period" is the sampling period for this context.

    # Calculate the maximum lag in samples for the ACF.
    # K represents the number of samples corresponding to acf_max_lag_periods * T0_ref.
    K = int(np.round(acf_max_lag_periods * T0_ref * fs))
    
    # Compute the autocorrelation using the dsp_lab.autocorr utility function.
    # 'x': The input signal (constant + noise) for which to compute the ACF.
    # 'max_lag=K': Specifies the maximum lag (in samples) up to which the ACF is computed.
    # 'unbiased=acf_unbiased': If True, applies an unbiased estimate, which often
    #                          gives more accurate results for finite data lengths.
    # 'normalize=acf_normalize': If True, scales the ACF so that the value at zero lag (rxx[0]) is 1.
    # 'detrend_mean=True': Removes the mean of the signal 'x' before computing the ACF.
    #                      This is crucial for signals with a non-zero mean (like C + noise_mean)
    #                      to prevent the mean from dominating the ACF and obscuring correlation patterns.
    lags_samples, rxx = autocorr(x, max_lag=K, unbiased=acf_unbiased, 
                                    normalize=acf_normalize, detrend_mean=True)
    
    # Convert lag values from samples to seconds for plotting on the x-axis.
    lags_sec = lags_samples / fs
    
    # Plot the autocorrelation function on the [1,0] subplot.
    axes[1,0].plot(lags_sec, rxx, lw=1)
    axes[1,0].set_xlabel("Lag [s]") # Label for the x-axis: Lag in seconds.
    axes[1,0].set_ylabel("Autocorrelation") # Label for the y-axis.
    # Set the title to indicate the ACF and the maximum lag in "periods" (as defined by the parameter).
    axes[1,0].set_title(f"Autocorrelation (max lag ±{acf_max_lag_periods:.2f} ref periods)")
    
    # Mark zero lag on the plot with a dotted black line for reference.
    axes[1,0].axvline(0.0, color='k', lw=0.8, linestyle=':')
    # For a constant + white noise signal, marking T0 or -T0 (sinusoid period) isn't directly relevant.
    # Here, T0_ref (1/fs) represents the minimum time increment.
    # If a specific 'period' for the constant signal was intended (e.g., for cyclic patterns if C wasn't constant),
    # it would need to be defined. For white noise, it's essentially delta-like at lag 0.
    # I've kept the lines but be aware of their interpretation in this context.
    axes[1,0].axvline(T0_ref,  color='k', lw=0.6, linestyle='--') # Mark the reference period (1/fs)
    axes[1,0].axvline(-T0_ref, color='k', lw=0.6, linestyle='--') # Mark the negative reference period (1/fs)



# (3) Power Spectral Density (periodogram)
    if show_psd:
        #PSD — robust plotting
        dl.plot_psd(axes[1,1], noise, fs=fs, f0=f0,
             use_welch=True,    # try Welch for a smoother PSD
             skip_dc=True,      # drop DC bin
             db_floor=-200.0,   # numeric floor for dB conversion
             y_percentiles=(5, 99.5),
             nperseg_factor=8)
    else:
        axes[2].axis("off")

    plt.tight_layout()
    plt.show()
    
    #return x

def cw_burst_with_delay(A=1.0, f0=200.0, fs=10_000.0,
                        periods_total=20, periods_burst=5, delay_periods=3,
                        noise_mean=0.0, noise_sigma=0.1, annotate=True, show_psd=True,
                        acf_max_lag_periods=5, acf_unbiased=True, acf_normalize=True):
    """
    Generates a Continuous Wave (CW) sine burst with configurable delay and added white noise,
    and visualizes its time-domain waveform, autocorrelation, and optionally its Power Spectral Density.

    Parameters
    ----------
    A : float, optional
        Amplitude of the sine wave. Defaults to 1.0.
    f0 : float, optional
        Carrier frequency of the sine burst in Hz. Defaults to 200.0.
    fs : float, optional
        Sampling frequency in Hz. Defaults to 10000.0.
    periods_total : int, optional
        Total duration of the signal, expressed as a number of carrier periods (1/f0).
        Defaults to 20.
    periods_burst : int, optional
        Duration of the active sine burst, expressed as a number of carrier periods.
        Defaults to 5.
    delay_periods : float, optional
        Time delay (in carrier periods) before the sine burst begins. Defaults to 3.
    noise_mean : float, optional
        Mean of the Gaussian white noise added to the signal. Defaults to 0.0.
    noise_sigma : float, optional
        Standard deviation (σ) of the Gaussian white noise. Defaults to 0.1.
    annotate : bool, optional
        If True, marks the start and end times of the burst on the time-domain plots.
        Defaults to True.
    show_psd : bool, optional
        If True, plots the Power Spectral Density (PSD) of the *noise*. If False,
        that subplot will remain empty. Defaults to True.
    acf_max_lag_periods : int or float, optional
        Maximum lag for the autocorrelation function, expressed as a number of
        carrier periods (T0). Defaults to 5.
    acf_unbiased : bool, optional
        If True, the autocorrelation estimate will be unbiased. Defaults to True.
    acf_normalize : bool, optional
        If True, the autocorrelation function will be normalized such that
        rxx(0) = 1. Defaults to True.

    Returns
    -------
    none
        
    Raises
    ------
    ValueError
        If the burst duration and delay cause it to extend beyond the total signal duration.

    Notes
    -----
    The function assumes the existence of `dsp_lab` module with `autocorr` and
    `plot_psd` functions.
    """
    # --- Time base calculation ---
    T0 = 1.0 / f0 # Calculate the period of the carrier frequency.
    # Calculate the total number of samples needed for the specified total periods.
    # np.round is used to handle potential floating point inaccuracies when casting to int.
    N_total = int(np.round(periods_total * T0 * fs))
    # Create the time vector, from 0 up to (N_total-1)/fs.
    t = np.arange(N_total) / fs

    # --- Burst timing ---
    # Calculate the start time of the burst based on delay_periods and T0.
    tau_start = delay_periods * T0
    # Calculate the end time of the burst.
    tau_end   = tau_start + periods_burst * T0
    # Calculate the total duration of the signal in seconds.
    T_total   = periods_total * T0

    # Validate that the burst fits within the total signal duration.
    if tau_end > T_total + (1e-9): # Adding a small tolerance for float comparisons
        raise ValueError("Burst does not fit inside the total duration. Adjust parameters.")

    # --- Signal generation ---
    # Create a boolean mask to identify the time samples where the burst is active.
    mask = (t >= tau_start) & (t < tau_end)
    # Initialize the signal array with zeros, same shape as time vector.
    s = np.zeros_like(t)
    # Apply the sine wave only to the active burst samples.
    # (t[mask] - tau_start) ensures the sine wave starts at phase 0 at tau_start.
    s[mask] = A * np.sin(2 * np.pi * f0 * (t[mask] - tau_start))

    # Add Gaussian white noise to the signal.
    noise = np.random.normal(noise_mean, noise_sigma, size=t.size)
    x = s + noise

    # --- Plotting ---
    # Create a figure with a 2x2 grid of subplots.
    fig, axes = plt.subplots(2, 2, figsize=(10, 8)) # Adjusted figsize for better visualization.

    # --- Subplot [0,0]: Full Time-Domain View ---
    # Plot the entire combined signal (burst + noise).
    axes[0,0].plot(t, x, lw=1)
    axes[0,0].set_xlabel("Time [s]")
    axes[0,0].set_ylabel("Amplitude")
    # Set the title with relevant signal parameters.
    axes[0,0].set_title(rf"Full Signal (f0={f0:.2f} Hz, A={A:.2f}, σ={noise_sigma:.2f})")
    # If annotate is True, draw vertical dashed lines to mark burst start and end.
    if annotate:
        axes[0,0].axvline(tau_start, color='r', linestyle='--', label='Burst Start/End')
        axes[0,0].axvline(tau_end,   color='r', linestyle='--')
        axes[0,0].legend() # Display legend for annotations.

    # --- Subplot [0,1]: Zoomed Time-Domain View around the Burst ---
    # Define time limits for the zoomed-in view: 2 periods before and after the burst.
    t_lo = max(0.0, tau_start - 2*T0)
    t_hi = min(t[-1], tau_end + 2*T0)
    axes[0,1].plot(t, x, lw=1)
    axes[0,1].set_xlim(t_lo, t_hi) # Apply zoom to the x-axis.
    axes[0,1].set_xlabel("Time [s]")
    axes[0,1].set_ylabel("Amplitude")
    axes[0,1].set_title("Zoom around burst")
    # If annotate is True, draw vertical dashed lines again for consistency.
    if annotate:
        axes[0,1].axvline(tau_start, color='r', linestyle='--')
        axes[0,1].axvline(tau_end,   color='r', linestyle='--')

    # --- Subplot [1,0]: Autocorrelation Function (ACF) ---
    # This section calculates and plots the Autocorrelation Function (ACF) of the combined signal 'x'.
    # For a sine burst with noise, the ACF will show periodic behavior corresponding to f0,
    # decaying as lag increases (due to the finite burst length and noise decorrelation).
    # The noise contributes a delta-like peak at lag 0 if `detrend_mean=True`.

    # The 'periods' in acf_max_lag_periods directly refers to the carrier period T0 (1/f0).
    # Calculate the maximum lag in samples for the ACF based on carrier periods.
    K = int(np.round(acf_max_lag_periods * T0 * fs))
    
    # Compute the autocorrelation using `dsp_lab.autocorr`.
    # 'x': The signal for which to compute the ACF.
    # 'max_lag=K': The maximum lag in samples.
    # 'unbiased=acf_unbiased': Use unbiased estimation.
    # 'normalize=acf_normalize': Normalize the ACF to 1 at zero lag.
    # 'detrend_mean=True': Remove the mean of 'x' before computing ACF. This helps
    #                      to see the AC of the AC-coupled signal and noise.
    lags_samples, rxx = autocorr(x, max_lag=K, unbiased=acf_unbiased, 
                                    normalize=acf_normalize, detrend_mean=True)
    
    # Convert lag values from samples to seconds for the plot's x-axis.
    lags_sec = lags_samples / fs
    
    # Plot the autocorrelation function.
    axes[1,0].plot(lags_sec, rxx, lw=1)
    axes[1,0].set_xlabel("Lag [s]")
    axes[1,0].set_ylabel("Autocorrelation")
    # Set the title, indicating the maximum lag in carrier periods.
    axes[1,0].set_title(f"Autocorrelation (±{acf_max_lag_periods:.2f} carrier periods)")
    
    # Mark zero lag and ±T0 (carrier period) on the plot for key references.
    axes[1,0].axvline(0.0, color='k', lw=0.8, linestyle=':')
    axes[1,0].axvline(T0,  color='k', lw=0.6, linestyle='--', label=r'$\pm T_0$')
    axes[1,0].axvline(-T0, color='k', lw=0.6, linestyle='--')
    # Also mark multiples of T0 if useful for visualizing periodicity.
    if acf_max_lag_periods > 1:
        axes[1,0].axvline(2*T0,  color='gray', lw=0.4, linestyle=':')
        axes[1,0].axvline(-2*T0, color='gray', lw=0.4, linestyle=':')
    axes[1,0].legend() # Display legend for reference lines.

    # --- Subplot [1,1]: Power Spectral Density (PSD) of the Noise (Conditional) ---
    if show_psd:
        # Plot the Power Spectral Density of the *noise* component.
        # This helps to confirm if the noise truly is 'white' (flat spectrum).
        plot_psd(axes[1,1], x, fs=fs, f0=f0, # Using f0 for reference, but fs/2 is common for full spectrum
             use_welch=True,    # Use Welch's method for smoother PSD estimation.
             skip_dc=True,      # Exclude the DC component.
             db_floor=-200.0,   # Numerical floor for dB conversion.
             y_percentiles=(5, 99.5), # Auto-scale y-axis based on percentiles.
             nperseg_factor=8) # Segment length factor for Welch's method.
        axes[1,1].set_title(f"Noise PSD (σ={noise_sigma:.2f})")
    else:
        # If show_psd is False, turn off the axis for this subplot.
        axes[1,1].axis("off")
        axes[1,1].set_title("PSD disabled")

    # Adjust subplot parameters for a tight layout, preventing labels from overlapping.
    plt.tight_layout()
    plt.show() # Display the generated figure.

    #return x # Return the generated signal x (sine burst + noise).
    
def sliding_stats(x, win=200):
    """
    Sliding (moving) mean and variance using a population variance (ddof=0).
    Returns arrays with length len(x) - win + 1 (i.e., 'valid' windows).
    """
    w = np.lib.stride_tricks.sliding_window_view(x, win)
    means = w.mean(axis=-1)
    vars_ = w.var(axis=-1)
    return means, vars_

def stationarity_demo(N=6000, win=200, mu2=0.5, sigma2=1.5, mu3=-0.5, sigma3=0.7):
    seg = N // 3
    rng = np.random.default_rng()

    # Build a non-stationary signal with three Gaussian segments
    x_ns = np.concatenate([
        rng.normal(0.0, 1.0, seg),
        rng.normal(mu2, sigma2, seg),
        rng.normal(mu3, sigma3, N - 2*seg)  # ensures total length N
    ])

    # Sliding statistics
    m, v = sliding_stats(x_ns, win=int(win))

    # Plots in a single figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True)

    axes[0].plot(x_ns)
    axes[0].set_title("Non-stationary signal (mean/variance drift)")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(m)
    axes[1].set_title(f"Sliding mean — window={int(win)}")
    axes[1].set_xlabel("Window index")
    axes[1].set_ylabel("Mean")

    axes[2].plot(v)
    axes[2].set_title(f"Sliding variance — window={int(win)}")
    axes[2].set_xlabel("Window index")
    axes[2].set_ylabel("Variance")

    plt.show()
    

# ============================================================
# CW radar burst simulation with optional interactive widgets
# Author: GPT-5 (2025)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Try to import widgets; if unavailable, still run core code
try:
    from ipywidgets import FloatSlider, IntSlider, VBox, HBox, Output
    from IPython.display import display, clear_output
    _WIDGETS_OK = True
except Exception:
    _WIDGETS_OK = False


def simulate_cw_radar(
    noise_mean: float = 0.0,
    noise_power: float = 0.01,
    delay_s: float = 2e-3,
    attenuation: float = 0.5,
    f0: float = 10e3,
    n_periods: int = 50,
    fs: float = 200e3,
    length_s: float = 0.05,
    random_seed: int | None = 1234,
):
    """
    Simulate a simple CW-burst radar scenario.

    Parameters
    ----------
    noise_mean : float
        Mean of the additive white Gaussian noise (AWGN).
    noise_power : float
        Variance of the AWGN (power).
    delay_s : float
        One-way propagation delay (seconds).
    attenuation : float
        Linear amplitude scaling (0..1 typical).
    f0 : float
        Carrier frequency (Hz).
    n_periods : int
        Number of cycles in the CW burst.
    fs : float
        Sampling frequency (Hz).
    length_s : float
        Total signal duration (s).
    random_seed : int | None
        Seed for reproducibility.

    Returns
    -------
    dict : contains time, tx, rx, xcorr, lag axes, and estimated delay.
    """
    N = int(np.round(length_s * fs))
    t = np.arange(N) / fs

    # CW burst
    burst_dur = n_periods / f0
    burst_samples = int(np.round(burst_dur * fs))
    burst_t = np.arange(burst_samples) / fs
    burst = np.cos(2 * np.pi * f0 * burst_t)

    tx = np.zeros(N)
    tx[:min(burst_samples, N)] = burst[:min(burst_samples, N)]

    # Received = delayed + attenuated + noise
    delay_samples = int(np.round(delay_s * fs))
    rx = np.zeros_like(tx)
    start = delay_samples
    stop = delay_samples + len(burst)
    if start < N:
        rx[start:min(stop, N)] = attenuation * burst[:max(0, min(len(burst), N - start))]

    rng = np.random.default_rng(random_seed)
    noise = rng.normal(noise_mean, np.sqrt(noise_power), size=N)
    rx_noisy = rx + noise

    # Cross-correlation
    rx_zm = rx_noisy - np.mean(rx_noisy)
    burst_zm = burst - np.mean(burst)
    xcorr = correlate(rx_zm, burst_zm, mode="full")
    lags = np.arange(-len(burst) + 1, len(rx_zm)) / fs
    est_delay_s = lags[np.argmax(xcorr)]

    return {
        "t": t,
        "tx": tx,
        "rx": rx_noisy,
        "lags_s": lags,
        "xcorr": xcorr,
        "est_delay_s": est_delay_s,
        "true_delay_s": delay_s,
    }


def plot_tx(t, tx):
    plt.figure()
    plt.plot(t, tx)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Transmitted CW Burst")
    plt.grid(True)
    plt.show()


def plot_rx(t, rx):
    plt.figure()
    plt.plot(t, rx)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Received Signal (Delayed + Noisy)")
    plt.grid(True)
    plt.show()


def plot_xcorr(lags_s, xcorr, true_delay_s=None, est_delay_s=None):
    plt.figure()
    plt.plot(lags_s, xcorr)
    plt.xlabel("Lag (s)")
    plt.ylabel("Cross-correlation")
    plt.title("Cross-correlation (RX vs Burst Template)")
    if true_delay_s is not None:
        plt.axvline(true_delay_s, linestyle="--", label="True delay")
    if est_delay_s is not None:
        plt.axvline(est_delay_s, linestyle=":", label="Estimated delay")
    plt.legend()
    plt.grid(True)
    plt.show()


def cw_radar_once(
    noise_mean=0.0,
    noise_power=0.01,
    delay_ms=2.0,
    attenuation=0.5,
    f0_khz=10.0,
    n_periods=50,
    fs_khz=200.0,
    length_ms=50.0,
    random_seed=1234,
):
    """Convenient wrapper using milliseconds/kHz."""
    res = simulate_cw_radar(
        noise_mean=noise_mean,
        noise_power=noise_power,
        delay_s=delay_ms * 1e-3,
        attenuation=attenuation,
        f0=f0_khz * 1e3,
        n_periods=n_periods,
        fs=fs_khz * 1e3,
        length_s=length_ms * 1e-3,
        random_seed=random_seed,
    )
    plot_tx(res["t"], res["tx"])
    plot_rx(res["t"], res["rx"])
    plot_xcorr(res["lags_s"], res["xcorr"], res["true_delay_s"], res["est_delay_s"])
    return res


def cw_radar_demo():
    """Interactive exploration (requires ipywidgets)."""
    if not _WIDGETS_OK:
        print("ipywidgets not available. Use cw_radar_once() instead.")
        return

    from ipywidgets import FloatSlider, IntSlider, VBox, HBox, Output
    from IPython.display import display, clear_output

    noise_mean_w = FloatSlider(description="Noise mean", value=0.0, min=-1, max=1, step=0.01)
    noise_power_w = FloatSlider(description="Noise power", value=0.01, min=1e-6, max=1, step=0.001)
    delay_ms_w = FloatSlider(description="Delay (ms)", value=2.0, min=0, max=50, step=0.1)
    attenuation_w = FloatSlider(description="Attenuation", value=0.5, min=0, max=1, step=0.01)
    f0_khz_w = FloatSlider(description="f0 (kHz)", value=10.0, min=0.1, max=200, step=0.1)
    n_periods_w = IntSlider(description="# periods", value=50, min=1, max=500, step=1)
    fs_khz_w = FloatSlider(description="fs (kHz)", value=200.0, min=5, max=2000, step=5)
    length_ms_w = FloatSlider(description="Length (ms)", value=50.0, min=5, max=500, step=5)
    seed_w = IntSlider(description="Seed", value=1234, min=0, max=999999, step=1)

    out = Output()

    def _update(*args):
        with out:
            clear_output(wait=True)
            res = cw_radar_once(
                noise_mean=noise_mean_w.value,
                noise_power=noise_power_w.value,
                delay_ms=delay_ms_w.value,
                attenuation=attenuation_w.value,
                f0_khz=f0_khz_w.value,
                n_periods=n_periods_w.value,
                fs_khz=fs_khz_w.value,
                length_ms=length_ms_w.value,
                random_seed=seed_w.value,
            )
            print(f"Estimated delay: {res['est_delay_s']*1e3:.3f} ms | True delay: {res['true_delay_s']*1e3:.3f} ms")

    for w in [noise_mean_w, noise_power_w, delay_ms_w, attenuation_w, f0_khz_w,
              n_periods_w, fs_khz_w, length_ms_w, seed_w]:
        w.observe(_update, names="value")

    ui = VBox([
        HBox([noise_mean_w, noise_power_w, attenuation_w]),
        HBox([delay_ms_w, f0_khz_w, n_periods_w]),
        HBox([fs_khz_w, length_ms_w, seed_w]),
        out
    ])

    display(ui)
    _update()

