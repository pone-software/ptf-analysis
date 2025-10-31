"""
Module containing commonly used functions for plotting waveforms and processing PTF data
"""

import h5py as h5 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from numba import njit
import time


@njit
def find_peaks_numba(signal, height, distance=1):
    """
    Find peaks in 1D array with minimum height and minimum distance.

    Parameters
    ----------
    signal : 1D numpy array
        Input waveform.
    height : float
        Minimum height of a peak.
    distance : int
        Minimum number of samples between peaks.

    Returns
    -------
    peaks : 1D numpy array
        Indices of the peaks in the input array.
    """
    n = len(signal)
    peaks = []

    # First pass: find local maxima above height
    for i in range(1, n-1):
        if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] >= height:
            peaks.append(i)

    peaks = np.asarray(peaks)

    # Second pass: enforce minimum distance
    if distance > 1 and len(peaks) > 1:
        filtered_peaks = []
        altered_peaks = False
        peak_distances = peaks[1:] - peaks[:-1]

        for i in range(len(peak_distances)):
            peak_distance = peak_distances[i]

            if peak_distance < distance:
                peak_left = peaks[i]; peak_right = peaks[i+1]
                
                if signal[peak_left] > signal[peak_right]:
                    filtered_peaks.append(peak_left)
                else:
                    filtered_peaks.append(peak_right)

                altered_peaks = True

        if altered_peaks:
            peaks = filtered_peaks
            peaks = np.asarray(peaks)

    return peaks



@njit
def integrate_trapz(y_dense, dt):
    """
    Integrates the interpolated waveforms for an accurate integral calculation using the trapezoid rule.
    Assumes constant dt spacing between points
    """
    total = 0.0

    for i in range(len(y_dense) - 1):
        total += (y_dense[i] + y_dense[i+1])

    return (dt/2) * total



@njit
def correct_wf(wf):
    """
    Function that inverts the waveform and vertically shifts it by the baseline noise to get a 
    corrected waveform
    """
    # calculate the baseline noise from an average of the first few samples before the peak
    # I have seen across all of my data files that our photon hits never occur before index 20
    # of each waveform
    baseline_noise = np.mean(wf[:20])

    return -1*wf + baseline_noise



def compute_ADC_statistics(integrated_ADCs):
    """
    Function to compute the mean and standard devation of the integrated ADCs for each scan point

    Will implement a gaussian fitting procedure here to calculate mean and standard deviation in the future
    """
    return np.mean(integrated_ADCs), np.std(integrated_ADCs, ddof=1), len(integrated_ADCs)



def interpolate_wfs(wfs, low, high, width = 2, num_pts = 250, plot=False):
    """
    Calculates the interpolation (using scipy's PchipInterpolator) of all the waveforms at a 
    given scan point inbetween the low and high index bounds.

    This function is primarily used for making the pmt waveforms at various bias voltages plot
    for the optics paper.

    Parameters:
    -----------
    - waveforms: the raw waveform data from the file at a given scan point
    - low: int, the lower index bound for isolating the region with the peak
    - high: int, the upper index bound for isolating the region with the peak
    - wdith = 2: the 
    - num_pts = 250: int, the number of fine time points to calculate the interpolation on
    """
    interp_wfs = np.zeros((len(wfs), num_pts))

    # set the time values of the integration window (shifted to start at 0)
    # multiplied by width = 2 ns since sampling rate is 2 ns
    t = width*np.arange(0, high-low) # ns

    # initialize a dense time array to integrate the interpolation over
    t_fine = width*np.linspace(0, high-low, num_pts)

    for i, wf in enumerate(wfs):
        # shift and invert the waveform
        wf_cleaned = correct_wf(wf)

        # narrow time range centralized around the peak
        wf_window = wf_cleaned[low:high]

        # calculate the interpolation of the waveform on the integration interval
        interp = PchipInterpolator(t, wf_window)

        interp_wfs[i] = interp(t_fine)

        if plot:
            plt.plot(t_fine, interp_wfs[i], 'k-')
            plt.plot(t, wf_window, 'ro')
            plt.show()

    return t_fine, interp_wfs


def integrate_pulses(waveforms, low, high, width=2, interpolate=False, plot=False):
    """
    Function that integrates each PMT pulse for a scan point's waveforms

    Parameters:
    -----------
    - waveforms: the raw waveform data from the file at a given scan point
    - low: lower bound of the integration window
    - high: upper bound of the integration window
    - min_ADC = 25: minimum ADC threshold (above background) to qualify as a peak
    - width: width in nanoseconds of each measurement (2 ns default)
    - plot, bool: mostly for debugging and checking on each waveform's interpolation, default False
    """
    # initialize array for storing the integrals
    integrated_ADCs = np.zeros(len(waveforms))

    if interpolate:
        # set the time values of the integration window (shifted to start at 0)
        # multiplied by width = 2 ns since sampling rate is 2 ns
        t = width*np.arange(0, high-low) # ns

        # initialize a dense time array to integrate the interpolation over
        t_fine = width*np.linspace(0, high-low, 500)

    # loop through every waveform and compute the integral of each
    for i in range(len(waveforms)):
        wf = waveforms[i]

        # shift and invert the waveform
        wf_cleaned = correct_wf(wf)

        # isolate the integration window
        wf_int_window = wf_cleaned[low:high]

        if interpolate:
            # calculate the interpolation of the waveform on the integration interval
            interp = PchipInterpolator(t, wf_int_window)
            wf_interp = interp(t_fine)

            # calculate the integral of the interpolation
            integrated_ADCs[i] = integrate_trapz(wf_interp, t_fine[1]-t_fine[0])

        else:
            integrated_ADCs[i] = 2*np.sum(wf_int_window)

        if plot and interpolate:
            plt.plot(t, wf_int_window, 'ko', ms=5)
            plt.plot(t_fine, wf_interp, 'k-', lw=0.75)
            plt.show()

    if plot:
        plt.hist(integrated_ADCs, bins=50, histtype="step")
        plt.xlabel(r"$\Delta_{\mathrm{integral}}$")
        plt.ylabel("Frequency")
        plt.show()

    return integrated_ADCs



def integrated_ADC_histogram(integrated_ADCs, bins = 50, file_name = None):
    """
    Plots a histogram of the integrated ADCs at a scan point, or of the supplied integrated ADCs
    """
    plt.figure(figsize=(7,5))

    plt.hist(integrated_ADCs, bins=bins, histtype="step", color="red", lw=1.25)

    plt.xlabel(r"Integrated ADC [ADC$\times$ns]", size=12)
    plt.ylabel("Frequency", size=12)

    x_range = np.max(integrated_ADCs) - np.min(integrated_ADCs)

    plt.xlim(np.min(integrated_ADCs) - 0.05*x_range, np.max(integrated_ADCs) + 0.05*x_range)

    plt.minorticks_on()
    plt.tick_params(which="both", direction="in", right=True, top=True)
    plt.tick_params(which="major", length=5)
    plt.tick_params(which="minor", length=3)


    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
    
    plt.show()