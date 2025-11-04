"""
Module containing commonly used functions for plotting waveforms and processing PTF data
"""

import h5py as h5 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from numba import njit
import time
from pathlib import Path


def get_dir_path(path_str):
    """
    Helper function to determine whether a given string leads to a path to a directory or file.
    If it leads to a file, grab the parent directory it's in.
    """
    p = Path(path_str)
    if p.is_file():
        # If it’s a file, return its parent directory
        return p.parent
    elif p.is_dir():
        # If it’s already a directory, keep as is
        return p
    else:
        # If it doesn’t exist or is invalid
        raise FileNotFoundError(f"Path does not exist: {path_str}")


# location of the PMT diffuser's center in space
pmt_y = 0.372
pmt_z = 0.495


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


def generate_sequence(group_centers, width=4, num_pts=9):
    """
    Generate sequence like [-2, -1, 0, 1, 2, -5, -4, -3, -2, -1, -8, ...].

    Parameters
    ----------
    group_centers : array
        Location of desired theta center point for each block
    width : int, optional
        Width of each block (default=4).
    num_pts : int, optional
        Number of points per block (implicitly defines even spacing between block elements)

    Returns
    -------
    np.ndarray
    """
    seq = []

    for center in group_centers:
        group = np.linspace(center-width/2, center+width/2, num_pts)
        seq.append(group)

    return np.concatenate(seq)



def find_best_theta(y_arr, z_arr, thetas, theta_centers, num_offsets, width=4):
    """
    Apologies for this convoluted function for just computing the best theta values. The
    PTF has some issues in how it records theta values using the optical box's on-board 
    phidget. For zenith (zenith and theta are intergangeable names) values below 87 degrees
    are offset by rougly 1-2 degrees. However, when the box tilts beyond 87 degrees, the
    phidget measures tilts that are offset by 3 or more degrees. Moreover, the the phidget
    isn't capable of measuring beyond around 87 degrees zenith, so, for zeniths past 90,
    the recorded values loop back and decrease from 87 degrees. Overall, I would like to
    discuss how to best proceed with this strange behaviour, but this is what I have so far.

    Parameters:
    -----------
      - y_arr: array of the optical box's y coordinates during the scan
      - z_arr: array of the optical box's z coordinates during the scan
      - thetas: array of averaged theta values measured at each scan point
      - theta_centers: array of the desired theta values
      - num_offsets: number of offset theta measurements computed at each theta center
      - width: the span of the theta values that the offsets take, default 4 degrees.

    Returns:
    --------
      - indices: array indices for theta values that best match the target theta centers
      - thetas: array of the closest theta values to the target theta centers
    """
    # compute the relative y and z positions of the box from the center of the diffuser
    rel_y = pmt_y - y_arr
    rel_z = pmt_z - z_arr

    # box records negative theta values, multiply by -1 to make them positive
    thetas = -1*thetas

    # this part computes what the exact theta offsets would be if the box measured perfectly
    # will use this to compare to the actual imperfect measurements
    set_theta = -1*generate_sequence(theta_centers, width=width, num_pts=num_offsets)

    # # NOTE: MANUALLY REMOVING THE -5th index
    # set_theta = np.delete(set_theta, -5)

    # plt.plot(figsize=(8,6))
    # plt.plot(np.arange(1,len(thetas)+1), thetas, "k-")
    # plt.ylim(-3,95)
    # plt.yticks(np.arange(0,91,10))
    # plt.grid()
    # # plt.savefig("scanpoint-angles-notcorrected.png", dpi=300, bbox_inches='tight')
    # plt.show()


    # compute the theta values that were supposed to be past 90 and correct them
    for i, goal_theta in enumerate(set_theta):
        theta = thetas[i]
        if np.isnan(theta):
            continue
        
        # adjust the measured thetas past 87 degrees since those were quite inaccurate
        # NOTE: this is the part I tried to explain in the docstring of this function
        # i simply just replace these values with the target which yields inaccurate
        # results past 87 degrees
        if goal_theta > 87:
            thetas[i] = set_theta[i]

    # remove any weird NaN values in the array (usually not an issue)
    thetas = thetas[~np.isnan(thetas)]

    # # ----------------------------------------------------------------------

    # plt.figure(figsize=(6,4))
    # plt.plot(np.arange(1, len(thetas)+1), thetas, "k-")
    # plt.plot([0,180], [90,90], 'k--', lw=0.7)
    # plt.xlabel("Scan point number", size=12)
    # plt.ylabel(r"Phidget measured zenith [$^{\circ}$]", size=12)
    # plt.title("Theta offset arc scans", size=12)
    # plt.xlim(0,180)
    # plt.ylim(-5,110)
    # plt.grid()
    # # plt.savefig("scanpoint-angles.png", dpi=300, bbox_inches='tight')
    # plt.show()


    # buh = np.arange(0,106,3)

    # theta_position = []

    # for duh in buh:
    #     for i in range(5):
    #         theta_position.append(duh)
    
    # theta_position = np.asarray(theta_position)

    # plt.figure(figsize=(6,4))
    # plt.plot(np.arange(1, len(thetas)+1), thetas-theta_position, "k-")
    # plt.xlabel("Scan point number", size=12)
    # plt.ylabel(r"$\theta_{\mathrm{phidget}} - \theta_{\mathrm{desired}}$ [$^{\circ}$]", size=12)
    # plt.title("Theta offset differences arc scans", size=12)
    # plt.xlim(0,180)
    # # plt.ylim(-5,110)
    # plt.grid()
    # # plt.savefig("scanpoint-angle-differences.png", dpi=300, bbox_inches='tight')
    # plt.show()

    # -------------------------------------------------------------------

    N = len(thetas)

    # these are the groups of theta offsets per target theta that we will try to find the
    # closest theta value
    group_indices = np.arange(0, N, num_offsets)

    indices = []

    # compute the closest thetas for all of the groups except the last (as it may have less entries)
    for i, i_g in enumerate(group_indices):
        y = rel_y[i_g]; z = rel_z[i_g]

        # compute desired theta
        desired_theta = np.atan2(z, y)*(180/np.pi)

        if i_g == group_indices[-1]:
            theta_group = thetas[i_g:len(thetas)]
        else:
            theta_group = thetas[i_g:group_indices[i+1]]

        # determine the closest phidget theta to the desired theta
        differences = np.array([np.abs(theta-desired_theta) for theta in theta_group])

        best_theta_i = np.argmin(differences)

        indices.append(i_g + best_theta_i)

    return np.asarray(indices), thetas