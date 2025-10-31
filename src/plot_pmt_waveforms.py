"""
This script is for creating a plot of the pmt waveforms at various bias voltages for one channel. This
is plotted in the upper part of the figure, while in the lower figure, histograms of the integrated ADC
are plotted.

Note that the pmt waveforms will be plotted as median +/- 25 and 75 percent quantile bands.
"""

import h5py as h5 
import numpy as np
from aux import *
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import time
from numba import njit



def interp_and_center_wfs(wfs, num_pts, left, right, time_res = 2, plot=False):
    """
    Given raw ptf waveform data (at a specific scan point), this function computes an interpolation 
    for each, then runs a peak finder on each waveform. Then a centering procedure is applied to 
    align all of the peaks.

    Parameters: 
    -----------
    - wfs: array containing all of the waveforms at a given scan point
    - num_pts, int: the number of interpolation data points
    - left, int: left bound of the idices to for the peak window (example: if num_pts = 500, left = 150)
    - right, int: right bound of the idices to for the peak window (example: if num_pts = 500, right = 300)
    - time_res, int = 2: time resolution of the PTF DAQ
    - plot, bool = False: Optional debugging plotting for me  

    Returns:
    --------
    - t: the fine grid time array, starting at zero
    - result: array of all of the interpolated waveforms
    """
    # interpolate all of the waveforms
    t_fine, wfs_interp = interpolate_wfs(wfs, 0, len(wfs[0]), width=time_res, num_pts=num_pts)

    # calculate the index location of the peak for each waveform 
    # cutoff search at half the number of interpolation points, since peaks are before that
    peak_indices = np.argmax(wfs_interp[:,:int(num_pts/2)], axis=1)
    
    # Choose a reference time to align to
    ref_pos = int(np.median(peak_indices))

    # shift each waveform to align the peaks
    for i, peak_idx in enumerate(peak_indices):
        # calculate the shift in peak indices needed to align
        shift = ref_pos - peak_idx
        
        # update the waveform data to the shifted version
        wfs_interp[i] = np.roll(wfs_interp[i], shift)


    wfs_trimmed = []

    # narrow down the range to exclude the excess time before and after the peak
    for i, wf in enumerate(wfs_interp):
        wfs_trimmed.append(wf[left : right])

    # calculate a new time array after all the waveform shifting took place
    t_narrow = t_fine[left : right]
    t_width = t_narrow[-1]-t_narrow[0]
    t = np.linspace(0, t_width, len(t_narrow))

    result = np.asarray(wfs_trimmed)

    # result = np.zeros((3, int(right-left)))

    # # calculate the median and quantiles of all the waveforms
    # result[0] = np.median(wfs_trimmed, axis=0)            # median
    # result[1] = np.percentile(wfs_trimmed, 25, axis=0)    # 25th percentile
    # result[2] = np.percentile(wfs_trimmed, 75, axis=0)    # 75th percentile


    # if plot:
    #     plt.figure(figsize=(7,5))
    #     plt.plot(t, result[0], 'b-')
    #     plt.fill_between(t, result[1], result[2], color="blue", alpha=0.2)
    #     # for wf in wfs_trimmed:
    #     #     plt.plot(t, wf, 'r-', lw=0.3, alpha=0.03)

    #     plt.show()

    return t, np.asarray(result)


@njit
def integrate_pulses(wfs, dt):
    ADC_integrals = np.zeros(len(wfs))

    # compute the integral for each wf
    for i in range(len(wfs)):
        wf = wfs[i]
        ADC_integrals[i] = integrate_trapz(wf, dt)

    return ADC_integrals


def main():
    # manually enter which data files to compute the plot for
    data_files = ["../RandonsScripts/hdf5_files/CH0_water/out_run05979.hdf5",]
    
    # initialize arrays to store the data
    t_arrays = []
    pmt_wfs_arrays = []
    ADC_data_arrays = []


    # plot options, ensuring the same order as the data files order
    plot_options = [
        {
            "color": "#790000",
            "linestyle": "-",
            "label": "10 V" ,
            "linewidth": 1
        },
    ]

    fs = 12 # fontsize

    ###############################################################################################
    ### GRAB THE PMT WAVEFORMS AND COMPUTE THE INTEGRATED ADCs FOR EACH.
    ###############################################################################################
    for data_file in data_files:
        with h5.File(data_file, "r") as f:
            print(f"Processing data file [{data_file}]...")
            print("-----------------------------------------------------------------------")
            wfs = np.array(f[f"waveforms/{1}"])

            # calculate the interpolated waveforms for this data file
            t1 = time.time()
            t, wfs_interp = interp_and_center_wfs(wfs, left=300, right=600, num_pts=1000, plot=False)
            t2 = time.time()
            print(f"Time taken to interpolate waveforms: {t2-t1:.4f} s")

            t_arrays.append(t); pmt_wfs_arrays.append(wfs_interp)

            # calculate the integral of each interpolated waveform 
            t3 = time.time()
            dt = t[1]-t[0]
            ADC_integrals = integrate_pulses(wfs_interp, dt)
            t4 = time.time()
            print(f"Time taken the calculate the ADC integral for each waveform: {t4-t3:.4f} s\n\n")

            ADC_data_arrays.append(ADC_integrals)



    ##############################################################################################
    ### CREATE THE FINAL PLOT
    ###############################################################################################
    fig = plt.figure(figsize=(6.5,5))
    gs = gridspec.GridSpec(2, 1, hspace=0.3, height_ratios=[1, 1])

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['xtick.direction'] = "in"
    plt.rcParams['ytick.direction'] = "in"
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 2.5
    plt.rcParams['ytick.minor.size'] = 2.5
    plt.rcParams['xtick.major.width'] = 1.1
    plt.rcParams['ytick.major.width'] = 1.1
    plt.rcParams['xtick.minor.width'] = 0.65
    plt.rcParams['ytick.minor.width'] = 0.65
    plt.rcParams['xtick.labelsize'] = fs
    plt.rcParams['ytick.labelsize'] = fs


    # upper plot
    # ------------------------------------------------
    up = fig.add_subplot(gs[0, 0])
    for i in range(len(data_files)):
        # grab the relevent data, colors, and labels...
        wfs = pmt_wfs_arrays[i]
        wfs_median = np.median(wfs, axis=0)            # median
        wfs_25 = np.percentile(wfs, 25, axis=0)    # 25th percentile
        wfs_75 = np.percentile(wfs, 75, axis=0)    # 75th percentile
        t = t_arrays[i]
        opts = plot_options[i]

        up.plot(t, wfs_median, c = opts["color"], ls=opts["linestyle"], lw=opts["linewidth"])
        up.fill_between(t, wfs_25, wfs_75, color=opts["color"], lw=0.25, alpha=0.2)
        
    up.set_xlim(0, 40)
    up.set_xlabel("Time [ns]", fontsize=fs)
    up.set_ylabel("ADC [a.u.]", fontsize=fs)


    # Create custom legend elements
    grey_line = Line2D([0], [0], color='lightgrey', lw=2)
    light_grey_box = Patch(facecolor='lightgrey', edgecolor='lightgrey', alpha=0.5)

    labels = ["Mean", r"25 — 75 % quantile"]
    handles = [grey_line, light_grey_box]

    for i in range(len(data_files)):
        opts = plot_options[i]
        labels.append(opts["label"])
        handles.append(Line2D([0], [0], color=opts["color"], lw=2, ls=opts["linestyle"]))

    up.legend(
        handles=handles,
        labels=labels,
        ncol=4,              # 4 columns
        loc='lower center',  # place legend below the bbox anchor point
        bbox_to_anchor=(0.5, 1.02),  # (x=0.5 center, y=1.05 above the axis)
        frameon=False,       # optional: remove box
        fontsize=fs-2   
    )


    # lower plot
    # ------------------------------------------------
    low = fig.add_subplot(gs[1, 0])
    for i in range(len(data_files)):
        # grab the relevent data, colors, and labels...
        ADC_data = ADC_data_arrays[i]
        opts = plot_options[i]

        low.hist(ADC_data, bins=50, histtype="step", color=opts["color"], ls=opts["linestyle"], 
                 lw=opts["linewidth"], density=True)

    low.set_xlabel(r"ADC$\times$ns", fontsize=fs)
    low.set_ylabel("Probability density", fontsize=fs)

    plt.savefig("pmtwfs_adchist.pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
