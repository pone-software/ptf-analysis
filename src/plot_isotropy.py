"""
Python script to plot isotropy. Can supply multiple data files to plot multiple angular emission profiles.
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

# location of the PMT diffuser's center in space
pmt_y = 0.372
pmt_z = 0.495


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


def export_integrated_ADC_data(thetas, integrated_ADCs, integrated_ADC_errs, dataset_names, out_file):
    with h5.File(out_file, "w") as f:
        for i, name in enumerate(dataset_names):
            theta_arr = thetas[i]
            ADCs = integrated_ADCs[i]
            ADC_errs = integrated_ADC_errs[i]

            grp = f.create_group(name)
            grp.create_dataset("theta", data=theta_arr)
            grp.create_dataset("mean_integrated_ADCs", data=ADCs)
            grp.create_dataset("standard_errors", data=ADC_errs)
        
        f.close()



def main():
    data_files = ["hdf5_files/CH0_air/out_run05963.hdf5",
                  "hdf5_files/CH0_air/out_run05964.hdf5",
                  "hdf5_files/CH0_air/out_run05965.hdf5",
                  "hdf5_files/CH0_air/out_run05966.hdf5"]

    n_files = len(data_files)

    export_ADC_data = True
    export_path = "hdf5_files/for_felix/CH0_air.hdf5"

    # plot options for each line plotted, ensure the same order as the data files order
    plot_options = [
        {
            "color": "#003502",
            "linestyle": "-",
            "label": "Run 5963" ,
            "linewidth": 1.2
        },
        {
            "color": "#007511",
            "linestyle": "-",
            "label": "Run 5964" ,
            "linewidth": 1.2
        },
        {
            "color": "#00B62A",
            "linestyle": "-",
            "label": "Run 5965" ,
            "linewidth": 1.2
        },
        {
            "color": "#2DE482",
            "linestyle": "-",
            "label": "Run 5966" ,
            "linewidth": 1.2
        },
    ]

    title = r"Channel 0, isotropy in air ($r=0.33$ m)"
    filename = "isotropy-air-CH0-thetaoffsets.png"

    fs = 12 # fontsize for labels on plots

    # number of scanpoints comprising 1 scan (there's probably a better way to do this, but it works for now)
    indices = np.arange(1, 181) # np.arange(1, 325)

    # intialize the target theta values the PTF was meant to slew to
    theta_centers = -1*np.arange(0, 107, 3) # -1*np.arange(0, 106, 3)

    # initialize an arrays for each storing the data for each scan
    ADCs = []
    ADC_errs = []
    final_thetas = []

    # process all of the raw data for each scan
    for data_file in data_files:
        run_num = data_file[-9:-5]
        print(f"Computing ADC integrals for run {run_num}:")
        print("-----------------------------------------")

        # initialize arrays for storing this specific scan's data
        integral_means = []
        integral_stds = []
        n_samples = []

        with h5.File(data_file, "r") as f:
            #######################################################################################
            # COMPUTE THE CORRECTED THETA VALUES (recall we have the theta offset method)
            #######################################################################################
            y = np.array(f["gantry0_y"])[1:]
            z = np.array(f["gantry0_z"])[1:]

            # initialize an array for the mean theta values at each scan point
            # NOTE: at each scan point, the box measures 10 theta values that vary by a few thenths 
            # of a degree, so we will take the average of all of them and call that our theta value.
            thetas = []
            for i in indices:
                thetas_scanpoint = f[f"zeniths/{i}"]
                thetas.append(np.mean(thetas_scanpoint))
            thetas = np.asarray(thetas)

            # compute the closest theta values to the targeted for each group of theta offsets.
            # best indices here are the theta indices which correspond to the closest to the targeted.
            """
            NOTE: I am still working on a better way of doing this, since past 87 degrees, the
            PTF becomes pretty badly shifted (like ~3-4 degrees from the actual) which complicates
            the calculation of the best theta value past 87 degrees. For now, I am just simplifying
            the calculation by setting the best theta to the targeted theta until I come up with a
            better method.
            """
            best_indices, thetas = find_best_theta(y, z, thetas, theta_centers, num_offsets=5)#9)
            final_thetas.append(thetas[best_indices])
    
            # loop through each scan point and compute  integrated ADC for each waveform at each
            # scan point
            for scan_point_index in best_indices:
                scan_point_index += 1 # add 1 since hdf5 file indexing starts at 1
                wfs = np.array(f[f"waveforms/{scan_point_index}"])
                
                # calculate the integrated ADC quantities for each waveform at this scanpoint
                # NOTE: low = 20 and high = 50 means my integration window stretches from the data point at
                # index 20 to the data point at index 50 ==> integration window is 2 ns x 30 = 60 ns wide
                t1 = time.time()
                integrated_ADCs = integrate_pulses(wfs, 20, 50, interpolate=False)
                t2 = time.time()

                print(f"Took {t2-t1:.3f} s to compute the integrals for scanpoint index {scan_point_index}")

                # Check for pulses that manage to be outside the integration window
                remove_mask = np.ones_like(integrated_ADCs)
                for i, ADC in enumerate(integrated_ADCs):
                    if ADC < 0:
                        remove_mask[i] = 0

                # clean the ADCs array
                remove_mask_bool = remove_mask.astype(bool)
                integrated_ADCs = integrated_ADCs[remove_mask_bool]

                # compute the mean and standard deviation of this ADC
                mean_ADC_integral, std_ADC_integral, n = compute_ADC_statistics(integrated_ADCs)

                # append the results of this scanpoint
                integral_means.append(mean_ADC_integral)
                integral_stds.append(std_ADC_integral)
                n_samples.append(n)

            f.close()

            # compute means of integrated ADCs at each scan point and standard errors
            integral_means = np.asarray(integral_means)
            integral_stds = np.asarray(integral_stds)
            n_samples = np.asarray(n_samples)

            # will normalize all ADC integrals with respect to this value
            max_ADC = np.max(integral_means)

            # return the normalized ADC integrals of the pulses
            ADCs.append(integral_means/max_ADC)
            ADC_errs.append(integral_stds/(max_ADC*np.sqrt(n_samples))) # standard error
            final_thetas.append(thetas[best_indices])

    
    if export_ADC_data:
        export_integrated_ADC_data(final_thetas, 
                                   ADCs, 
                                   ADC_errs,
                                   [d["label"] for d in plot_options],
                                   export_path
                                   )



    ##############################################################################################
    ### CREATE THE FINAL PLOT
    ###############################################################################################
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

    plt.figure(figsize=(7,5))

    # plot the data
    for i in range(n_files):
        thetas = final_thetas[i]
        integrated_ADCs = ADCs[i]
        integral_errors = ADC_errs[i]
        upper = integrated_ADCs + integral_errors
        lower = integrated_ADCs - integral_errors

        opts = plot_options[i]

        plt.plot(thetas, integrated_ADCs, c = opts["color"], ls=opts["linestyle"], lw=opts["linewidth"])
        plt.fill_between(thetas, lower, upper, color=opts["color"], lw=0.25, alpha=0.2)

    labels = []; handles = []

    # make the legend handles and labels
    for i in range(len(data_files)):
        opts = plot_options[i]
        labels.append(opts["label"])
        handles.append(Line2D([0], [0], color=opts["color"], lw=2, ls=opts["linestyle"]))

    # make the legend
    plt.legend(
        handles=handles,
        labels=labels,
        loc=3,  # place legend below the bbox anchor point
        frameon=False,       # optional: remove box
        fontsize=fs-1   
    )

    plt.grid(which='major',
             color="grey",
             linestyle='-',
             linewidth=0.5)

    plt.grid(which='minor',
             color="silver",
             linestyle='--',
             linewidth=0.25)


    # set the axis labels, limits, and title
    plt.xlim(-1, 106)
    plt.ylim(0, 1.05)
    plt.xlabel(r"$\theta$ [$^{\circ}$]", size=fs)
    plt.ylabel(r"Normalized integral [ADC$\times$ns]", size=fs)
    plt.title(title, size=fs)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
