"""
Primary entry point script to the 'isotropy' command. Meant to deal with different tasks one would
like to perform with regards to isotropy (e.g., plotting, exporting the data, interpolating, etc...)
"""

import h5py as h5 
import numpy as np
from ..aux import *
from . import export_data, plot
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Converts raw PTF root files into concise hdf5 files.")

    parser.add_argument('-f', '--files', 
                        required=True, 
                        type=str, 
                        help='Path from current directory to either a directory containing all of the '
                        'data files you wish to convert, or directly specifying a path to 1 hdf5 file.')
    
    parser.add_argument('-tr', '--theta_range',
                        required=True,
                        type=float,
                        nargs=3,
                        metavar=('min_theta', 'max_theta', 'dtheta'),
                        help="Specify 'min_theta', 'max_theta', and 'dtheta' for the target thetas that " \
                        "optical box was supposed to slew to.")

    parser.add_argument('-oi', '--offset_info',
                        required=True,
                        type=float,
                        nargs=2,
                        metavar=('num_offsets', 'dtheta'),
                        default=[0.0, 0.0],
                        help="Specify the number of theta offsets per target theta, 'num_offsets', "
                        "and the width between each theta offset, 'dtheta'.")

    parser.add_argument('-e', '--export',
                        required=False,
                        type=bool,
                        default=False,
                        help='Optional request to export the isotropy data for each file supplied. ' \
                        'Outputs hdf5 files in a new folder next to the folder containing the original ' \
                        'files.')
    
    parser.add_argument('-p', '--plot',
                        type=bool,
                        required=False,
                        default=False,
                        help='Optional request to plot the angular emission profiles for the data files ' \
                        'requested. Outputs a plot with default labels, colors, and names to the parent ' \
                        'directory containing the original datafiles.'
                        )

    parser.add_argument('-po', '--plot_options',
                        type=str,
                        required=False,
                        help='Path to an optional yaml file for specifying various settings on the plot '
                        '(e.g., labels, colors, fontsizes, etc.). If not supplied, defaults are taken.')
    
    args = parser.parse_args()

    tr = args.theta_range
    theta_centers = -1*np.arange(tr[0], tr[1]+tr[2], tr[2])

    oi = args.offset_info
    offset_span = (oi[0]-1) * oi[1]


    ###############################################################################################
    ### LOOP THROUGH EACH FILE AND COMPUTE THE ANGULAR EMISSION PROFILES FOR EACH SCAN
    ###############################################################################################
    # initialize an arrays for each storing the data for each scan
    ADCs = []
    ADC_errs = []
    final_thetas = []
    run_nums = []

    # NOTE: can likely make this cleaner with pathlib library
    # if just one file path was supplied, append the lone file to the list
    if args.files[-5:] == ".hdf5":
        file_paths = [args.files]

    # store each file in a list of files
    else:
        file_paths = []
        for f in os.listdir(args.files):
            if f[-5:] == ".hdf5":
                file_paths.append(os.path.join(args.files, f))


    # process each data file
    for path in file_paths:
        run_num = path[-9:-5] # run number for this data file
        run_nums.append(run_num)

        print(f"Computing ADC integrals for run {run_num}:")
        print("-------------------------------------")
        t1 = time.time()

        # initialize arrays for storing this specific scan's data
        integral_means = []
        integral_stds = []
        n_samples = []

        with h5.File(path, "r") as f:
            indices = np.arange(1, len(f["waveforms"])+1, 1)

            #--------------------------------------------------------------------------------------
            # COMPUTE THE CORRECTED THETA VALUES (recall we have the theta offset method)
            #--------------------------------------------------------------------------------------
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
            best_indices, thetas = find_best_theta(y, z, thetas, theta_centers, 
                                                   num_offsets=int(oi[0]), width=offset_span)
            final_thetas.append(thetas[best_indices])

            # loop through each scan point and compute  integrated ADC for each waveform at each
            # scan point
            for scan_point_index in best_indices:
                scan_point_index += 1 # add 1 since hdf5 file indexing starts at 1
                wfs = np.array(f[f"waveforms/{scan_point_index}"])
                
                # calculate the integrated ADC quantities for each waveform at this scanpoint
                # NOTE: low = 20 and high = 50 means my integration window stretches from the data point at
                # index 20 to the data point at index 50 ==> integration window is 2 ns x 30 = 60 ns wide
                integrated_ADCs = integrate_pulses(wfs, 20, 50, interpolate=False)

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

            t2 = time.time()
            print(f"Took {t2-t1:.4f} s to compute angluar emission profile for run {run_num}.\n")

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



    ###############################################################################################
    ### EXPORT DATA IF REQUESTED
    ###############################################################################################
    if args.export:
        print("Export of angular emission profiles to hdf5 file requested:")

        # get the parent directory the input data files are located in and create a path to the
        # new files
        export_dir = get_dir_path(args.files)
        export_path = export_dir / "output" / (export_dir.name + "_emission_profiles.hdf5")

        # Make sure the parent directory exists
        export_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"File stored at '{export_path}'\n")

        export_data.export_integrated_ADC_data(
            final_thetas, 
            ADCs, 
            ADC_errs,
            ["run " + num for num in run_nums],
            export_path
        )

    else:
        print("Angular emission profiles export to hdf5 file not requested, skipping...\n")



    ###############################################################################################
    ### PLOT THE ISOTROPY RESULTS IF REQUESTED
    ###############################################################################################
    if args.plot:
        # place plot in the output directory
        export_dir = get_dir_path(args.files)
        plot_path = export_dir / "output" / (export_dir.name + "isotropy_profile.png")

        # Make sure the parent directory exists
        export_path.parent.mkdir(parents=True, exist_ok=True)

        print(final_thetas)

        print("Plotting the angular emission profiles...")
        plot.plot_emission_profiles(final_thetas,
                                    ADCs,
                                    ADC_errs,
                                    ["Run " + num for num in run_nums],
                                    filename=plot_path
                                    )



if __name__ == "__main__":
    main()
