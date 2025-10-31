import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import uproot
import h5py as h5
from aux import correct_wf
import argparse
import os


# the relevant pieces of information in the raw PTF root files
necessary_keys = [
    "gantry0_x",
    "gantry0_y",
    "gantry0_z",
    "gantry0_rot",
    "phidg0_tilt",
    "V1730_wave2",
]

# minimum ADC level (after correction) to determine a peak
min_ADC = 20


def root_to_hdf5(root_file_path, hdf5_file_path, low = 20, high = 50):
    """
    Uses uproot to to convert raw PTF root files to hdf5 files
    """
    # Open the ROOT file using uproot
    out_file = h5.File(hdf5_file_path, 'w')
    
    with uproot.open(root_file_path) as root_file:
        # Create an HDF5 file
        tree = root_file["scan_tree;1"]

        # gather only the necessary data
        for key in necessary_keys:
            # collect the waveform data and remove any events without photon hits
            if ("V1730" in key):
                # create a folder for all of the waveforms
                waveforms_group = out_file.create_group("waveforms")

                waveforms = tree[key].array(library='np')[1:]

                for i in range(len(waveforms)):
                    # the waveforms taken at this scan point
                    scanpoint_wfs = waveforms[i]

                    scanpoint_wfs_with_photons = []

                    # address each waveform at this specific scan point
                    for j in range(len(scanpoint_wfs)):
                        wf = scanpoint_wfs[j]

                        # NOTE: this is an out-of-date version, but I want to keep it just in case
                        """
                        # estimate the mean value of the noise
                        # ------------------------------------
                        # initialize an array of all 1s (which represent True)
                        wf_nopeak = np.ones_like(wf)

                        # replace window with peak with 0s (which represent False)
                        wf_nopeak[low:high] = 0

                        # boolean mask for removing integration window
                        # wf_nopeak_bool = wf_nopeak.astype(bool)
                        wf_nopeak_bool = wf_nopeak.astype(np.bool_)
                        wf_noise = wf[wf_nopeak_bool]

                        # calculate the mean value of the noise
                        mean_noise = np.mean(wf_noise)

                        # invert and add the mean to have the noise at roughly 0 ADC 
                        # and the peak going to positive values
                        wf_cleaned = -1*wf+ mean_noise
                        """

                        # compute the corrected waveform value
                        wf_cleaned = correct_wf(wf)

                        # determine if the waveform has a peak (corresponding to photon hits)
                        peaks, _ = find_peaks(wf_cleaned[low:high], height=min_ADC, distance = 4)

                        # only append the waveform if it has a peak
                        if len(peaks) > 0:
                            scanpoint_wfs_with_photons.append(wf)

                    scanpoint_wfs_with_photons = np.asarray(scanpoint_wfs_with_photons)

                    waveforms_group.create_dataset(str(i+1), data=scanpoint_wfs_with_photons)

            # compute the zeniths at each scan point (average of the zeniths measured at a scan point)
            elif "tilt" in key:
                tilt_group = out_file.create_group("zeniths")

                tilts = tree[key].array(library='np')[1:]

                for i in range(len(tilts)):
                    scanpoint_tilts = tilts[i]

                    tilt_group.create_dataset(str(i+1), data=scanpoint_tilts)

            # save the coordinate keys
            else:
                branch_data = tree[key].array(library="np")
                out_file.create_dataset(key, data=branch_data)

    out_file.close()


def main():
    parser = argparse.ArgumentParser(description="Converts raw PTF root files into concise hdf5 files.")

    parser.add_argument('-fp', '--filepath', 
                        type=str, 
                        required=True, 
                        help='Path from current directory to either a directory containing all of the '
                        'data files you wish to convert, or directly specifying a path to 1 root file.')
    
    parser.add_argument('-d', '--destination',
                        type=str,
                        required=True,
                        help='Path to the directory where the resulting hdf5 files will go to.')
    
    args = parser.parse_args()
    

    # if just one file path was supplied, append the lone file to the list
    if args.filepath[-5:] == ".root":
        file_paths = [args.filepath]

    # store each file in a list of files
    else:
        file_paths = []
        for f in os.listdir(args.filepath):
            if f[-5:] == ".root":
                file_paths.append(os.path.join(args.filepath, f))


    # convert each root file to an hdf5 file
    for root_file_path in file_paths:
        print(f"Converting {root_file_path}...")

        run_num = root_file_path[-10:-5]
        
        hdf5_file_name = "out_run" + run_num + ".hdf5"
        hdf5_file_path = os.path.join(args.destination, hdf5_file_name)

        root_to_hdf5(root_file_path, hdf5_file_path)

        print(f"Done! Output file located at {hdf5_file_path}...\n")


if __name__ == "__main__":
    main()
