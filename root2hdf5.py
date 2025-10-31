import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import uproot
import h5py as h5
from aux import correct_wf


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

root_file_path = "../data/out_run05966.root"
hdf5_file_path = "hdf5_files/CH0_air/out_run05966.hdf5"


def root_to_hdf5(root_file_path, hdf5_file_path, low = 20, high = 50):
    """
    Uses uproot to rid ourselves of root.
    """
    # Open the ROOT file using uproot
    out_file = h5.File(hdf5_file_path, 'w')
    
    with uproot.open(root_file_path) as root_file:
        # Create an HDF5 file
        tree = root_file["scan_tree;1"]

        for key in necessary_keys:

            if ("V1730" in key):
                # create a folder for all of the waveforms
                waveforms_group = out_file.create_group("waveforms")

                waveforms = tree[key].array(library='np')[1:]

                for i in range(len(waveforms)):
                    # the waveforms taken at this scan point
                    scanpoint_wfs = waveforms[i]

                    scanpoint_wfs_with_photons = []

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

                        wf_cleaned = correct_wf(wf)

                        # determine if the waveform has a peak (corresponding to photon hits)
                        peaks, _ = find_peaks(wf_cleaned[low:high], height=min_ADC, distance = 4)

                        # only append the waveform if it has a peak
                        if len(peaks) > 0:
                            scanpoint_wfs_with_photons.append(wf)

                    scanpoint_wfs_with_photons = np.asarray(scanpoint_wfs_with_photons)

                    waveforms_group.create_dataset(str(i+1), data=scanpoint_wfs_with_photons)

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


if __name__ == "__main__":
    root_to_hdf5(root_file_path, hdf5_file_path)
