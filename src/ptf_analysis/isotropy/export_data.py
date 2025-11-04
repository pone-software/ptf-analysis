"""
Module containing functionality for exporting angular emission data to a file. Can perform some more 
specific plotting with the exported file without processing all of the data again. Can add more here 
later if necessary.
"""

import h5py as h5 
import numpy as np
from ..aux import *


def export_integrated_ADC_data(thetas, integrated_ADCs, integrated_ADC_errs, dataset_names, out_file):
    """
    After the angular emission profiles for each dataset have been calculated, this optional function
    stores the trimmed data in a short root file for slimmer storage of the relevant data.
    """
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
