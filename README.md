# ptf-analysis
## About
This python package contains code specialized analyzing data from TRIUMF's photosensor test facility
(PTF) and quickly making paper-ready plots.


## Setup
```bash
git clone git@github.com:rhall14/ptf-analysis.git
cd ptf-analysis
```

I am thinking of making this into an actual package with executable commands since I think it's good
to practice that. Once inside the cloned directory run 

```bash
pip install -e .
```

for an editable install of the package. That way, if you want to make an edit to any of the existing 
commands, the changes are immediately reflected in the executable commands. Feel free to add any new 
commands for any additional functionality you want to implement.


## Using the ptf-analysis code
This is still very much a work in progress, but my goal is to make running the code seamless once 
we get a steady flow of data from the PTF. I want the simple workflow to proceed as follows:
1. Convert the raw root files from the PTF into more workable hdf5 files via `root2hdf5`. I 
anticipate further customizing this script to be able to save different types of data files once
we end up doing different types of analyses (e.g., exptrapolation to far field emission)
2. Create the desired plot by simply running the relevant command. Each command has (or will 
eventually have) optional flags for requesting different tasks. These are the following commands
I have made or plan to implement:
    1. `isotropy`: plots the angular emission profile of the light emitted by the P-CAL. You
    can choose to supply multiple data files to plot multiple emission curves on one figure.
    2. `pmt_waveforms`: plots the pmt waveforms for one laser channel at different bias 
    voltages for comparison as the upper plot. The lower plot consists of histograms of the 
    integrated ADC integrals. (work-in-progress)
    3. `farfield`: not working on this yet as I want to prioritize getting the angular emission
    measurements out first

### Example usage of `root2hdf5`
Suppose the PTF root files you want to convert are in folder '/path/to/root/files/'. You can convert
them all, and store the output in directory '/path/to/hdf5/files' by running

```bash
root2hdf5 --filepath /path/to/root/files/ --destination /path/to/hdf5/files
```

Alternatively, if you just want to convert one file, you can pass in the path to that specific file
and specify the output directory:

```bash
root2hdf5 --filepath /path/to/root/files/out_run0001.root --destination /path/to/hdf5/files
```

Note that the hdf5 files have the same name as the root files. Also, the code for this script lives in
ptf_analysis/root2hdf5.py

### Example usage of `isotropy`
This command plots the angular emission profiles. So far, I have implemented the options to export the 
angular emission profiles in a separate hdf5 file, and also make a plot of the profiles. If your hdf5 files live in '/path/to/hdf5/files', you can run 

```bash
isotropy --files /path/to/hdf5/files --theta_range 0 105 3 --offset_info 5 1 --export True --plot True
```

Here, `--theta_range 0 105 3` means the minimum target theta is 0 degrees, the maximum target theta is 105 degrees, and the spacing between each target theta is 3 degrees. Also, `--offset_info 5 1` means at each target theta, say t1, the optical box swept through 5 theta offsets at 1 degree spacing, meaning it scanned through the following theta values (t1-2, t1-1, t1, t1+1, t1+2). This is necessary due to the PTF's poor zenith tilt accuracy. The closest theta offset to the target theta is taken as our data point.

If exporting and plotting was requested, the outputs go inside a new directory called outputs (i.e., in /path/to/hdf5/files/output).

For Thomas: The code is still a tad messy and I have some unfinished pieces, so if you see anything that's an easy fix or optimization, go for it. I also added some comments in a file called `plot_data_sims.py`, which is essentially where I want to add a new plotting script.