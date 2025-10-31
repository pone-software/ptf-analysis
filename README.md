# ptf-analysis
## About
This python package contains code specialized analyzing data from TRIUMF's photosensor test facility
(PTF) and quickly making paper-ready plots.


## Setup
```bash
git clone git@github.com:rhall14/ptf-analysis.git
cd ptf-analysis
```

I still have not yet implemented a `pyproject.toml` file containing the necessary dependencies.
However, they are just the usual matplotlib, numpy, scipy, etc...


## Using the ptf-analysis code
This is still very much a work in progress, but my goal is to make running the code seamless once 
we get a steady flow of data from the PTF. I want the simple workflow to proceed as follows:
1. Convert the raw root files from the PTF into more workable hdf5 files via `root2hdf5.py`. I 
anticipate further customizing this script to be able to save different types of data files once
we end up doing different types of analyses (e.g., exptrapolation to far field emission)
2. Create the desired plot by simply running the relevant script. Each script is designed to (or 
will be designed to) simply call the script, while specifying a few parameters when executing.
The scripts I have worked on so far are (UPDATE THIS LIST AS I GO ON):
    1. `plot_isotropy.py`: plots the angular emission profile of the light emitted by the P-CAL. You
    can choose to supply multiple data files to plot multiple emission curves on one figure.
    2. `plot_pmt_waveforms.py`: plots the pmt waveforms for one laser channel at different bias 
    voltages for comparison as the upper plot. The lower plot consists of histograms of the 
    integrated ADC integrals.
    3. More to come :)

### Example usage of `root2hdf5.py`
Suppose the PTF root files you want to convert are in folder '/path/to/root/files/'. You can convert
them all, and store the output in '/path/to/hdf5/files' by running

```bash
python root2hdf5.py --filepath /path/to/root/files/ --destination /path/to/hdf5/files
```

Alternatively, if you just want to convert one file, you can pass in the path to that specific file
and specify the output directory:

```bash
python root2hdf5.py --filepath /path/to/root/files/out_run0001.root --destination /path/to/hdf5/files
```

Note that the hdf5 files have the same name as the root files.

### Example usage of `plot_isotropy.py`