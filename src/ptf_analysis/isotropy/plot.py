"""
Python script to plot isotropy. Can supply multiple data files to plot multiple angular emission profiles.
"""

import h5py as h5 
import numpy as np
from ..aux import *
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import time
from numba import njit


# DEFAULT PLOT SETTINGS
# plot options for each line plotted, ensure the same order as the data files order
plot_options = [
    {
        "color": "#00284C",
        "linestyle": "-",
        "label": "Run 5963" ,
        "linewidth": 1.2
    },
    {
        "color": "#00098DFF",
        "linestyle": "-",
        "label": "Run 5964" ,
        "linewidth": 1.2
    },
    {
        "color": "#0056CE",
        "linestyle": "-",
        "label": "Run 5965" ,
        "linewidth": 1.2
    },
    {
        "color": "#2EB5EF",
        "linestyle": "-",
        "label": "Run 5966" ,
        "linewidth": 1.2
    },
]

title = r"Channel 0, isotropy in air ($r=0.33$ m)"
filename = "isotropy-air-CH0-thetaoffsets.png"

fs = 12 # fontsize for labels on plots

def plot_emission_profiles(thetas, profiles, error, labels, filename: str = "isotropy_profile.png", 
                           settings: dict | None = None):
    """
    Generates a quick initial plot of the angular emission profiles of the P-CAL, for a set
    of data files. The error band is from +/- 1 sigma bands among all of the integrated ADCs 
    measured at each scan point.
    """
    # simple plot settings to set first
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

    # number of emission profiles to plot
    n = len(thetas)
    print(n)

    # plot the data
    for i in range(n):
        theta = thetas[i]
        integrated_ADCs = profiles[i]
        upper = integrated_ADCs + error[i]
        lower = integrated_ADCs - error[i]

        opts = plot_options[i]

        plt.plot(np.cos(theta*(np.pi/180)), integrated_ADCs, c = opts["color"], ls='-', lw=1.2)
        plt.fill_between(np.cos(theta*(np.pi/180)), lower, upper, color=opts["color"], lw=0.25, alpha=0.2)

    handles = []

    # make the legend handles and labels
    for i in range(n):
        opts = plot_options[i]
        handles.append((
            Line2D([0], [0], color=opts["color"], lw=2, ls=opts["linestyle"]),
            Patch(color=opts["color"], alpha=0.2)
        ))

    # make the legend
    plt.legend(
        handles=handles,
        labels=labels,
        loc=4,  # place legend below the bbox anchor point
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
    plt.xlim(-0.25, 1.0)
    plt.ylim(0, 1.05)
    plt.xlabel(r"$\mathrm{cos}(\theta)$", size=fs)
    plt.ylabel(r"Normalized integral [ADC$\times$ns]", size=fs)
    plt.title(title, size=fs)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

