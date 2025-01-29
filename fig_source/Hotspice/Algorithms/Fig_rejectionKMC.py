import matplotlib.pyplot as plt
import numpy as np

import hotspice
import thesis_utils


def main_plot():
    ## Data
    xmin, xmax = -2, 2
    delta_E = np.linspace(xmin, xmax, 101)
    MH = np.where(delta_E < 0, 1, np.exp(-delta_E))
    GD = np.exp(-delta_E)/(1+np.exp(-delta_E))
    
    ## Main axes
    thesis_utils.init_style()
    fig, axes = plt.subplots(1, 1, squeeze=False, figsize=(thesis_utils.page_width*0.5, 1.7))
    ax: plt.Axes = axes[0,0]
    ax.plot(delta_E, MH, label="Metropolis-Hastings")
    ax.plot(delta_E, GD, label="Glauber dynamics")
    ax.set_xlabel(r"$\Delta E/k_\mathrm{B}T$")
    ax.set_ylabel("Acceptance\nprobability " + r"$P(\Delta E)$")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-.1, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(ncol=1, loc="lower left", fontsize=9, borderaxespad=0.2)

    ## Finish plot
    fig.subplots_adjust(top=.95, bottom=0.25, left=0.21, right=0.98)
    hotspice.utils.save_results(figures={'RejectionKMC': fig}, timestamped=False)


if __name__ == "__main__":
    main_plot()
