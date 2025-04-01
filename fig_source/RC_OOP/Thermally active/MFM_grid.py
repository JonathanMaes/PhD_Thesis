import matplotlib.pyplot as plt
import numpy as np
import pickle

from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.ticker import LogLocator
from pathlib import Path
from PIL import Image

import hotspice
import thesis_utils

from relaxation_BO import compare_exp_sim, get_exp_data


def SASI_to_params(S_ASI):
    file = Path(__file__).parent / Path(r"relaxation_BO.out\finite_magnets-simple_barrier") / f"{S_ASI:.0f}nm" / "data.pkl"
    with open(file, "rb") as datafile:
        return pickle.load(datafile)['best_guess']['params']

def color_ax(ax: plt.Axes, color):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
        spine.set_capstyle('round')
    ax.tick_params(axis='both', which='major', length=6, width=2, color=color)
    ax.tick_params(axis='both', which='minor', length=3, width=.5, color=color)

def show_MFM(ax: plt.Axes, MFMfile):
    img = np.asarray(Image.open(MFMfile).convert('L'))
    ax.imshow(img, cmap="grey", extent=[0,1,0,1])
    ax.set_xticks([])
    ax.set_yticks([])

def plot():
    thesis_utils.init_style('default')
    
    figsize = (thesis_utils.page_width, 6)
    fig, axes = plt.subplots(3, 3, figsize=figsize, height_ratios=[1,1,1.5])
    ax: plt.Axes
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.1, wspace=0.1, hspace=0.3)
    
    S_ASI_values = [20, 25, 30]
    col_ors = ["#128110", "#20500F", "#244010"]
    switched = [[(1,1), (3,1), (4,5), (4,8), (4,10), (8,9), (9,3), (10,0), (10,7)],
                [(0,4), (0,7), (1,6), (1,9), (3,2), (3,5), (5,8), (6,1), (7,2), (7,8), (8,6), (9,10), (10,6), (10,8)],
                [(1,4), (1,9), (2,2), (2,6), (3,0), (3,4), (3,8), (5,8), (6,4), (7,3), (7,7), (7,9), (8,10), (9,6), (10,9)]]

    for col in range(3):
        S_ASI = S_ASI_values[col]
        color = col_ors[col]
        ylabel_kwargs = dict(fontsize=plt.rcParams['axes.titlesize'], labelpad=10)
        
        ## Common commands for all rows
        for row in range(3):
            ax: plt.Axes = axes[row,col]
            ax.tick_params(axis="both", labelsize="large")
            color_ax(ax, color)
            if col != 0: ax.set_yticklabels([])

        ## Row 1: MFM half hour
        ax1: plt.Axes = axes[0,col]
        show_MFM(ax1, Path(__file__).parent / f"MFM/SASI{S_ASI_values[col]}_t1e3.png")
        ax1.set_title(r"$S_\mathrm{ASI} = %.0f\, \mathrm{nm}$" % S_ASI, pad=5, color=color)
        if col == 0:
            ax1.set_ylabel(r"$t = 1000 \, \mathrm{s}$", **ylabel_kwargs)
            thesis_utils.label_ax(ax1, 0, offset=(-0.6, 0), va="top")

        ## Row 2: MFM two years
        ax2: plt.Axes = axes[1,col]
        show_MFM(ax2, Path(__file__).parent / f"MFM/SASI{S_ASI_values[col]}_t7e7.jpg")
        factor = 1/13 # Size of one magnet in axis units
        for x, y in switched[col]:
            ax2.add_artist(Circle(((x+1.5)*factor, 1 - (y+1.5)*factor), factor/2,
                                    edgecolor="yellow", linewidth=0.5, facecolor="none"))
        if col == 0:
            ax2.set_ylabel(r"$t = 7 \times 10^7 \, \mathrm{s}$", **ylabel_kwargs)
            thesis_utils.label_ax(ax2, 1, offset=(-0.6, -1.1), va='top')
        
        ## Arrow from row 1 to 2
        center = (ax1.get_position().x0 + ax1.get_position().x1)/2
        fig.patches.append(FancyArrowPatch((center, ax1.get_position().y0), (center, ax2.get_position().y1),
                                            shrinkB=0, shrinkA=0, transform=fig.transFigure, color=color,
                                            arrowstyle='-|>', capstyle='butt', mutation_scale=20, linewidth=3))

        ## Row 3: fitted relaxation traces
        ax3: plt.Axes = axes[2,col]
        msr = 170/(170 + S_ASI)
        data = SASI_to_params(S_ASI)
        EEA, EMC, J = data['E_EA'], data['E_MC'], data['J']
        traces = compare_exp_sim(get_exp_data(S_ASI), EEA, EMC, J, n_avg=200, magnet_size_ratio=msr, plot=True, ax=ax3)

        textstr = '\n'.join([r'$%s=%.1f k_\mathrm{B}T$' % (s,p) for s,p in {r'E_\mathrm{EA}': EEA, r'E_\mathrm{MC}': EMC, 'J': J}.items()])
        ax3.text(0.97, 0.97, textstr, transform=ax3.transAxes, fontsize="medium", color=color,
                va='top', ha='right', bbox=dict(boxstyle='round', pad=.2, facecolor='#F2F2F2', edgecolor='none', alpha=0.7))
        ax3.xaxis.set_minor_locator(LogLocator(numticks=999, subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9, .99)))
        plt.setp(ax3.get_xminorticklabels(), visible=False)

    fig.supxlabel("Elapsed time [s]", fontsize=thesis_utils.fs_large, x=0.52)
    y = axes[2,1].get_position().y1
    fig.legend(traces, [l.get_label() for l, f in traces], ncols=4, bbox_to_anchor=(0.5, y), loc="lower center")

    hotspice.utils.save_results(figures={"MFM_grid": fig}, timestamped=False, copy_script=False, dpi=200)


if __name__ == "__main__":
    plot()