import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from scipy.ndimage.filters import gaussian_filter

import hotspice
import thesis_utils

from thermally_active_ASI import get_thermal_mm


def run(t_max=1e3, samples=5, m_avgs=None, order_parameters=None):
    if m_avgs is None: m_avgs = []
    if order_parameters is None: order_parameters = []
    
    ## SWEEP PARAMETERS
    E_MC_range = np.linspace(0, 40, 81)
    E_EA_range = np.linspace(0, 80, 81)
    samples = samples # The number of relaxations to run for each combination of (E_EA, E_MC)
    MCsweeps = 10 # After <MCsweeps*mm.n> Néel switching attempts, the decay is stopped.
    
    ## SYSTEM PARAMETERS
    E_B_std = 0.05
    size = 11
    
    ## RUN SIMULATIONS
    q_NN = np.empty((E_MC_range.size, E_EA_range.size, samples))
    m_avg = np.empty_like(q_NN)
    for i, E_MC in enumerate(E_MC_range):
        for j, E_EA in enumerate(E_EA_range):
            print(i, j)
            for s in range(int(samples)):
                mm: hotspice.ASI.OOP_Square = get_thermal_mm(E_EA_ratio=E_EA, E_MC_ratio=E_MC, pattern='uniform', E_B_std=E_B_std, size=size)
                mm.progress(t_max=t_max, MCsteps_max=MCsweeps)
                q_NN[i,j,s] = (1 - mm.correlation_NN())/2
                m_avg[i,j,s] = mm.m_avg
    
    q_NN = q_NN.mean(axis=2)
    m_avg = m_avg.mean(axis=2)

    ## Save
    hotspice.utils.save_results(
        parameters={"E_MC_range": E_MC_range, "E_EA_range": E_EA_range, "t_max": t_max},
        data={"m_avg": m_avg, "q_NN": q_NN}
    )
    plot()



def plot(data_dir=None):
    """ (Re)plots the figures in the `data_dir` folder.
        If `data_dir` is not specified, the most recently created directory in <filename>.out is used.
    """
    ## Load data
    if data_dir is None: data_dir = thesis_utils.get_last_outdir()
    params, data = hotspice.utils.load_results(data_dir)
    
    E_MC_range, E_EA_range, t_max = np.asarray(params["E_MC_range"]), np.asarray(params["E_EA_range"]), params["t_max"]
    m_avg, q_NN = data["m_avg"], data["q_NN"]
    
    ## Initialise plot
    thesis_utils.init_style(style="default")

    ## FIGURE
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(thesis_utils.page_width, thesis_utils.page_width/2), constrained_layout=True, sharey=True)
    axes: list[plt.Axes]
    ax1, ax2 = axes
    cmap = plt.get_cmap('inferno')
    E_MC_mesh, E_EA_mesh = np.meshgrid(E_MC_range, E_EA_range)
    
    markers = [(3, 70), (17, 65), (30, 30), (10, 10), (-2, 20)] # (E_MC, E_EA) combinations [kBT]
    marker_colors = [('black', 'white'), ('white', 'white'), ('white', 'black'), ('white', 'black'), ('black', 'black')] # (m_avg, q_NN) combinations
    marker_labels = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ', 'Ⅴ']
    for a, (ax, val, level, color) in enumerate([(ax1, m_avg, 0.05, 'white'), (ax2, q_NN, 0.95, 'black')]):
        # Phase space heatmap
        pmesh = ax.pcolormesh(E_MC_mesh, E_EA_mesh, val.T, vmin=0, vmax=1, shading='gouraud', cmap=cmap, rasterized=False)
        ax.grid(which='both', axis='both', alpha=0.2, color='#444')
        if a == 0: cb = fig.colorbar(pmesh, ax=axes, location='right', aspect=10)
        
        # Contour
        smoothed = gaussian_filter(val, 1)
        ax.contour(E_MC_mesh, E_EA_mesh, smoothed.T, [level], colors=color, linestyles='dotted')
        cb.ax.axhline(level, color=color, linestyle='dotted')
        
        # Box around region V
        y = np.log(t_max/1e-10)
        ax.add_artist(Rectangle((0,0), 1.5, y, facecolor='none', edgecolor=color))

        # Markers
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(40))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        for i, marker in enumerate(markers):
            ax.annotate(marker_labels[i], marker, color=marker_colors[i][a], va='center_baseline', ha='center',
                        weight='bold', fontproperties='serif', fontsize=14, annotation_clip=False)

        # Fitted MFM parameter markers
        fit_E_EA = [58.7, 66.5, 63.1]
        fit_E_MC = [21.0, 9.8, 9.6]
        fit_symbols = ['o', 'x', '^']
        for EEA, EMC, symbol in zip(fit_E_EA, fit_E_MC, fit_symbols):
            ax.scatter(EMC, EEA, s=40, marker=symbol, facecolor='white', edgecolor='black', linewidth=1, zorder=100)
        
        # 9 panels markers
        E_MC_ratios = [1.25, 2.5, 10]
        E_EA_ratios = [40, 30, 20]
        for i, EMC in enumerate(E_MC_ratios):
            dEMC = [-.5,.7,0][i]
            for j, EEA in enumerate(E_EA_ratios):
                scatter_kwargs = dict(facecolor='grey', zorder=100)
                ax.scatter(EMC+dEMC, EEA+4, s=30, marker=f"${1 + i + j*len(E_EA_ratios):d}$", linewidth=0.8, **scatter_kwargs)
                ax.scatter(EMC, EEA, s=20, marker="+", linewidth=1.414, **scatter_kwargs)

    fs_labels = 11
    fig.supxlabel(r"NN magnetostatic coupling $E_\mathrm{MC}/k_\mathrm{B}T$", fontsize=fs_labels, x=0.485) # To put subplots in exactly the same place as for fig1
    ax1.set_ylabel(r"Net OOP anisotropy $E_\mathrm{EA}/k_\mathrm{B}T$", fontsize=fs_labels)
    ax1.set_title(r"$m_\mathrm{avg}$" + "\naverage magnetisation")
    ax2.set_title(r"$q_\mathrm{NN}$" + "\nlocal AFM parameter")
    ax2.yaxis.set_tick_params(labelleft=False)
    
    ## SAVE PLOT
    hotspice.utils.save_results(figures={"OOP_relaxation_continuous": fig}, outdir=data_dir, copy_script=False, figure_format=('.pdf', '.png'))


if __name__ == "__main__":
    # run(t_max=10000, samples=10)
    thesis_utils.replot_all(plot)
    # plot(thesis_utils.get_last_outdir())
