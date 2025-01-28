r""" This file tests the correspondence between theory and simulation for a
    two-dimensional square Ising model, with exchange and dipolar interactions,
    by observing striped phases as a function of the relative strength $\delta$
    between the exchange and dipolar interaction at low temperature.
    Based on the paper
        J. H. Toloza, F. A. Tamarit, and S. A. Cannas. Aging in a two-dimensional Ising model
        with dipolar interactions. Physical Review B, 58(14):R8885, 1998.
"""
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["HOTSPICE_USE_GPU"] = "True"
import hotspice
import thesis_utils


def run(delta_range=np.linspace(0, 4, 81), N: float = 10, size: int = 800):
    """ Runs the simulation and saves it to a timestamped folder.
        `T_range` specifies the examined temperature range in multiples of T_c.
        At each step, `N` calls of `Magnets.update()` are performed.
    """
    delta_range, size = np.asarray(delta_range), int(size)
    T, a = 50, 1e-6 # A combination that is NOT in the paramagnetic state
    
    NN_corr_avg, NN_corr_std = np.empty_like(delta_range), np.empty_like(delta_range)
    states = np.empty(shape=(delta_range.size, size, size), dtype=bool) #! States saved as boolean arrays.
    
    def apply_delta(delta: float):
        energyExch.J = delta*energyDD.get_NN_interaction()/2

    ## Simulate
    energyDD, energyExch = hotspice.DipolarEnergy(), hotspice.ExchangeEnergy()
    mm = hotspice.ASI.OOP_Square(a, size, E_B=0, T=T, energies=[energyDD, energyExch], pattern='AFM', PBC=True, params=hotspice.SimParams(UPDATE_SCHEME="Metropolis"))
    for i, delta in enumerate(delta_range):
        print(f"[{i+1}/{delta_range.size}] delta = {delta:.2f}...")
        apply_delta(delta)
        NN_corrs = []
        for _ in mm.progress(t_max=np.inf, MCsteps_max=N, Q=np.inf):
            NN_corrs.append(mm.correlation_NN())
        halfway = len(NN_corrs) // 2
        NN_corr_avg[i], NN_corr_std[i] = np.mean(NN_corrs[halfway:]), np.std(NN_corrs[halfway:])
        states[i,:,:] = (hotspice.utils.asnumpy(mm.m) + 1).astype(bool)
    
    ## Save
    hotspice.utils.save_results(parameters={"size": size, "N": N, "a": a, "T": T, "PBC": mm.PBC, "scheme": mm.params.UPDATE_SCHEME},
                                data={"delta_range": delta_range, "NN_corr_avg": NN_corr_avg, "NN_corr_std": NN_corr_std, "states": states})
    plot()


def plot(data_dir=None):
    """ (Re)plots the figures in the `data_dir` folder.
        If `data_dir` is not specified, the most recently created directory in <filename>.out is used.
    """
    ## Load data
    if data_dir is None: data_dir = thesis_utils.get_last_outdir()
    params, data = hotspice.utils.load_results(data_dir)
    
    ## Plot
    thesis_utils.init_style()
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(thesis_utils.page_width, 2.5))
    ax1: plt.Axes = axes[0,0]
    # Plot the curve
    Sim1_OOP_Exchange_N_values = [10, 100, 1600]
    i = np.argmin(np.abs(np.asarray(Sim1_OOP_Exchange_N_values) - params['N'])) # Get nearest N
    ax1.errorbar(data['delta_range'], data['NN_corr_avg'], yerr=data['NN_corr_std'], color=f"C{i}", fmt=thesis_utils.marker_cycle[i], label="Hotspice NN corr")
    ax1.axvline(0.85, linestyle=':', color='black') # Theoretical transition point
    ax1.axvline(2.65, linestyle=':', color='black') # Theoretical transition point (approximate)
    ax1.set_xlabel(r"Relative exchange/MS coupling $\delta$")
    ax1.set_ylabel("NN correlation\n" + r"$\langle S_i S_{i+1} \rangle$", labelpad=-5)
    ax1.set_yticks([-1, -.5, 0, .5, 1])
    ax1.set_xlim([data['delta_range'].min()-.005, data['delta_range'].max()+.005])
    ax1.set_ylim([-1.05, 1.05])
    # Finish the axes
    for transition in [-1, 0, 0.5]: ax1.axhline(transition, linestyle=':', color='grey')
    plt.gcf().tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.2, left=0.1, right=0.97)
    
    ## Plot the states. #! This must happen after subplots_adjust to get the correct scaling for equal aspect ratio.
    def get_aspect(ax):
        figW, figH = ax.get_figure().get_size_inches() # Total figure size
        _, _, w, h = ax.get_position().bounds # Axis size on figure
        disp_ratio = (figH * h) / (figW * w) # Ratio of display units
        data_ratio = (ax.get_ylim()[0] - ax.get_ylim()[1]) / (ax.get_xlim()[0] - ax.get_xlim()[1]) # Ratio of data units
        return disp_ratio / data_ratio
    ar = get_aspect(ax1)
    cutout_magnets, cutout_size, ys = min(20, params['size']), 0.44, [-0.47, -0.53, -0.5]
    dx = cutout_size*ar if ar < 1 else cutout_size
    dy = cutout_size if ar < 1 else cutout_size/ar
    deltas = [0.35, 1.75, 3.35]
    for i, delta in enumerate(data['delta_range']):
        if not np.any(np.isclose(delta, deltas)): continue
        j = np.argmin(np.abs(deltas - delta))
        y = ys[j]
        # rx, ry = np.random.randint(params['size'] - cutout_magnets + 1), np.random.randint(params['size'] - cutout_magnets + 1)
        rx, ry = 70, 100 # Specific choice for 20240827144415 run
        cutout_data = data['states'][i][rx:rx+cutout_magnets,ry:ry+cutout_magnets]
        ax1.plot([delta, delta], [y, data['NN_corr_avg'][i]], zorder=-100, color="gray", linewidth=2)
        ax1.imshow(cutout_data, extent=[delta+dx, delta-dx, y+dy, y-dy], vmin=0, vmax=1, cmap='gray', aspect='auto', origin='lower')
        ax1.add_patch(plt.Rectangle((delta-dx, y-dy), 2*dx, 2*dy, fill=False, color="gray", linewidth=1))
    
    hotspice.utils.save_results(figures={'OOP_Dipolar': fig}, outdir=data_dir, copy_script=False)


if __name__ == "__main__":
    # run(N=10, size=200)
    thesis_utils.replot_all(plot)
